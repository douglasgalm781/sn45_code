# The MIT License (MIT)
# Copyright © 2024 Yuma Rao
# Copyright © 2023 Opentensor Foundation
# Copyright © 2024 Macrocosmos
# Copyright © 2024 Broke


# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import time
import numpy as np
from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Any, Union, Dict


class RewardModelTypeEnum(Enum):
    WEIGHTED_REWARD = "reward"
    FILTER_REWARD = "filter"
    PENALTY = "penalty"


@dataclass
class RewardEvent(ABC):
    """Contains rewards for all the responses in a batch"""

    model_name: str
    rewards: Any
    rewards_normalized: Any
    timings: Any
    model_type: RewardModelTypeEnum
    batch_time: float
    extra_info: dict

    # implement custom asdict to return a dict with the same keys as the dataclass using the model name
    def asdict(self) -> dict:
        return {
            f"{self.model_name}_raw_{self.model_type.value}": self.rewards.tolist(),
            f"{self.model_name}_{self.model_type.value}": self.rewards_normalized,
            f"{self.model_name}_{self.model_type.value}_timings": self.timings,
            f"{self.model_name}_{self.model_type.value}_batch_time": self.batch_time,
            f"{self.model_name}_{self.model_type.value}_extra_info": self.extra_info,
        }


class RewardResult:
    def __init__(self, reward_pipeline, task, response_event, device):
        """Passes the responses through the reward models and calculates the total reward

        Args:
            reward_pipeline (RewardPipeline): List of all loaded/ative reward models
            task (Task): Task instance which contains reward_definition (list of reward model requirements) and a reference answer (str)
            response_event (DendriteResponseEvent): Network responses to the prompt
            device (str): Device to run the reward models on
        """
        self.reward_pipeline = reward_pipeline
        self.task = task
        self.response_event = response_event
        self.device = device
        self.task_rewards = task.reward_definition
        self.task_penalties = task.penalty_definition
        self.reward_events = self.reward_responses(
            reference=task.reference,
            models=self.task_rewards,
            reward_type=RewardModelTypeEnum.WEIGHTED_REWARD,
            task=task,
        )
        self.penalty_events = self.reward_responses(
            reference=task.reference,
            models=self.task_penalties,
            reward_type=RewardModelTypeEnum.PENALTY,
            task=task,
        )
        self.rewards = self.total_reward()

    def __state_dict__(self):
        state = {
            "rewards": self.rewards.tolist(),
            "highest_reward": self.rewards.max(),
            "average_reward": self.rewards.mean(),
            "task_name": self.task.name,
        }
        for event in self.reward_events + self.penalty_events:
            state.update(event.asdict())
        return state

    def reward_responses(
        self,
        reference: Union[str, List[str], Dict],
        models: List[dict],
        reward_type: RewardModelTypeEnum,
        task,
    ) -> List[RewardEvent]:
        """Calculates the rewards for the responses given the task and returns a RewardEvent for each reward model
        reward_events: List[RewardEvent] = [
            RewardEvent(model_name='rouge', rewards=torch.zeros(50), timings=torch.zeros(50), ...),
            RewardEvent(model_name='relevance', rewards=torch.zeros(50), timings=torch.zeros(50), ...),
        ]
        """
        reward_events = []
        ref = reference
        for reward_info in models:
            # Select the reward model from preloaded reward model pipeline
            reward_model = self.reward_pipeline.get(reward_info["name"])
            if not reward_model:
                raise ValueError(
                    f"Reward model {reward_info['name']} not supported. Please choose from {self.reward_pipeline.keys()}"
                )
            if isinstance(reference, dict):
                ref = reference.get(reward_info["name"])

            if reward_model == "self":
                reward_event = self.task.reward_apply(
                    self.response_event, reward_type=reward_type
                )
            else:
                # Compute the rewards for the responses given the prompt
                reward_event = reward_model.apply(
                    ref, self.response_event, reward_type=reward_type, task=task
                )
            reward_events.append(reward_event)

        return reward_events

    def total_reward(self):
        """Combines the rewards from all the reward models into a single reward tensor"""
        # Compute the rewards for the responses given the prompt
        rewards = np.zeros_like(self.response_event.uids, dtype=np.float64)
        for event in self.reward_events:
            for reward_info in filter(
                lambda x: x["name"] == event.model_name, self.task_rewards
            ):
                rewards += reward_info["weight"] * event.rewards

        for event in self.penalty_events:
            for reward_info in filter(
                lambda x: x["name"] == event.model_name, self.task_penalties
            ):
                rewards *= 1 - reward_info["weight"] * event.rewards

        return rewards

    def __str__(self):
        return f"{self.__class__.__name__}(rewards={self.rewards!r}, reward_events={self.reward_events!r}, penalty_events={self.penalty_events!r})"


@dataclass
class BatchRewardOutput:
    rewards: Any
    timings: Any
    extra_info: dict

    def __post_init__(self):
        self.rewards = np.asarray(self.rewards)
        self.timings = np.asarray(self.timings)
        if self.rewards.shape != self.timings.shape:
            raise ValueError(
                f"rewards.shape {self.rewards.shape} != timings.shape {self.timings.shape}"
            )

        self.rewards_normalized = (self.rewards - self.rewards.min()) / (
            self.rewards.max() - self.rewards.min() + 1e-6
        )
        self.rewards_normalized = self.rewards_normalized.tolist()


class BaseRewardModel(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def reward(self, reference: str, completions: List[str]) -> BatchRewardOutput:
        pass

    def apply(self, reference: str, response_event, reward_type, task) -> RewardEvent:
        t0 = time.time()
        if self.name == "speed":
            batch_rewards_output = self.reward(response_event.timings)
        elif self.name == "validcode":
            if "<|fim_hole|>" in task.query:
                batch_rewards_output = self.reward(
                    task.context.content,
                    [
                        task.query.replace("<|fim_hole|>", completion)
                        for completion in response_event.completions
                    ],
                    task.context.topic,
                )
            else:
                batch_rewards_output = self.reward(
                    task.context.content, response_event.completions, task.context.topic
                )
        # elif self.name == "debugrun": #TODO remove
        # batch_rewards_output = self.reward(task, response_event)
        else:
            batch_rewards_output = self.reward(reference, response_event.completions)
        batch_rewards_time = time.time() - t0

        return RewardEvent(
            model_name=self.name,
            rewards=batch_rewards_output.rewards,
            rewards_normalized=batch_rewards_output.rewards_normalized,
            model_type=reward_type,
            batch_time=batch_rewards_time,
            extra_info=batch_rewards_output.extra_info,
            timings=batch_rewards_output.timings,
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"
