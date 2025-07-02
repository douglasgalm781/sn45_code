import math
import time
from typing import List
from .reward import (
    BaseRewardModel,
    BatchRewardOutput,
)


class SpeedModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return "speed"

    def __init__(self, ideal_time: float = 12.0, decay_rate: float = 1.0, **kwargs):
        super().__init__()
        self.ideal_time = ideal_time
        self.decay_rate = decay_rate

    def score_time(self, time_taken: float) -> float:
        """
        Calculates a score from 0 to 1 based on how fast an event occurs.
        The score decreases exponentially as the time taken increases beyond the ideal time.

        :param time_taken: Time taken for the event in seconds.
        :param ideal_time: Ideal time for the event in seconds.
        :return: Score between 0 and 1.
        """
        if time_taken <= 0 or self.ideal_time <= 0:
            raise ValueError("Time taken and ideal time must be positive values.")

        # Calculate the score using an exponential decay function
        score = math.exp(
            -self.decay_rate * (time_taken - self.ideal_time) / self.ideal_time
        )

        # Ensure the score is between 0 and 1
        return max(0, min(1, score))

    def reward(self, times) -> BatchRewardOutput:
        """Get the score between two strings."""

        rewards = []
        timings = []

        for time_taken in times:
            t0 = time.time()
            rewards.append(self.score_time(time_taken))
            timings.append(time.time() - t0)

        output = BatchRewardOutput(
            rewards=rewards,
            timings=timings,
            extra_info={"ideal_time": self.ideal_time},
        )

        return output
