# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
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
import dotenv

dotenv.load_dotenv()
import os
import sys
import time
import random
import docker
import asyncio
import threading
import traceback
from time import sleep
import bittensor as bt
from typing import Awaitable, Tuple
from code_bert_score import BERTScorer
from langchain_openai import ChatOpenAI
from concurrent.futures import ThreadPoolExecutor

from coding.utils.logging import clean_wandb
from coding.validator import forward
from coding.rewards.pipeline import RewardPipeline
from coding.protocol import ScoresSynapse

# import base validator class which takes care of most of the boilerplate
from coding.finetune.model import ModelStore
from coding.utils.config import config as util_config
from coding.base.validator import BaseValidatorNeuron
from coding.finetune.dockerutil import test_docker_container
from coding.helpers.containers import DockerServer, test_docker_server
from coding.utils.logging import init_wandb_if_not_exists
from coding.finetune.llm.manager import LLMManager

class Validator(BaseValidatorNeuron):
    """
    Your validator neuron class. You should use this class to define your validator's behavior. In particular, you should replace the forward function with your own logic.

    This class inherits from the BaseValidatorNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a validator such as keeping a moving average of the scores of the miners and using them to set weights at the end of each epoch. Additionally, the scores are reset for new hotkeys at the end of each epoch.
    """

    def __init__(self, config=None):
        if not config:
            config = util_config(self)
        self.finetune_results = {}
        super(Validator, self).__init__(config=config)

        self.last_task_update = 0
        self.last_wandb_clean = self.block
        self.last_model_clear = 0
        bt.logging.info("load_state()")
        self.load_state()
        if self.last_task_update == 0:
            self.last_task_update = self.block
        init_wandb_if_not_exists(self)
        # self.active_tasks = [
        #     task
        #     for task, p in zip(
        #         self.config.neuron.tasks, self.config.neuron.task_weights
        #     )
        #     if p > 0
        # ]
        self.executor = ThreadPoolExecutor()
        self.llm_manager = LLMManager()
        self.llm_manager.init_key(os.getenv("OPENROUTER_API_KEY"))
        # Load the reward pipeline
        # self.reward_pipeline = RewardPipeline(
            # selected_tasks=self.active_tasks,
            # device=self.device,
            # code_scorer=None,
        # )
        self.docker_server = DockerServer(
            remote_host_url=os.getenv("REMOTE_DOCKER_HOST"),
            remote_host_registry=f"{os.getenv('DOCKER_HOST_IP')}:5000"
        )
        try:
            self.docker_server.remote.run("registry:2", name="swe-registry", network_mode="host")
        except Exception as e:
            bt.logging.error(f"Error running registry: {e}")
            print(traceback.format_exc())
        test_result = test_docker_container(os.getenv("REMOTE_DOCKER_HOST"))
        if not test_result:
            bt.logging.error("Docker container test failed, exiting.")
            sys.exit(1)
        while True:
            docker_server_test = test_docker_server()
            if docker_server_test:
                break
            bt.logging.error("Docker server test failed, waiting 3 minutes and trying again.")
            sleep(60*3)
        self.model_store = ModelStore(config=self.config)
        self.model_store.load()
        self.llm_manager.clear_cost()

    def _forward(
        self, synapse: ScoresSynapse
    ) -> (
        ScoresSynapse
    ):  
        """
        forward method that is called when the validator is queried with an axon
        """
        return ScoresSynapse(scores=self.scores.tolist())

    async def forward(self, synapse: ScoresSynapse) -> Awaitable:
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """
        return forward(self, synapse)

    async def blacklist(self, synapse: ScoresSynapse) -> Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contructed via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (template.protocol.Dummy): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """
        if synapse.dendrite.hotkey == "5Fy7c6skhxBifdPPEs3TyytxFc7Rq6UdLqysNPZ5AMAUbRQx":
            return False, "Subnet owner hotkey"
        if synapse.dendrite.hotkey in ["5F2CsUDVbRbVMXTh9fAzF9GacjVX7UapvRxidrxe7z8BYckQ", "5F4tQyWrhfGVcNhoqeiNsR6KjD4wMZ2kfhLj4oHYuyHbZAc3", "5FFApaS75bv5pJHfAp2FVLBj9ZaXuFDjEypsaBNc1wCfe52v", "5C4z2FJzsxh9uWxGTThdK7EBKuMNpWbCEFnrKrQene6Rsn45"]:
            return False, "Allowed validator hotkey"
        return True, "Blacklisted"

    async def priority(self, synapse: ScoresSynapse) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (template.protocol.Dummy): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may recieve messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return 0.0

        # TODO(developer): Define how miners should prioritize requests.
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        priority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}"
        )
        return priority


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            if not validator.thread.is_alive():
                bt.logging.error("Child thread has exited, terminating parent thread.")
                sys.exit(1)  # Exit the parent thread if the child thread dies
            bt.logging.info(f"Validator running... {time.time()}")
            time.sleep(5)
