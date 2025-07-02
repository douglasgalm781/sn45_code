import os
import time
import bittensor as bt
from starlette.types import Send
from functools import partial
from typing import Dict, Awaitable
from langchain_openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.base import RunnableSequence

from coding.protocol import StreamCodeSynapse


def miner_init(self):
    """
    Initializes the miner. This function is called once when the miner is created.
    """
    _ = load_dotenv(find_dotenv())
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    # Set openai key and other args
    self.model = OpenAI(
        api_key=api_key,
        model_name=self.config.neuron.model_id,
        max_tokens=2048,
        temperature=0.7,
    )


def miner_process(self, synapse: StreamCodeSynapse) -> Awaitable:
    """
    The miner process function is called every time the miner receives a request. This function should contain the main logic of the miner.
    """

    async def _forward(
        self,
        query: str,
        init_time: float,
        timeout_threshold: float,
        chain: RunnableSequence,
        chain_formatter: Dict[str, str],
        send: Send,
    ):
        buffer = []
        temp_completion = ""  # for wandb logging
        timeout_reached = False

        try:
            # Langchain built in streaming. 'astream' also available for async
            for token in chain.stream(chain_formatter):
                buffer.append(token)

                if time.time() - init_time > timeout_threshold:
                    bt.logging.debug(f"⏰ Timeout reached, stopping streaming")
                    timeout_reached = True
                    break

                if len(buffer) == self.config.neuron.streaming_batch_size:
                    joined_buffer = "".join(buffer)
                    temp_completion += joined_buffer
                    bt.logging.debug(f"Streamed tokens: {joined_buffer}")

                    await send(
                        {
                            "type": "http.response.body",
                            "body": joined_buffer.encode("utf-8"),
                            "more_body": True,
                        }
                    )
                    buffer = []

            if (
                buffer and not timeout_reached
            ):  # Don't send the last buffer of data if timeout.
                joined_buffer = "".join(buffer)
                await send(
                    {
                        "type": "http.response.body",
                        "body": joined_buffer.encode("utf-8"),
                        "more_body": False,
                    }
                )

        except Exception as e:
            bt.logging.error(f"Error in forward: {e}")
            if self.config.neuron.stop_on_forward_exception:
                self.should_exit = True

    bt.logging.debug(f"📧 Query received, forwarding synapse: {synapse}")

    prompt = PromptTemplate.from_template("{query}")
    chain = prompt | self.model | StrOutputParser()

    query = synapse.query

    chain_formatter = {"query": query}

    init_time = time.time()
    timeout_threshold = synapse.timeout

    token_streamer = partial(
        _forward,
        self,
        query,
        init_time,
        timeout_threshold,
        chain,
        chain_formatter,
    )
    return synapse.create_streaming_response(token_streamer)
