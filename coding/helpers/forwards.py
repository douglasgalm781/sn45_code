import json
import time
import traceback
import bittensor as bt
from starlette.types import Send
from typing import List, Any, Dict
from langchain_core.runnables.base import RunnableSequence


async def string_forward(string, send: Send):
    await send(
        {
            "type": "http.response.body",
            "body": string,
            "more_body": False,
        }
    )


async def chain_forward(
    self,
    query: str,
    files: List[Any],
    extra_info: Dict[str, Any],
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
            if not isinstance(token, str):
                token = token.content
            buffer.append(token)

            if time.time() - init_time > timeout_threshold:
                bt.logging.debug(f"⏰ Timeout reached, stopping streaming")
                timeout_reached = True
                break

            if (
                not "broken_file" in extra_info.keys()
                and len(buffer) == self.config.neuron.streaming_batch_size
            ):
                joined_buffer = "".join(buffer)
                temp_completion += joined_buffer
                bt.logging.debug(f"Streamed tokens: {repr(joined_buffer)}")

                await send(
                    {
                        "type": "http.response.body",
                        "body": joined_buffer,
                        "more_body": True,
                    }
                )
                buffer = []

        if (
            buffer and not timeout_reached
        ):  # Don't send the last buffer of data if timeout.
            body = "".join(buffer)
            await send(
                {
                    "type": "http.response.body",
                    "body": body,
                    "more_body": False,
                }
            )
    except Exception as e:
        bt.logging.error(f"Error in forward: {e}, - {traceback.format_exc()}")
        if self.config.neuron.stop_on_forward_exception:
            self.should_exit = True
