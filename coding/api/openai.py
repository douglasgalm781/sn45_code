import os
import httpx
import dotenv
import logging
import asyncio
import argparse
import bittensor as bt
from cachetools.func import ttl_cache
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from coding.protocol import StreamCodeSynapse
from coding.api.loggers import CallCountManager
from coding.api.protocol import CompletionRequest, ChatCompletionRequest
from coding.api.completion import (
    completion,
    chat_completion,
    chat_completion_stream_generator,
    completion_stream_generator,
)
from coding.api.cleaners import clean_fixes, remove_secret_lines, remove_generate_prompt

dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(
    description="Run the FastAPI server with configurable constants."
)
parser.add_argument(
    "--wallet", type=str, default="test_validator", help="Name of the wallet"
)  # TODO change to validator
parser.add_argument("--hotkey", type=str, default="default", help="Name of the hotkey")
parser.add_argument(
    "--network", type=str, default="ws://127.0.0.1:9946", help="Network address"
)  # TODO change to finney
parser.add_argument(
    "--netuid", type=int, default=1, help="NetUID value"
)  # TODO change to real
parser.add_argument(
    "--stat_api_url", type=str, default=None, help="Url of the statistics API"
)
parser.add_argument(
    "--stat_api_key", type=str, default=None, help="Key for the statistics API"
)
args = parser.parse_args()

WALLET_NAME = args.wallet
HOTKEY_NAME = args.hotkey
NETWORK = args.network
NETUID = args.netuid

STAT_API_URL = os.getenv("STAT_API_URL", args.stat_api_url)
STAT_API_KEY = os.getenv("STAT_API_KEY", args.stat_api_key)
CALL_COUNTER = None

if STAT_API_URL and STAT_API_KEY:
    CALL_COUNTER = CallCountManager(url=STAT_API_URL, key=STAT_API_KEY)


subtensor = None
subnet = None
wallet = None
dendrite = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global subtensor
    global subnet
    global dendrite
    global wallet
    app.requests_client = httpx.AsyncClient()
    subtensor = bt.subtensor(network=NETWORK)
    subnet = subtensor.metagraph(netuid=NETUID)
    wallet = bt.wallet(name=WALLET_NAME, hotkey=HOTKEY_NAME)
    dendrite = bt.dendrite(wallet=wallet)
    yield
    await app.requests_client.aclose()


@ttl_cache(maxsize=100, ttl=60 * 60)
def get_top_miner_uid():
    global subtensor
    global subnet
    subtensor = bt.subtensor(network=NETWORK)
    subnet = subtensor.metagraph(netuid=NETUID)
    return int(subnet.I.argmax())


async def forward(uid, synapse, timeout=25):
    global dendrite
    response = await dendrite(
        axons=subnet.axons[uid],
        synapse=synapse,
        deserialize=False,
        timeout=timeout,
        streaming=True,
    )
    return response


app = FastAPI(
    lifespan=lifespan,
    docs_url="/",
    redoc_url=None,
)


@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if CALL_COUNTER:
        asyncio.create_task(CALL_COUNTER.add())
    if not request.attachments:
        request.attachments = []
    if not request.files:
        request.files = []
    try:
        generator = await forward(
            0,
            StreamCodeSynapse(
                messages=request.messages,
                attachments=request.attachments,
                files=request.files,
                uid=0,
            ),
        )
        if request.stream:
            return StreamingResponse(
                chat_completion_stream_generator(request, generator),
                media_type="text/event-stream",
            )
        else:
            return JSONResponse(
                content=(await chat_completion(request, generator)).model_dump()
            )
    except httpx.ReadTimeout:
        raise HTTPException(408) from None
    except Exception as e:
        raise HTTPException(500) from None


async def collect_async_gen(gen):
    return [item async for item in gen]


@app.post("/completions")
@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    if CALL_COUNTER:
        asyncio.create_task(CALL_COUNTER.add())
    if isinstance(request.prompt, list):
        request.prompt = " ".join(request.prompt)
    # remove any fim prefix/suffixes
    request.prompt = remove_generate_prompt(
        remove_secret_lines(clean_fixes(request.prompt))
    )
    try:
        # generator = await forward(
        #     get_top_miner_uid(), StreamCodeSynapse(query=clean_deepseek(request.prompt))
        # )
        generator = await forward(0, StreamCodeSynapse(query=request.prompt, uid=0))

        if request.stream:
            return StreamingResponse(
                completion_stream_generator(request, generator),
                media_type="text/event-stream",
            )
        else:
            return JSONResponse(
                content=(await completion(request, generator)).model_dump()
            )
    except httpx.ReadTimeout:
        raise HTTPException(408) from None
    except Exception as e:
        print(e)
        raise HTTPException(500) from None


@app.get("/models")
@app.get("/v1/models")
async def models():
    try:
        return "code"
    except httpx.ReadTimeout:
        raise HTTPException(408) from None
    except Exception:
        raise HTTPException(500) from None


if __name__ == "__main__":
    import uvicorn

    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["loggers"]["uvicorn"]["level"] = "DEBUG"
    log_config["loggers"]["uvicorn.error"]["level"] = "DEBUG"
    log_config["loggers"]["uvicorn.access"]["level"] = "DEBUG"
    uvicorn.run("coding.api.openai:app", host="0.0.0.0", port=9990, reload=False)
