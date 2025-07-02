import time
import json

from http import HTTPStatus
from typing import AsyncGenerator, AsyncIterator, Union

from coding.api.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    DeltaMessage,
    ErrorResponse,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    CompletionResponse,
    CompletionResponseChoice,
)


def create_streaming_error_response(
    self,
    message: str,
    err_type: str = "BadRequestError",
    status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
) -> str:
    json_str = json.dumps(
        {
            "error": self.create_error_response(
                message=message, err_type=err_type, status_code=status_code
            ).model_dump()
        }
    )
    return json_str


async def chat_completion_stream_generator(
    request: ChatCompletionRequest, result_generator: AsyncIterator
) -> Union[ErrorResponse, AsyncGenerator[str, None]]:

    model_name = request.model
    created_time = int(time.time())
    chunk_object_type = "chat.completion.chunk"
    first_iteration = True

    try:
        async for res in result_generator:
            if not isinstance(res, str):
                break
            if first_iteration:
                role = request.messages[-1].role
                choice_data = ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(role=role),
                    logprobs=None,
                    finish_reason=None,
                )
                chunk = ChatCompletionStreamResponse(
                    id="",
                    object=chunk_object_type,
                    created=created_time,
                    choices=[choice_data],
                    model=model_name,
                )
                data = chunk.model_dump_json(exclude_unset=True)
                yield f"data: {data}\n\n"

                first_iteration = False

            choice_data = ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(content=res),
                logprobs=None,
                finish_reason="stop",
                stop_reason="",
            )
            chunk = ChatCompletionStreamResponse(
                id="",
                object=chunk_object_type,
                created=created_time,
                choices=[choice_data],
                model=model_name,
            )
            data = chunk.model_dump_json(exclude_unset=True, exclude_none=True)
            yield f"data: {data}\n\n"
    except ValueError as e:
        data = create_streaming_error_response(str(e))
        yield f"data: {data}\n\n"
    print("DONE")
    yield "data: [DONE]\n\n"


async def chat_completion(
    request: ChatCompletionRequest, result_generator: AsyncIterator
) -> Union[ErrorResponse, ChatCompletionResponse]:
    completion = ""
    async for chunk in result_generator:
        completion += chunk

    return ChatCompletionResponse(
        id="",
        object="chat.completion",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(content=completion),
                logprobs=None,
                finish_reason="stop",
                stop_reason="",
            )
        ],
    )


async def completion_stream_generator(
    request: CompletionRequest, result_generator: AsyncIterator
) -> Union[ErrorResponse, AsyncGenerator[str, None]]:

    model_name = request.model
    created_time = int(time.time())
    chunk_object_type = "chat.completion.chunk"
    first_iteration = True

    try:
        async for res in result_generator:
            if not isinstance(res, str):
                break
            if first_iteration:
                choice_data = CompletionResponseStreamChoice(
                    index=0, text="", logprobs=None, finish_reason=None
                )
                chunk = CompletionStreamResponse(
                    choices=[choice_data], model=model_name
                )
                data = chunk.model_dump_json(exclude_unset=True)
                yield f"data: {data}\n\n"

                first_iteration = False
            choice_data = CompletionResponseStreamChoice(
                index=0, text=res, logprobs=None, finish_reason=None
            )
            chunk = CompletionStreamResponse(
                id="",
                object=chunk_object_type,
                created=created_time,
                choices=[choice_data],
                model=model_name,
            )
            data = chunk.model_dump_json(exclude_unset=True, exclude_none=True)
            yield f"data: {data}\n\n"
    except ValueError as e:
        data = create_streaming_error_response(str(e))
        yield f"data: {data}\n\n"
    yield "data: [DONE]\n\n"


async def completion(
    request: CompletionRequest, result_generator: AsyncIterator
) -> Union[ErrorResponse, ChatCompletionResponse]:
    completion = ""
    async for chunk in result_generator:
        completion += chunk

    return CompletionResponse(
        model=request.model,
        choices=[
            CompletionResponseChoice(
                index=0, text=completion, finish_reason="stop", stop_reason=""
            )
        ],
    )
