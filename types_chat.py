"""OpenAI互換API用pydantic。"""

import typing

import openai.types.chat
import openai.types.chat.chat_completion
import openai.types.completion_usage
import openai.types.shared.metadata
import openai.types.shared.reasoning_effort
import pydantic
from openai._types import NOT_GIVEN, NotGiven


class ChatRequest(pydantic.BaseModel):
    """チャット補完APIのリクエスト。"""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    messages: typing.Iterable[openai.types.chat.ChatCompletionMessageParam]
    model: str
    audio: openai.types.chat.ChatCompletionAudioParam | None | NotGiven = NOT_GIVEN
    frequency_penalty: float | None | NotGiven = NOT_GIVEN
    logit_bias: dict[str, int] | None | NotGiven = NOT_GIVEN
    logprobs: bool | None | NotGiven = NOT_GIVEN
    max_completion_tokens: int | None | NotGiven = NOT_GIVEN
    max_tokens: int | None | NotGiven = NOT_GIVEN
    metadata: openai.types.shared.metadata.Metadata | None | NotGiven = NOT_GIVEN
    modalities: list[typing.Literal["text", "audio"]] | None | NotGiven = NOT_GIVEN
    n: int | None | NotGiven = NOT_GIVEN
    parallel_tool_calls: bool | NotGiven = NOT_GIVEN
    prediction: (
        openai.types.chat.ChatCompletionPredictionContentParam | None | NotGiven
    ) = NOT_GIVEN
    presence_penalty: float | None | NotGiven = NOT_GIVEN
    reasoning_effort: (
        openai.types.shared.reasoning_effort.ReasoningEffort | None | NotGiven
    ) = NOT_GIVEN
    response_format: (
        openai.types.chat.completion_create_params.ResponseFormat | NotGiven
    ) = NOT_GIVEN
    seed: int | None | NotGiven = NOT_GIVEN
    service_tier: typing.Literal["auto", "default", "flex"] | None | NotGiven = (
        NOT_GIVEN
    )
    stop: str | list[str] | None | NotGiven = NOT_GIVEN
    store: bool | None | NotGiven = NOT_GIVEN
    stream: bool = False
    stream_options: (
        openai.types.chat.ChatCompletionStreamOptionsParam | None | NotGiven
    ) = NOT_GIVEN
    temperature: float | None | NotGiven = NOT_GIVEN
    tool_choice: openai.types.chat.ChatCompletionToolChoiceOptionParam | NotGiven = (
        NOT_GIVEN
    )
    tools: typing.Iterable[openai.types.chat.ChatCompletionToolParam] | NotGiven = (
        NOT_GIVEN
    )
    top_logprobs: int | None | NotGiven = NOT_GIVEN
    top_p: float | None | NotGiven = NOT_GIVEN
    user: str | NotGiven = NOT_GIVEN
    web_search_options: (
        openai.types.chat.completion_create_params.WebSearchOptions | NotGiven
    ) = NOT_GIVEN
