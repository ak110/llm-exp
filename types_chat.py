"""OpenAI互換API用pydantic。"""

import typing
from typing import Literal

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


class ChatResponse(pydantic.BaseModel):
    """チャット補完APIのレスポンス。"""

    id: str
    """A unique identifier for the chat completion."""

    choices: list[openai.types.chat.chat_completion.Choice]
    """A list of chat completion choices.

    Can be more than one if `n` is greater than 1.
    """

    created: int
    """The Unix timestamp (in seconds) of when the chat completion was created."""

    model: str
    """The model used for the chat completion."""

    object: Literal["chat.completion"]
    """The object type, which is always `chat.completion`."""

    service_tier: Literal["auto", "default", "flex"] | None = None
    """Specifies the latency tier to use for processing the request.

    This parameter is relevant for customers subscribed to the scale tier service:

    - If set to 'auto', and the Project is Scale tier enabled, the system will
      utilize scale tier credits until they are exhausted.
    - If set to 'auto', and the Project is not Scale tier enabled, the request will
      be processed using the default service tier with a lower uptime SLA and no
      latency guarentee.
    - If set to 'default', the request will be processed using the default service
      tier with a lower uptime SLA and no latency guarentee.
    - If set to 'flex', the request will be processed with the Flex Processing
      service tier.
      [Learn more](https://platform.openai.com/docs/guides/flex-processing).
    - When not set, the default behavior is 'auto'.

    When this parameter is set, the response body will include the `service_tier`
    utilized.
    """

    system_fingerprint: str | None = None
    """This fingerprint represents the backend configuration that the model runs with.

    Can be used in conjunction with the `seed` request parameter to understand when
    backend changes have been made that might impact determinism.
    """

    usage: "ChatResponseUsage | None" = None
    """Usage statistics for the completion request."""


class ChatResponseUsage(pydantic.BaseModel):
    """チャット補完APIのレスポンスの使用量統計。"""

    completion_tokens: int
    """Number of tokens in the generated completion."""

    prompt_tokens: int
    """Number of tokens in the prompt."""

    total_tokens: int
    """Total number of tokens used in the request (prompt + completion)."""

    completion_tokens_details: (
        openai.types.completion_usage.CompletionTokensDetails
    ) | None = None
    """Breakdown of tokens used in a completion."""

    prompt_tokens_details: (
        openai.types.completion_usage.PromptTokensDetails
    ) | None = None
    """Breakdown of tokens used in the prompt."""
