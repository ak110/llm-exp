"""VertexAI関連のレスポンス処理モジュール。"""

import json
import logging
import secrets
import time
import typing
import uuid

import google.genai.types
import openai.types.chat
import openai.types.chat.chat_completion
import openai.types.chat.chat_completion_chunk
import openai.types.completion_usage

import types_chat

logger = logging.getLogger(__name__)


def process_non_streaming_response(
    request: types_chat.ChatRequest,
    response: google.genai.types.GenerateContentResponse,
) -> openai.types.chat.ChatCompletion:
    """非ストリーミングレスポンスをOpenAI形式に変換します。"""
    logger.debug(f"{response=}")

    choices: list[openai.types.chat.chat_completion.Choice] = []
    if response.candidates:
        for i, candidate in enumerate(response.candidates):
            openai_content: list[str] = []
            tool_calls: list[openai.types.chat.ChatCompletionMessageToolCall] = []

            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if part.text is not None:
                        openai_content.append(part.text)
                    elif part.function_call is not None:
                        # ツール呼び出しの処理
                        tool_call = openai.types.chat.ChatCompletionMessageToolCall(
                            id=secrets.token_urlsafe(8),
                            type="function",
                            function=openai.types.chat.chat_completion_message_tool_call.Function(
                                name=(
                                    ""
                                    if part.function_call.name is None
                                    else part.function_call.name
                                ),
                                arguments=(
                                    ""
                                    if part.function_call.args is None
                                    else json.dumps(dict(part.function_call.args))
                                ),
                            ),
                        )
                        tool_calls.append(tool_call)

            message = openai.types.chat.ChatCompletionMessage(
                role="assistant",
                content="\n".join(openai_content) if openai_content else None,
            )
            if tool_calls:
                message.tool_calls = tool_calls

            finish_reason = _convert_finish_reason(candidate.finish_reason)

            choices.append(
                openai.types.chat.chat_completion.Choice(
                    index=i,
                    message=message,
                    finish_reason="stop" if finish_reason is None else finish_reason,
                )
            )

    return openai.types.chat.ChatCompletion.model_construct(
        id=str(uuid.uuid4()),
        choices=choices,
        created=int(time.time()),
        model=request.model,
        object="chat.completion",
        usage=_convert_usage(response.usage_metadata),
    )


def process_stream_chunk(
    request: types_chat.ChatRequest, chunk: google.genai.types.GenerateContentResponse
) -> openai.types.chat.ChatCompletionChunk | None:
    """ストリームチャンクをOpenAI形式に変換します。"""
    logger.debug(f"Processing stream event: {chunk}")

    choices: list[openai.types.chat.chat_completion_chunk.Choice] = []
    if chunk.candidates:
        for candidate in chunk.candidates:
            delta = openai.types.chat.chat_completion_chunk.ChoiceDelta()

            if candidate.content and candidate.content.parts:
                content_parts: list[str] = []
                tool_calls: list[
                    openai.types.chat.chat_completion_chunk.ChoiceDeltaToolCall
                ] = []

                for part in candidate.content.parts:
                    if part.text is not None:
                        content_parts.append(part.text)
                    if part.function_call is not None:
                        # ツール呼び出しの処理
                        tool_calls.append(
                            openai.types.chat.chat_completion_chunk.ChoiceDeltaToolCall(
                                index=len(tool_calls),
                                id=secrets.token_urlsafe(8),
                                type="function",
                                function=openai.types.chat.chat_completion_chunk.ChoiceDeltaToolCallFunction(
                                    name=part.function_call.name,
                                    arguments=(
                                        None
                                        if part.function_call.args is None
                                        else json.dumps(dict(part.function_call.args))
                                    ),
                                ),
                            )
                        )

                if len(content_parts) > 0:
                    delta.content = "\n".join(content_parts)

                if tool_calls:
                    delta.tool_calls = tool_calls

            finish_reason = _convert_finish_reason(candidate.finish_reason)

            choices.append(
                openai.types.chat.chat_completion_chunk.Choice(
                    index=candidate.index or 0, delta=delta, finish_reason=finish_reason
                )
            )

    if not choices:
        # 空のチャンクの場合はスキップ
        return None

    return openai.types.chat.ChatCompletionChunk(
        id=str(uuid.uuid4()),
        object="chat.completion.chunk",
        choices=choices,
        created=int(time.time()),
        model=request.model,
        usage=_convert_usage(chunk.usage_metadata),
    )


def _convert_finish_reason(
    finish_reason: google.genai.types.FinishReason | None,
) -> typing.Literal["stop", "length", "tool_calls", "content_filter"] | None:
    """finish_reasonをOpenAI形式に変換する。"""
    if finish_reason is None:
        return None

    _FINISH_REASON_TABLE: dict[
        google.genai.types.FinishReason,
        typing.Literal["stop", "length", "tool_calls", "content_filter"],
    ] = {
        google.genai.types.FinishReason.STOP: "stop",
        google.genai.types.FinishReason.MAX_TOKENS: "length",
        google.genai.types.FinishReason.SAFETY: "content_filter",
        google.genai.types.FinishReason.RECITATION: "content_filter",
        google.genai.types.FinishReason.LANGUAGE: "content_filter",
        google.genai.types.FinishReason.OTHER: "stop",
        google.genai.types.FinishReason.BLOCKLIST: "content_filter",
        google.genai.types.FinishReason.PROHIBITED_CONTENT: "content_filter",
        google.genai.types.FinishReason.SPII: "content_filter",
        google.genai.types.FinishReason.MALFORMED_FUNCTION_CALL: "tool_calls",
        google.genai.types.FinishReason.IMAGE_SAFETY: "content_filter",
        google.genai.types.FinishReason.UNEXPECTED_TOOL_CALL: "tool_calls",
    }
    return _FINISH_REASON_TABLE.get(finish_reason, "stop")


def _convert_usage(
    usage_metadata: google.genai.types.GenerateContentResponseUsageMetadata | None,
):
    """usage_metadataをOpenAI形式に変換する。"""
    if usage_metadata is None:
        return None
    # 値がほぼ全部Noneの場合がある
    if (
        usage_metadata.prompt_token_count is None
        and usage_metadata.candidates_token_count is None
        and usage_metadata.total_token_count is None
    ):
        return None

    # 例: https://ai.google.dev/gemini-api/docs/caching?hl=ja&lang=python
    # prompt_token_count: 696219
    # cached_content_token_count: 696190
    # candidates_token_count: 214
    # total_token_count: 696433

    return openai.types.completion_usage.CompletionUsage(
        prompt_tokens=usage_metadata.prompt_token_count or 0,
        completion_tokens=usage_metadata.candidates_token_count or 0,
        total_tokens=usage_metadata.total_token_count or 0,
        prompt_tokens_details=openai.types.completion_usage.PromptTokensDetails(
            audio_tokens=sum(
                (detail.token_count or 0)
                for detail in (usage_metadata.prompt_tokens_details or [])
                if detail.modality == google.genai.types.MediaModality.AUDIO
            ),
            cached_tokens=usage_metadata.cached_content_token_count,
        ),
        completion_tokens_details=openai.types.completion_usage.CompletionTokensDetails(
            reasoning_tokens=usage_metadata.thoughts_token_count or 0,
            audio_tokens=sum(
                (detail.token_count or 0)
                for detail in (usage_metadata.candidates_tokens_details or [])
                if detail.modality == google.genai.types.MediaModality.AUDIO
            ),
            accepted_prediction_tokens=None,
            rejected_prediction_tokens=None,
        ),
    )
