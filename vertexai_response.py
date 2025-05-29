"""VertexAI関連のレスポンス処理モジュール。"""

import json
import logging
import time
import typing
import uuid

import google.genai.types
import openai.types.chat
import openai.types.completion_usage

import types_chat

logger = logging.getLogger(__name__)


def process_non_streaming_response(
    request: types_chat.ChatRequest,
    response: google.genai.types.GenerateContentResponse,
) -> openai.types.chat.ChatCompletion:
    """非ストリーミングレスポンスをOpenAI形式に変換します。"""

    choices = []
    if response.candidates:
        for i, candidate in enumerate(response.candidates):
            content_parts = []
            tool_calls = []

            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text:
                        content_parts.append(part.text)
                    elif hasattr(part, "function_call") and part.function_call:
                        # ツール呼び出しの処理
                        tool_call = {
                            "id": str(
                                uuid.uuid4()
                            ),  # Geminiではツール呼び出しIDがないので生成
                            "type": "function",
                            "function": {
                                "name": part.function_call.name,
                                "arguments": json.dumps(dict(part.function_call.args)),
                            },
                        }
                        tool_calls.append(tool_call)

            message = {
                "role": "assistant",
                "content": " ".join(content_parts) if content_parts else None,
            }

            if tool_calls:
                message["tool_calls"] = tool_calls

            finish_reason = _convert_finish_reason(candidate.finish_reason)

            choices.append(
                {"index": i, "message": message, "finish_reason": finish_reason}
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

    choices = []
    if chunk.candidates:
        for i, candidate in enumerate(chunk.candidates):
            delta = {}

            if candidate.content and candidate.content.parts:
                content_parts = []
                tool_calls = []

                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text:
                        content_parts.append(part.text)
                    elif hasattr(part, "function_call") and part.function_call:
                        # ツール呼び出しの処理
                        tool_call = {
                            "index": len(tool_calls),
                            "id": str(uuid.uuid4()),
                            "type": "function",
                            "function": {
                                "name": part.function_call.name,
                                "arguments": json.dumps(dict(part.function_call.args)),
                            },
                        }
                        tool_calls.append(tool_call)

                if content_parts:
                    delta["content"] = " ".join(content_parts)

                if tool_calls:
                    delta["tool_calls"] = tool_calls

            finish_reason = (
                _convert_finish_reason(candidate.finish_reason)
                if candidate.finish_reason
                else None
            )

            choices.append({"index": i, "delta": delta, "finish_reason": finish_reason})

    if not choices:
        # 空のチャンクの場合はスキップ
        return None

    return openai.types.chat.ChatCompletionChunk.model_construct(
        id=str(uuid.uuid4()),
        object="chat.completion.chunk",
        choices=choices,
        created=int(time.time()),
        model=request.model,
        usage=_convert_usage(chunk.usage_metadata),
    )


def _convert_finish_reason(
    finish_reason: typing.Any,
) -> typing.Literal["stop", "length", "tool_calls", "content_filter"] | None:
    """finish_reasonをOpenAI形式に変換する。"""
    if finish_reason is None:
        return None

    # Geminiのfinish_reasonをOpenAIの形式にマッピング
    reason_str = str(finish_reason).lower()

    if "stop" in reason_str or "finish" in reason_str:
        return "stop"
    elif "length" in reason_str or "max_tokens" in reason_str:
        return "length"
    elif "safety" in reason_str or "filter" in reason_str:
        return "content_filter"
    elif "tool" in reason_str or "function" in reason_str:
        return "tool_calls"
    else:
        return "stop"  # デフォルト


def _convert_usage(
    usage_metadata: google.genai.types.GenerateContentResponseUsageMetadata | None,
):
    """usage_metadataをOpenAI形式に変換する。"""
    if usage_metadata is None:
        return None
    return openai.types.completion_usage.CompletionUsage(
        prompt_tokens=usage_metadata.prompt_token_count or 0,
        completion_tokens=usage_metadata.candidates_token_count or 0,
        total_tokens=usage_metadata.total_token_count or 0,
        prompt_tokens_details=openai.types.completion_usage.PromptTokensDetails(
            audio_tokens=sum(
                detail.token_count
                for detail in (usage_metadata.prompt_tokens_details or [])
                if detail.modality == google.genai.types.MediaModality.AUDIO
            ),
            cached_tokens=usage_metadata.cached_content_token_count,
        ),
        completion_tokens_details=openai.types.completion_usage.CompletionTokensDetails(
            reasoning_tokens=usage_metadata.thoughts_token_count or 0,
            audio_tokens=sum(
                detail.token_count
                for detail in (usage_metadata.candidates_tokens_details or [])
                if detail.modality == google.genai.types.MediaModality.AUDIO
            ),
            accepted_prediction_tokens=None,
            rejected_prediction_tokens=None,
        ),
    )
