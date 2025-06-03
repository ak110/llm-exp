"""AWS関連のレスポンス処理モジュール。"""

import json
import logging
import time
import typing
import uuid

import openai.types.chat
import openai.types.completion_usage
import types_aiobotocore_bedrock_runtime.type_defs as bedrock_types

import types_chat

logger = logging.getLogger(__name__)


def process_non_streaming_response(
    request: types_chat.ChatRequest, response: bedrock_types.ConverseResponseTypeDef
) -> openai.types.chat.ChatCompletion:
    """非ストリーミングレスポンスをOpenAI形式に変換します。"""
    logger.debug(f"{response=}")

    # レスポンスメッセージの処理
    output = response.get("output", {})
    message = output.get("message", {})
    content_blocks = message.get("content", [])

    # OpenAI形式に変換
    openai_content: list[str] = []
    tool_calls: list[openai.types.chat.ChatCompletionMessageToolCall] = []

    for content_block in content_blocks:
        if "text" in content_block:
            openai_content.append(content_block["text"])
        elif "toolUse" in content_block:
            tool_use = content_block["toolUse"]
            tool_calls.append(
                openai.types.chat.ChatCompletionMessageToolCall(
                    id=tool_use["toolUseId"],
                    type="function",
                    function=openai.types.chat.chat_completion_message_tool_call.Function(
                        name=tool_use.get("name", ""),
                        arguments=json.dumps(tool_use.get("input", {})),
                    ),
                )
            )

    # メッセージ作成
    openai_message = openai.types.chat.ChatCompletionMessage(
        role="assistant",
        content="\n".join(openai_content) if openai_content else None,
        tool_calls=tool_calls if tool_calls else None,
    )

    return openai.types.chat.ChatCompletion.model_construct(
        id=str(uuid.uuid4()),
        choices=[
            {
                "message": openai_message,
                "finish_reason": _convert_finish_reason(response.get("stopReason")),
                "index": 0,
            }
        ],
        created=int(time.time()),
        model=request.model,
        object="chat.completion",
        system_fingerprint=None,
        usage=_convert_usage(response.get("usage")),
    )


def process_stream_event(
    request: types_chat.ChatRequest,
    event_data: bedrock_types.ConverseStreamOutputTypeDef,
) -> openai.types.chat.ChatCompletionChunk | None:
    """ストリームイベントをOpenAI形式のチャンクに変換します。"""
    logger.debug(f"Processing stream event: {event_data}")

    # エラーイベントの処理
    if "internalServerException" in event_data:
        error_msg = event_data["internalServerException"].get(
            "message", "Internal server error"
        )
        logger.error(f"Internal server exception: {error_msg}")
        return None

    if "modelStreamErrorException" in event_data:
        error_msg = event_data["modelStreamErrorException"].get(
            "message", "Model stream error"
        )
        logger.error(f"Model stream error: {error_msg}")
        return None

    if "validationException" in event_data:
        error_msg = event_data["validationException"].get("message", "Validation error")
        logger.error(f"Validation exception: {error_msg}")
        return None

    if "throttlingException" in event_data:
        error_msg = event_data["throttlingException"].get("message", "Throttling error")
        logger.error(f"Throttling exception: {error_msg}")
        return None

    if "serviceUnavailableException" in event_data:
        error_msg = event_data["serviceUnavailableException"].get(
            "message", "Service unavailable"
        )
        logger.error(f"Service unavailable exception: {error_msg}")
        return None

    chunk = openai.types.chat.ChatCompletionChunk(
        id=str(uuid.uuid4()),
        choices=[],
        created=int(time.time()),
        model=request.model,
        object="chat.completion.chunk",
    )

    # メッセージ開始イベント
    # OpenAIではmessageStartに相当するイベントは無いが、roleを含むchunkを返す
    message_start: bedrock_types.MessageStartEventTypeDef | None = event_data.get(
        "messageStart"
    )
    if message_start is not None:
        role = message_start.get("role", "assistant")
        chunk.choices.append(
            openai.types.chat.chat_completion_chunk.Choice(
                index=0,
                delta=openai.types.chat.chat_completion_chunk.ChoiceDelta(role=role),
            )
        )

    # コンテンツブロック開始イベント
    content_block_start: bedrock_types.ContentBlockStartEventTypeDef | None = (
        event_data.get("contentBlockStart")
    )
    if content_block_start is not None:
        start_data: bedrock_types.ContentBlockStartTypeDef = content_block_start.get(
            "start", {}
        )
        content_block_index = content_block_start.get("contentBlockIndex", 0)
        chunk.choices.append(
            openai.types.chat.chat_completion_chunk.Choice(
                index=content_block_index,
                delta=openai.types.chat.chat_completion_chunk.ChoiceDelta(),
            )
        )

        # ツール使用の開始
        tool_use_block_start: bedrock_types.ToolUseBlockStartTypeDef | None = (
            start_data.get("toolUse")
        )
        if tool_use_block_start is not None:
            chunk.choices[-1].delta = (
                openai.types.chat.chat_completion_chunk.ChoiceDelta(
                    tool_calls=[
                        openai.types.chat.chat_completion_chunk.ChoiceDeltaToolCall(
                            index=0,
                            id=tool_use_block_start.get("toolUseId"),
                            type="function",
                            function=openai.types.chat.chat_completion_chunk.ChoiceDeltaToolCallFunction(
                                name=tool_use_block_start.get("name", ""), arguments=""
                            ),
                        )
                    ]
                )
            )

    # コンテンツブロック差分イベント（メインのストリーミングコンテンツ）
    content_block_delta: bedrock_types.ContentBlockDeltaEventTypeDef | None = (
        event_data.get("contentBlockDelta")
    )
    if content_block_delta is not None:
        delta: bedrock_types.ContentBlockDeltaTypeDef = content_block_delta.get(
            "delta", {}
        )
        content_block_index = content_block_delta.get("contentBlockIndex", 0)
        chunk.choices.append(
            openai.types.chat.chat_completion_chunk.Choice(
                index=content_block_index,
                delta=openai.types.chat.chat_completion_chunk.ChoiceDelta(),
            )
        )
        text_contents: list[str] = []

        # テキストデルタ
        delta_text = delta.get("text")
        if delta_text is not None:
            text_contents.append(delta_text)

        # ツール使用デルタ
        tool_use_delta: bedrock_types.ToolUseBlockDeltaTypeDef | None = delta.get(
            "toolUse"
        )
        if tool_use_delta is not None:
            chunk.choices[-1].delta.tool_calls = [
                openai.types.chat.chat_completion_chunk.ChoiceDeltaToolCall(
                    index=0,
                    type="function",
                    function=openai.types.chat.chat_completion_chunk.ChoiceDeltaToolCallFunction(
                        arguments=tool_use_delta.get("input", "")
                    ),
                )
            ]

        # 推論コンテンツデルタ（Claude 3.5 Sonnetなどの場合）
        # 推論内容はOpenAI APIには直接対応するものが無いが、contentとして扱う
        reasoning_delta = delta.get("reasoningContent")
        if reasoning_delta is not None:
            reasoning_text = reasoning_delta.get("text")
            if reasoning_text is not None:
                text_contents.append(reasoning_text)

        if len(text_contents) > 0:
            chunk.choices[-1].delta.content = "\n".join(text_contents)

    # コンテンツブロック終了イベント
    # if "contentBlockStop" in event_data:

    # メッセージ終了イベント
    message_stop: bedrock_types.MessageStopEventTypeDef | None = event_data.get(
        "messageStop"
    )
    if message_stop is not None:
        chunk.choices.append(
            openai.types.chat.chat_completion_chunk.Choice(
                index=0,
                delta=openai.types.chat.chat_completion_chunk.ChoiceDelta(),
                finish_reason=_convert_finish_reason(message_stop.get("stopReason")),
            )
        )
        # message_stop["additionalModelResponseFields"]

    # メタデータイベント
    if "metadata" in event_data:
        metadata: bedrock_types.ConverseStreamMetadataEventTypeDef = event_data[
            "metadata"
        ]
        bedrock_usage: bedrock_types.TokenUsageTypeDef | None = metadata.get("usage")
        if bedrock_usage is not None:
            cached_tokens = bedrock_usage.get("cacheReadInputTokens")  # type: ignore
            chunk.usage = openai.types.completion_usage.CompletionUsage(
                prompt_tokens=bedrock_usage.get("inputTokens", 0),
                completion_tokens=bedrock_usage.get("outputTokens", 0),
                total_tokens=bedrock_usage.get("totalTokens", 0),
                # 可能ならキャッシュトークン情報も含める
                prompt_tokens_details=(
                    openai.types.completion_usage.PromptTokensDetails(
                        cached_tokens=typing.cast(int, cached_tokens)
                    )
                    if cached_tokens is not None
                    else None
                ),
            )

    return chunk


def _convert_finish_reason(
    stop_reason: str | None,
) -> typing.Literal["stop", "length", "tool_calls", "content_filter"] | None:
    """stopReasonをOpenAIのfinish_reasonに変換する。"""
    if stop_reason is None:
        return None
    return {  # type: ignore[return-value]
        "end_turn": "stop",
        "tool_use": "tool_calls",
        "max_tokens": "length",
        "stop_sequence": "stop",
        "guardrail_intervened": "content_filter",
        "content_filtered": "content_filter",
    }.get(stop_reason)


def _convert_usage(
    bedrock_usage: bedrock_types.TokenUsageTypeDef | None,
) -> openai.types.completion_usage.CompletionUsage | None:
    """usageをOpenAIの形式に変換する。"""
    if bedrock_usage is None:
        return None
    # TODO: cacheWriteがOpenAIに無いっぽい ⇒ どこかに無理やり入れる？
    # bedrock_usage.get("cacheWriteInputTokens", 0)
    return openai.types.completion_usage.CompletionUsage.model_construct(
        prompt_tokens=bedrock_usage.get("inputTokens", 0),
        completion_tokens=bedrock_usage.get("outputTokens", 0),
        total_tokens=bedrock_usage.get("totalTokens", 0),
        prompt_tokens_details=openai.types.completion_usage.PromptTokensDetails.model_construct(
            cached_tokens=bedrock_usage.get("cacheReadInputTokens", 0)
            # audio_tokens=0,
        ),
        # completion_tokens_details=openai.types.completion_usage.CompletionTokensDetails.model_construct(
        #     accepted_prediction_tokens=0
        #     rejected_prediction_tokens=0,
        #     audio_tokens=0,
        #     reasoning_tokens=0,
        # ),
    )
