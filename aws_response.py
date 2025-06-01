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
    """非ストリーミングレスポンスをOpenAI形式に変換します。

    Args:
        response: Bedrockからのレスポンス
        request: リクエスト情報

    Returns:
        ChatCompletion: OpenAI形式のレスポンス
    """
    logger.debug(f"{response=}")
    print(f"{response=}")  # TODO: 仮

    # レスポンスメッセージの処理
    output = response.get("output", {})
    message = output.get("message", {})
    content_blocks = message.get("content", [])

    # OpenAI形式に変換
    openai_message_content: list[str] = []
    tool_calls: list[openai.types.chat.ChatCompletionMessageToolCall] = []

    for content_block in content_blocks:
        if "text" in content_block:
            openai_message_content.append(content_block["text"])
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
        content=" ".join(openai_message_content) if openai_message_content else None,
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
    """ストリームイベントをOpenAI形式のチャンクに変換します。

    Args:
        event_data: イベントデータ
        request: リクエスト情報

    Returns:
        ChatCompletionChunk | None: OpenAI形式のチャンク。イベントが処理不要な場合はNone。
    """
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

    # メッセージ開始イベント
    if "messageStart" in event_data:
        # OpenAIではmessageStartに相当するイベントは無いが、roleを含むchunkを返す
        message_start: bedrock_types.MessageStartEventTypeDef = event_data[
            "messageStart"
        ]
        role = message_start.get("role", "assistant")
        return openai.types.chat.ChatCompletionChunk(
            id=str(uuid.uuid4()),
            choices=[{"delta": {"role": role}, "finish_reason": None, "index": 0}],
            created=int(time.time()),
            model=request.model,
            object="chat.completion.chunk",
        )

    # コンテンツブロック開始イベント
    if "contentBlockStart" in event_data:
        content_block_start: bedrock_types.ContentBlockStartEventTypeDef = event_data[
            "contentBlockStart"
        ]
        start_data: bedrock_types.ContentBlockStartTypeDef = content_block_start.get(
            "start", {}
        )
        content_block_index = start_data.get("contentBlockIndex", 0)

        # ツール使用の開始
        if "toolUse" in start_data:
            tool_use: bedrock_types.ToolUseBlockStartTypeDef = start_data["toolUse"]
            return openai.types.chat.ChatCompletionChunk(
                id=str(uuid.uuid4()),
                choices=[
                    {
                        "index": content_block_index,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": tool_use.get("toolUseId"),
                                    "type": "function",
                                    "function": {
                                        "name": tool_use.get("name", ""),
                                        "arguments": "",
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
                created=int(time.time()),
                model=request.model,
                object="chat.completion.chunk",
            )

        # 他のブロック開始は特に処理しない（テキストブロックなど）
        return None

    # コンテンツブロック差分イベント（メインのストリーミングコンテンツ）
    if "contentBlockDelta" in event_data:
        content_block_delta: bedrock_types.ContentBlockDeltaEventTypeDef = event_data[
            "contentBlockDelta"
        ]
        delta: bedrock_types.ContentBlockDeltaTypeDef = content_block_delta.get(
            "delta", {}
        )
        content_block_index = content_block_delta.get("contentBlockIndex", 0)

        # テキストデルタ
        if "text" in delta:
            return openai.types.chat.ChatCompletionChunk(
                id=str(uuid.uuid4()),
                choices=[
                    {
                        "delta": {"content": delta["text"]},
                        "finish_reason": None,
                        "index": 0,
                    }
                ],
                created=int(time.time()),
                model=request.model,
                object="chat.completion.chunk",
            )

        # ツール使用デルタ
        if "toolUse" in delta:
            tool_use_delta: bedrock_types.ToolUseBlockDeltaTypeDef = delta["toolUse"]
            return openai.types.chat.ChatCompletionChunk(
                id=str(uuid.uuid4()),
                choices=[
                    {
                        "index": content_block_index,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": tool_use_delta.get("input", "")
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
                created=int(time.time()),
                model=request.model,
                object="chat.completion.chunk",
            )

        # 推論コンテンツデルタ（Claude 3.5 Sonnetなどの場合）
        if "reasoningContent" in delta:
            reasoning_delta = delta["reasoningContent"]
            # 推論内容はOpenAI APIには直接対応するものが無いが、contentとして扱う
            if "text" in reasoning_delta:
                return openai.types.chat.ChatCompletionChunk(
                    id=str(uuid.uuid4()),
                    choices=[
                        {
                            "delta": {"content": reasoning_delta["text"]},
                            "finish_reason": None,
                            "index": 0,
                        }
                    ],
                    created=int(time.time()),
                    model=request.model,
                    object="chat.completion.chunk",
                )

        # その他のデルタは無視
        return None

    # コンテンツブロック終了イベント
    if "contentBlockStop" in event_data:
        # OpenAIではコンテンツブロック単位の終了イベントは無いので無視
        return None

    # メッセージ終了イベント
    if "messageStop" in event_data:
        stop_reason = event_data["messageStop"].get("stopReason")
        return openai.types.chat.ChatCompletionChunk(
            id=str(uuid.uuid4()),
            choices=[
                {
                    "delta": {},
                    "finish_reason": _convert_finish_reason(stop_reason),
                    "index": 0,
                }
            ],
            created=int(time.time()),
            model=request.model,
            object="chat.completion.chunk",
        )

    # メタデータイベント
    if "metadata" in event_data:
        metadata = event_data["metadata"]
        usage = None

        if "usage" in metadata:
            bedrock_usage = metadata["usage"]
            usage = openai.types.completion_usage.CompletionUsage.model_construct(
                prompt_tokens=bedrock_usage.get("inputTokens", 0),
                completion_tokens=bedrock_usage.get("outputTokens", 0),
                total_tokens=bedrock_usage.get("totalTokens", 0),
                # 可能ならキャッシュトークン情報も含める
                prompt_tokens_details=(
                    openai.types.completion_usage.PromptTokensDetails.model_construct(
                        cached_tokens=bedrock_usage.get("cacheReadInputTokens", 0)
                    )
                    if "cacheReadInputTokens" in bedrock_usage
                    else None
                ),
            )

        return openai.types.chat.ChatCompletionChunk(
            id=str(uuid.uuid4()),
            choices=[],  # OpenAI APIではusage情報の際はchoicesは空
            created=int(time.time()),
            model=request.model,
            object="chat.completion.chunk",
            usage=usage,
        )

    # 処理されないイベントタイプ
    logger.debug(f"Unhandled stream event type: {list(event_data.keys())}")
    return None


def _convert_finish_reason(
    stop_reason: str | None,
) -> typing.Literal["stop", "length", "tool_calls", "content_filter"] | None:
    """stopReasonをOpenAIのfinish_reasonに変換する。"""
    if stop_reason is None:
        return None
    return {
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
