"""AWS関連のレスポンス処理モジュール。"""

import json
import logging
import time
import typing
import uuid

import openai.types.chat
import openai.types.completion_usage
import types_aiobotocore_bedrock_runtime.type_defs

import types_chat

logger = logging.getLogger(__name__)


def process_non_streaming_response(
    request: types_chat.ChatRequest,
    response: types_aiobotocore_bedrock_runtime.type_defs.ConverseResponseTypeDef,
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

    usage: openai.types.completion_usage.CompletionUsage | None = None
    if (bedrock_usage := response.get("usage")) is not None:
        # TODO: cacheWriteがOpenAIに無いっぽい ⇒ どこかに無理やり入れる？
        # bedrock_usage.get("cacheWriteInputTokens", 0)
        usage = openai.types.completion_usage.CompletionUsage.model_construct(
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

    # レスポンスメッセージの処理
    output = response.get("output", {})
    message = output.get("message", {})
    content_blocks = message.get("content", [])

    # OpenAI形式に変換
    openai_message_content = []
    tool_calls = []

    for content_block in content_blocks:
        if "text" in content_block:
            openai_message_content.append(content_block["text"])
        elif "toolUse" in content_block:
            tool_use = content_block["toolUse"]
            tool_call = {
                "id": tool_use["toolUseId"],
                "type": "function",
                "function": {
                    "name": tool_use.get("name", ""),
                    "arguments": json.dumps(tool_use.get("input", {})),
                },
            }
            tool_calls.append(tool_call)

    # メッセージ作成
    openai_message = {
        "role": "assistant",
        "content": (
            " ".join(openai_message_content) if openai_message_content else None
        ),
    }

    if tool_calls:
        openai_message["tool_calls"] = tool_calls

    return openai.types.chat.ChatCompletion.model_construct(
        id=str(uuid.uuid4()),
        choices=[
            {
                "message": openai_message,
                "finish_reason": get_finish_reason(response.get("stopReason")),
                "index": 0,
            }
        ],
        created=int(time.time()),
        model=request.model,
        object="chat.completion",
        system_fingerprint=None,
        usage=usage,
    )


def process_stream_event(
    request: types_chat.ChatRequest,
    event_data: types_aiobotocore_bedrock_runtime.type_defs.ConverseStreamOutputTypeDef,
) -> openai.types.chat.ChatCompletionChunk | None:
    """ストリームイベントをOpenAI形式のチャンクに変換します。

    Args:
        event_data: イベントデータ
        request: リクエスト情報

    Returns:
        ChatCompletionChunk | None: OpenAI形式のチャンク。イベントが処理不要な場合はNone。
    """
    event_dict = typing.cast(dict[str, typing.Any], event_data)
    logger.debug(f"Processing stream event: {event_dict}")

    # エラーイベントの処理
    if "internalServerException" in event_dict:
        error_msg = event_dict["internalServerException"].get(
            "message", "Internal server error"
        )
        logger.error(f"Internal server exception: {error_msg}")
        return None

    if "modelStreamErrorException" in event_dict:
        error_msg = event_dict["modelStreamErrorException"].get(
            "message", "Model stream error"
        )
        logger.error(f"Model stream error: {error_msg}")
        return None

    if "validationException" in event_dict:
        error_msg = event_dict["validationException"].get("message", "Validation error")
        logger.error(f"Validation exception: {error_msg}")
        return None

    if "throttlingException" in event_dict:
        error_msg = event_dict["throttlingException"].get("message", "Throttling error")
        logger.error(f"Throttling exception: {error_msg}")
        return None

    if "serviceUnavailableException" in event_dict:
        error_msg = event_dict["serviceUnavailableException"].get(
            "message", "Service unavailable"
        )
        logger.error(f"Service unavailable exception: {error_msg}")
        return None

    # メッセージ開始イベント
    if "messageStart" in event_dict:
        # OpenAIではmessageStartに相当するイベントは無いが、roleを含むchunkを返す
        role = event_dict["messageStart"].get("role", "assistant")
        return openai.types.chat.ChatCompletionChunk(
            id=str(uuid.uuid4()),
            choices=[{"delta": {"role": role}, "finish_reason": None, "index": 0}],
            created=int(time.time()),
            model=request.model,
            object="chat.completion.chunk",
        )

    # コンテンツブロック開始イベント
    if "contentBlockStart" in event_dict:
        start_data = event_dict["contentBlockStart"]["start"]
        content_block_index = event_dict["contentBlockStart"]["contentBlockIndex"]

        # ツール使用の開始
        if "toolUse" in start_data:
            tool_use = start_data["toolUse"]
            return openai.types.chat.ChatCompletionChunk(
                id=str(uuid.uuid4()),
                choices=[
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": content_block_index,
                                    "id": tool_use["toolUseId"],
                                    "type": "function",
                                    "function": {
                                        "name": tool_use.get("name", ""),
                                        "arguments": "",
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                        "index": 0,
                    }
                ],
                created=int(time.time()),
                model=request.model,
                object="chat.completion.chunk",
            )

        # 他のブロック開始は特に処理しない（テキストブロックなど）
        return None

    # コンテンツブロック差分イベント（メインのストリーミングコンテンツ）
    if "contentBlockDelta" in event_dict:
        delta = event_dict["contentBlockDelta"]["delta"]
        content_block_index = event_dict["contentBlockDelta"]["contentBlockIndex"]

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
            tool_use_delta = delta["toolUse"]
            return openai.types.chat.ChatCompletionChunk(
                id=str(uuid.uuid4()),
                choices=[
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": content_block_index,
                                    "function": {
                                        "arguments": tool_use_delta.get("input", "")
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                        "index": 0,
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
    if "contentBlockStop" in event_dict:
        # OpenAIではコンテンツブロック単位の終了イベントは無いので無視
        return None

    # メッセージ終了イベント
    if "messageStop" in event_dict:
        stop_reason = event_dict["messageStop"].get("stopReason")
        return openai.types.chat.ChatCompletionChunk(
            id=str(uuid.uuid4()),
            choices=[
                {
                    "delta": {},
                    "finish_reason": get_finish_reason(stop_reason),
                    "index": 0,
                }
            ],
            created=int(time.time()),
            model=request.model,
            object="chat.completion.chunk",
        )

    # メタデータイベント
    if "metadata" in event_dict:
        metadata = event_dict["metadata"]
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


def get_finish_reason(
    stop_reason: str | None,
) -> typing.Literal["stop", "length", "tool_calls", "content_filter"] | None:
    """stopReasonをOpenAIのfinish_reasonに変換。"""
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
