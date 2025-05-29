"""VertexAI関連のリクエスト処理モジュール。"""

import json
import logging
import typing

import google.genai.types
import openai.types.chat
from openai._types import NOT_GIVEN

import types_chat

logger = logging.getLogger(__name__)


def make_generation_config(
    request: types_chat.ChatRequest,
) -> google.genai.types.GenerateContentConfigOrDict:
    """生成設定を作成します。"""
    generation_config = google.genai.types.GenerateContentConfig()

    if request.temperature is not NOT_GIVEN:
        generation_config.temperature = typing.cast(float | None, request.temperature)

    if request.max_tokens is not NOT_GIVEN:
        generation_config.max_output_tokens = typing.cast(
            int | None, request.max_tokens
        )

    if request.top_p is not NOT_GIVEN:
        generation_config.top_p = typing.cast(float | None, request.top_p)

    if request.stop is not NOT_GIVEN:
        if isinstance(request.stop, str):
            generation_config.stop_sequences = [request.stop]
        elif isinstance(request.stop, list):
            generation_config.stop_sequences = request.stop

    return generation_config


def format_messages(
    messages: typing.Iterable[openai.types.chat.ChatCompletionMessageParam],
) -> tuple[list[google.genai.types.ContentUnion], str | None]:
    """メッセージをVertex AI（Gemini）の形式に変換します。"""
    formatted_messages: list[google.genai.types.ContentUnion] = []
    system_instruction: str | None = None

    for message in messages:
        if message["role"] == "system":
            # システムメッセージは system_instruction として扱う
            content = message.get("content")
            if content and isinstance(content, str):
                system_instruction = content
        elif message["role"] in ("user", "assistant"):
            # ユーザーとアシスタントのメッセージを変換
            parts = []
            content: (
                str
                | typing.Iterable[
                    (
                        openai.types.chat.ChatCompletionContentPartTextParam
                        | openai.types.chat.ChatCompletionContentPartImageParam
                        | openai.types.chat.ChatCompletionContentPartInputAudioParam
                        | openai.types.chat.ChatCompletionContentPartRefusalParam
                        | openai.types.chat.chat_completion_content_part_param.File
                    )
                ]
                | None
            ) = message.get("content")

            if content:
                if isinstance(content, str):
                    parts.append(google.genai.types.Part(text=content))
                elif isinstance(content, list):
                    for part in content:
                        if part.get("type") == "text":
                            parts.append(google.genai.types.Part(text=part["text"]))
                        elif part.get("type") == "image_url":
                            # 画像URLの処理（必要に応じて実装）
                            logger.warning(
                                "Image URL content is not fully supported yet"
                            )

            # ツール呼び出しの処理（アシスタントメッセージの場合）
            if message["role"] == "assistant":
                tool_calls: list[dict[str, typing.Any]] = message.get("tool_calls", [])
                for tool_call in tool_calls:
                    if tool_call.get("type") == "function":
                        function = tool_call.get("function", {})
                        # ツール呼び出しをPartとして追加
                        parts.append(
                            google.genai.types.Part(
                                function_call=google.genai.types.FunctionCall(
                                    name=function.get("name", ""),
                                    args=json.loads(function.get("arguments", "{}")),
                                )
                            )
                        )

            if parts:
                formatted_messages.append(
                    google.genai.types.Content(role=message["role"], parts=parts)
                )

    return formatted_messages, system_instruction
