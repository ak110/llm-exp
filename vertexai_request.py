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
    # 未サポートのパラメータをチェック
    if request.web_search_options is not NOT_GIVEN:
        logger.warning(
            "web_search_options is not supported in Vertex AI implementation"
        )
    generation_config = google.genai.types.GenerateContentConfig()

    if request.temperature is not NOT_GIVEN:
        generation_config.temperature = typing.cast(float | None, request.temperature)

    if request.max_tokens is not NOT_GIVEN:
        generation_config.max_output_tokens = typing.cast(
            int | None, request.max_tokens
        )

    if request.n is not NOT_GIVEN:
        generation_config.candidate_count = typing.cast(int | None, request.n)

    if request.top_p is not NOT_GIVEN:
        generation_config.top_p = typing.cast(float | None, request.top_p)

    if request.presence_penalty is not NOT_GIVEN:
        generation_config.presence_penalty = typing.cast(
            float | None, request.presence_penalty
        )

    if request.frequency_penalty is not NOT_GIVEN:
        generation_config.frequency_penalty = typing.cast(
            float | None, request.frequency_penalty
        )

    if request.stop is not NOT_GIVEN:
        if isinstance(request.stop, str):
            generation_config.stop_sequences = [request.stop]
        elif isinstance(request.stop, list):
            generation_config.stop_sequences = request.stop

    if request.tools is not NOT_GIVEN:
        tools = []
        for tool in request.tools:
            if tool["type"] == "function":
                function = tool["function"]
                tools.append(
                    google.genai.types.Tool(
                        function_declarations=[
                            google.genai.types.FunctionDeclaration(
                                name=function.get("name", ""),
                                description=function.get("description", ""),
                                parameters=function.get("parameters", {}),
                            )
                        ]
                    )
                )
        generation_config.tools = tools

    if request.tool_choice is not NOT_GIVEN and request.tool_choice != "auto":
        if request.tool_choice["type"] == "function":
            generation_config.tool_config = google.genai.types.ToolConfig(
                function_call_behavior=google.genai.types.FunctionCall(
                    name=request.tool_choice["function"]["name"]
                )
            )

    if request.top_logprobs is not NOT_GIVEN:
        generation_config.logprobs = typing.cast(int | None, request.top_logprobs)

    if request.response_format is not NOT_GIVEN:
        response_format_type = request.response_format.get("type")
        if response_format_type == "text":
            pass
        elif response_format_type == "json_schema":
            generation_config.response_schema = request.response_format["json_schema"]
        else:
            logger.warning(
                f"Unsupported response format type: {response_format_type}. "
                "Defaulting to text response."
            )

    if request.seed is not NOT_GIVEN:
        generation_config.seed = typing.cast(int | None, request.seed)

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
                            image_url = part.get("image_url", {}).get("url", "")
                            # inline_dataの場合はbase64データを直接使用
                            if image_url.startswith("data:"):
                                parts.append(
                                    google.genai.types.Part(
                                        inline_data=google.genai.types.Blob(
                                            mime_type=part.get("image_url", {}).get(
                                                "mime_type", "image/jpeg"
                                            ),
                                            data=image_url.split(",")[1],
                                        )
                                    )
                                )
                            else:
                                parts.append(google.genai.types.Part(uri=image_url))
                        elif isinstance(
                            part,
                            openai.types.chat.chat_completion_content_part_param.File,
                        ):
                            if part.type == "file":
                                # ファイル添付の処理
                                logger.warning(
                                    "File attachments are not fully supported yet"
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
