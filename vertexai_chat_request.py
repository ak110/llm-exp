"""VertexAI関連のリクエスト処理モジュール。"""

import json
import logging
import typing

import google.genai.types
import openai.types.chat
import openai.types.chat.chat_completion_content_part_param
from openai._types import NotGiven

import errors
import types_chat

logger = logging.getLogger(__name__)


def convert_request(
    request: types_chat.ChatRequest,
) -> tuple[
    google.genai.types.GenerateContentConfig, list[google.genai.types.ContentUnion]
]:
    formatted_messages, system_instruction = _format_messages(request.messages)
    generation_config = _make_generation_config(request)
    if system_instruction is not None:
        generation_config.system_instruction = system_instruction
    return generation_config, formatted_messages


def _make_generation_config(
    request: types_chat.ChatRequest,
) -> google.genai.types.GenerateContentConfig:
    """生成設定を作成します。"""
    generation_config = google.genai.types.GenerateContentConfig()

    if not isinstance(request.audio, NotGiven):
        logger.warning("audioパラメータはVertex AI実装ではサポートされていません")

    if not isinstance(request.frequency_penalty, NotGiven):
        generation_config.frequency_penalty = request.frequency_penalty

    if not isinstance(request.logit_bias, NotGiven):
        logger.warning("logit_biasパラメータはVertex AI実装ではサポートされていません")

    if not isinstance(request.logprobs, NotGiven):
        generation_config.logprobs = request.logprobs

    if not isinstance(request.max_completion_tokens, NotGiven):
        generation_config.max_output_tokens = request.max_completion_tokens

    # metadataは生成設定には影響しない

    if not isinstance(request.modalities, NotGiven):
        logger.warning("modalitiesパラメータはVertex AI実装ではサポートされていません")

    if not isinstance(request.n, NotGiven):
        generation_config.candidate_count = request.n

    if not isinstance(request.parallel_tool_calls, NotGiven):
        logger.warning(
            "parallel_tool_callsパラメータはVertex AI実装ではサポートされていません"
        )

    # predictionは生成設定には影響しない

    if not isinstance(request.presence_penalty, NotGiven):
        generation_config.presence_penalty = request.presence_penalty

    if not isinstance(request.reasoning_effort, NotGiven):
        logger.warning(
            "reasoning_effortパラメータはVertex AI実装ではサポートされていません"
        )

    if not isinstance(request.response_format, NotGiven):
        response_format_type = request.response_format.get("type")
        if response_format_type == "text":
            pass
        elif response_format_type == "json_schema":
            generation_config.response_schema = request.response_format["json_schema"]  # type: ignore
        else:
            logger.warning(
                f"サポートされていないレスポンスフォーマットタイプです: {response_format_type}。"
                "テキストレスポンスにデフォルトで戻します。"
            )

    if not isinstance(request.seed, NotGiven):
        generation_config.seed = request.seed

    # service_tierは生成設定には影響しない

    if not isinstance(request.stop, NotGiven):
        if isinstance(request.stop, str):
            generation_config.stop_sequences = [request.stop]
        elif isinstance(request.stop, list):
            generation_config.stop_sequences = request.stop

    # storeは生成設定には影響しない
    # streamは生成設定には影響しない
    # stream_optionsは生成設定には影響しない

    if not isinstance(request.temperature, NotGiven):
        generation_config.temperature = request.temperature

    if not isinstance(request.tools, NotGiven):
        tools: google.genai.types.ToolListUnion = []
        for tool in request.tools:
            if tool["type"] == "function":
                function = tool["function"]
                tools.append(
                    google.genai.types.Tool(
                        function_declarations=[
                            google.genai.types.FunctionDeclaration(
                                name=function.get("name", ""),
                                description=function.get("description", ""),
                                parameters=function.get("parameters"),  # type: ignore
                            )
                        ]
                    )
                )
        generation_config.tools = tools

    if not isinstance(request.tool_choice, NotGiven):
        if isinstance(request.tool_choice, dict):
            if request.tool_choice.get("type") != "function":
                raise errors.InvalidParameterValue(
                    f"サポートされていないツール選択です: {request.tool_choice}",
                    param="tool_choice",
                )
            generation_config.tool_config = google.genai.types.ToolConfig(
                function_calling_config=google.genai.types.FunctionCallingConfig(
                    allowed_function_names=[
                        request.tool_choice.get("function", {}).get("name", "")
                    ]
                )
            )
        elif request.tool_choice == "none":
            generation_config.tool_config = google.genai.types.ToolConfig(
                function_calling_config=google.genai.types.FunctionCallingConfig(
                    mode=google.genai.types.FunctionCallingConfigMode.NONE
                )
            )
        elif request.tool_choice == "auto":
            generation_config.tool_config = google.genai.types.ToolConfig(
                function_calling_config=google.genai.types.FunctionCallingConfig(
                    mode=google.genai.types.FunctionCallingConfigMode.AUTO
                )
            )
        elif request.tool_choice == "required":
            generation_config.tool_config = google.genai.types.ToolConfig(
                function_calling_config=google.genai.types.FunctionCallingConfig(
                    mode=google.genai.types.FunctionCallingConfigMode.ANY
                )
            )
        else:
            raise errors.InvalidParameterValue(
                f"サポートされていないツール選択です: {request.tool_choice}",
                param="tool_choice",
            )

    if not isinstance(request.top_logprobs, NotGiven):
        generation_config.logprobs = request.top_logprobs

    if not isinstance(request.top_p, NotGiven):
        generation_config.top_p = request.top_p

    # userは生成設定には影響しない

    if not isinstance(request.web_search_options, NotGiven):
        logger.warning("web_search_optionsはVertex AI実装ではサポートされていません")

    generation_config.safety_settings = [
        google.genai.types.SafetySetting(
            category=google.genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=google.genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        ),
        google.genai.types.SafetySetting(
            category=google.genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=google.genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        ),
        google.genai.types.SafetySetting(
            category=google.genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=google.genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        ),
        google.genai.types.SafetySetting(
            category=google.genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=google.genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        ),
    ]
    return generation_config


def _format_messages(
    messages: typing.Iterable[openai.types.chat.ChatCompletionMessageParam],
) -> tuple[list[google.genai.types.ContentUnion], str | None]:
    """メッセージをVertex AI（Gemini）の形式に変換します。"""
    formatted_messages: list[google.genai.types.ContentUnion] = []
    system_instruction_parts: list[str] = []

    for message in messages:
        role = message.get("role")
        if role in ("system", "developer"):
            # システムメッセージは system_instruction として扱う
            system_content = message.get("content")
            if system_content is not None:
                if isinstance(system_content, str):
                    system_instruction_parts.append(system_content)
                elif isinstance(system_content, typing.Iterable):
                    # システムメッセージが複数のパートを持つ場合
                    system_instruction_parts.extend(
                        part.get("text", "") for part in system_content  # type: ignore
                    )
                else:
                    logger.warning(
                        f"システムメッセージの内容が文字列ではありません: {system_content=}。"
                        "システムメッセージをスキップします。"
                    )
        elif role in ("user", "assistant"):
            # ユーザーとアシスタントのメッセージを変換
            parts: list[google.genai.types.Part] = []
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

            if content is not None:
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
                                raise errors.InvalidParameterValue(
                                    f"サポートされていない画像URLです: {image_url}。インラインデータを期待しています",
                                    param="image_url",
                                )
                        elif part.get("type") == "file":
                            # ファイル添付の処理
                            raise errors.InvalidParameterValue(
                                "ファイル添付はまだ完全にサポートされていません",
                                param="file",
                            )
                        else:
                            raise errors.InvalidParameterValue(
                                f"サポートされていないコンテンツ部分です: {part}",
                                param="content_part",
                            )
                else:
                    raise errors.InvalidParameterValue(
                        f"サポートされていないコンテンツです: {content}。文字列またはコンテンツ部分のリストを期待しています",
                        param="content",
                    )

            # ツール呼び出しの処理（アシスタントメッセージの場合）
            if role == "assistant":
                tool_calls = message.get("tool_calls")
                if tool_calls is not None:
                    for tool_call in typing.cast(
                        list[openai.types.chat.ChatCompletionMessageToolCallParam],
                        tool_calls,
                    ):
                        if tool_call.get("type") == "function":
                            function = tool_call.get("function", {})
                            # ツール呼び出しをPartとして追加
                            parts.append(
                                google.genai.types.Part(
                                    function_call=google.genai.types.FunctionCall(
                                        name=function.get("name", ""),
                                        args=json.loads(
                                            function.get("arguments", "{}")
                                        ),
                                    )
                                )
                            )
                        else:
                            raise errors.InvalidParameterValue(
                                f"サポートされていないツール呼び出しです: {tool_call}",
                                param="tool_call",
                            )

            if len(parts) > 0:
                formatted_messages.append(
                    google.genai.types.Content(role=role, parts=parts)
                )
        elif role == "tool":
            # ツールメッセージを変換
            tool_call_message = typing.cast(
                openai.types.chat.ChatCompletionToolMessageParam, message
            )
            content = tool_call_message.get("content")
            tool_call_id = tool_call_message.get("tool_call_id")

            if content is not None and tool_call_id is not None:
                response_content: str = ""
                if isinstance(content, str):
                    response_content = content
                elif isinstance(content, typing.Iterable):
                    # 複数のパートがある場合はテキストを結合
                    response_content = "\n".join(
                        part.get("text", "") for part in content  # type: ignore
                    )
                else:
                    logger.warning(
                        f"ツールメッセージの内容が文字列ではありません: {content=}。"
                        "ツールメッセージをスキップします。"
                    )

                # FunctionResponseとしてPartsに追加
                parts = [
                    google.genai.types.Part(
                        function_response=google.genai.types.FunctionResponse(
                            name=_find_tool_name_by_id(messages, tool_call_id),
                            response={"output": response_content},
                        )
                    )
                ]
                formatted_messages.append(
                    google.genai.types.Content(role="user", parts=parts)
                )
        else:
            raise errors.InvalidParameterValue(
                f"サポートされていないメッセージです: {message}", param="message"
            )

    system_instruction = (
        "\n".join(system_instruction_parts)
        if len(system_instruction_parts) > 0
        else None
    )

    return formatted_messages, system_instruction


def _find_tool_name_by_id(
    messages: typing.Iterable[openai.types.chat.ChatCompletionMessageParam],
    tool_call_id: str,
) -> str:
    """ツール呼び出しIDからツール名を取得します。"""
    for message in messages:
        if message.get("role") == "assistant":
            message = typing.cast(
                openai.types.chat.ChatCompletionAssistantMessageParam, message
            )
            tool_calls = list(message.get("tool_calls", []))
            for tool_call in tool_calls:
                if tool_call.get("id") == tool_call_id:
                    return tool_call.get("function", {}).get("name", "")
    raise errors.InvalidParameterValue(
        f"ツール呼び出しID '{tool_call_id}' がメッセージ内に見つかりません",
        param="tool_call_id",
    )
