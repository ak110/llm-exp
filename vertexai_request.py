"""VertexAI関連のリクエスト処理モジュール。"""

import json
import logging
import typing

import google.genai.types
import openai.types.chat
from openai._types import NotGiven

import types_chat

logger = logging.getLogger(__name__)


def make_generation_config(
    request: types_chat.ChatRequest,
) -> google.genai.types.GenerateContentConfigOrDict:
    """生成設定を作成します。"""
    generation_config = google.genai.types.GenerateContentConfig()

    if not isinstance(request.audio, NotGiven):
        logger.warning("audio parameter is not supported in Vertex AI implementation")

    if not isinstance(request.frequency_penalty, NotGiven):
        generation_config.frequency_penalty = request.frequency_penalty

    if not isinstance(request.logit_bias, NotGiven):
        logger.warning(
            "logit_bias parameter is not supported in Vertex AI implementation"
        )

    if not isinstance(request.logprobs, NotGiven):
        generation_config.logprobs = request.logprobs

    if not isinstance(request.max_completion_tokens, NotGiven):
        generation_config.max_output_tokens = request.max_completion_tokens

    # metadataは生成設定には影響しない

    if not isinstance(request.modalities, NotGiven):
        logger.warning(
            "modalities parameter is not supported in Vertex AI implementation"
        )

    if not isinstance(request.n, NotGiven):
        generation_config.candidate_count = request.n

    if not isinstance(request.parallel_tool_calls, NotGiven):
        logger.warning(
            "parallel_tool_calls parameter is not supported in Vertex AI implementation"
        )

    # predictionは生成設定には影響しない

    if not isinstance(request.presence_penalty, NotGiven):
        generation_config.presence_penalty = request.presence_penalty

    if not isinstance(request.reasoning_effort, NotGiven):
        logger.warning(
            "reasoning_effort parameter is not supported in Vertex AI implementation"
        )

    if not isinstance(request.response_format, NotGiven):
        response_format_type = request.response_format.get("type")
        if response_format_type == "text":
            pass
        elif response_format_type == "json_schema":
            generation_config.response_schema = request.response_format["json_schema"]  # type: ignore
        else:
            logger.warning(
                f"Unsupported response format type: {response_format_type}. "
                "Defaulting to text response."
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

    if not isinstance(request.tool_choice, NotGiven):
        if isinstance(request.tool_choice, dict):
            if request.tool_choice.get("type") != "function":
                raise ValueError(f"Unsupported tool choice: {request.tool_choice}")
            generation_config.tool_config = google.genai.types.ToolConfig(
                function_call_behavior=google.genai.types.FunctionCall(  # type: ignore
                    name=request.tool_choice.get("function", {}).get("name", "")
                )
            )
        elif request.tool_choice == "none":
            generation_config.tool_config = google.genai.types.ToolConfig(
                function_call_behavior=google.genai.types.FunctionCall(
                    name=google.genai.types.FunctionCallingConfigMode.NONE
                )
            )
        elif request.tool_choice == "auto":
            generation_config.tool_config = google.genai.types.ToolConfig(
                function_call_behavior=google.genai.types.FunctionCall(
                    name=google.genai.types.FunctionCallingConfigMode.AUTO
                )
            )
        elif request.tool_choice == "required":
            generation_config.tool_config = google.genai.types.ToolConfig(
                function_call_behavior=google.genai.types.FunctionCall(
                    name=google.genai.types.FunctionCallingConfigMode.ANY
                )
            )
        else:
            raise ValueError(f"Unsupported tool choice: {request.tool_choice}")

    if not isinstance(request.top_logprobs, NotGiven):
        generation_config.logprobs = request.top_logprobs

    if not isinstance(request.top_p, NotGiven):
        generation_config.top_p = request.top_p

    # userは生成設定には影響しない

    if not isinstance(request.web_search_options, NotGiven):
        logger.warning(
            "web_search_options is not supported in Vertex AI implementation"
        )

    return generation_config


def format_messages(
    messages: typing.Iterable[openai.types.chat.ChatCompletionMessageParam],
) -> tuple[list[google.genai.types.ContentUnion], str | None]:
    """メッセージをVertex AI（Gemini）の形式に変換します。"""
    formatted_messages: list[google.genai.types.ContentUnion] = []
    system_instruction: str | None = None

    for message in messages:
        role = message.get("role")
        if role in ("system", "developer"):
            # システムメッセージは system_instruction として扱う
            content = message.get("content")
            if content is not None:
                if isinstance(content, str):
                    system_instruction = content
                elif isinstance(content, typing.Iterable):
                    # システムメッセージが複数のパートを持つ場合
                    system_instruction = "\n".join(
                        part.get("text", "") for part in content
                    )
                else:
                    logger.warning(
                        f"System message content is not a string: {content=}. "
                        "Skipping system message."
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
                        elif part.get("type") == "file":
                            # ファイル添付の処理
                            logger.warning(
                                "File attachments are not fully supported yet"
                            )
                        else:
                            logger.warning(f"Unsupported content part type: {part=}")

            # ツール呼び出しの処理（アシスタントメッセージの場合）
            if role == "assistant":
                tool_calls = message.get("tool_calls", [])
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
                    else:
                        logger.warning(f"Unsupported tool call: {tool_call=}")

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
                if isinstance(content, str):
                    response_content = content
                elif isinstance(content, typing.Iterable):
                    # 複数のパートがある場合はテキストを結合
                    response_content = "\n".join(
                        part.get("text", "") for part in content
                    )
                else:
                    logger.warning(
                        f"Tool message content is not a string: {content=}. "
                        "Skipping tool message."
                    )
                    response_content: str = ""

                # FunctionResponseとしてPartsに追加

                # 参考:
                # class FunctionResponseScheduling(_common.CaseInSensitiveEnum):
                #   """会話における応答のスケジュール方法を指定します。"""
                #   SCHEDULING_UNSPECIFIED = 'SCHEDULING_UNSPECIFIED'
                #   """この値は使用されません。"""
                #   SILENT = 'SILENT'
                #   """結果を会話コンテキストに追加するだけで、進行中の処理を中断したり生成を開始させたりしません。"""
                #   WHEN_IDLE = 'WHEN_IDLE'
                #   """結果を会話コンテキストに追加し、進行中の生成処理を中断せずに出力生成を促すプロンプトを表示します。"""
                #   INTERRUPT = 'INTERRUPT'
                #   """結果を会話コンテキストに追加し、進行中の生成処理を中断した上で出力生成を促すプロンプトを表示します。"""

                parts = [
                    google.genai.types.Part(
                        function_response=google.genai.types.FunctionResponse(
                            name=_find_tool_name_by_id(messages, tool_call_id),
                            response={"output": response_content},
                            scheduling=google.genai.types.FunctionResponseScheduling.WHEN_IDLE,
                        )
                    )
                ]
                formatted_messages.append(
                    google.genai.types.Content(role="model", parts=parts)
                )
        else:
            # role == "function"は未サポート
            logger.warning(f"Unsupported message: {message=}")

    return formatted_messages, system_instruction


def _find_tool_name_by_id(
    messages: typing.Iterable[openai.types.chat.ChatCompletionMessageParam],
    tool_call_id: str,
) -> str:
    """ツール呼び出しIDからツール名を取得します。"""
    for message in messages:
        if message.get("role") == "assistant":
            tool_calls = message.get("tool_calls", [])
            for tool_call in tool_calls:
                if tool_call.get("id") == tool_call_id:
                    return tool_call.get("function", {}).get("name", "")
    return ""
