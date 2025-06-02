"""AWS関連のリクエスト処理モジュール。"""

import base64
import json
import logging
import pathlib
import typing

import openai.types.chat
import openai.types.chat.chat_completion_assistant_message_param
import openai.types.chat.chat_completion_content_part_param
import pytilpack.data_url
import types_aiobotocore_bedrock_runtime.literals as bedrock_literals
import types_aiobotocore_bedrock_runtime.type_defs as bedrock_types
from openai._types import NotGiven

import types_chat

logger = logging.getLogger(__name__)


class ConverseRequestTypeDef(typing.TypedDict):
    """BedrockのConverse APIのリクエストの型定義。"""

    modelId: str
    messages: typing.NotRequired[typing.Sequence[bedrock_types.MessageUnionTypeDef]]
    system: typing.NotRequired[typing.Sequence[bedrock_types.SystemContentBlockTypeDef]]
    inferenceConfig: typing.NotRequired[bedrock_types.InferenceConfigurationTypeDef]
    toolConfig: typing.NotRequired[bedrock_types.ToolConfigurationTypeDef]
    # guardrailConfig: typing.NotRequired[
    #     bedrock_types.GuardrailConfigurationTypeDef |
    #     bedrock_types.GuardrailStreamConfigurationTypeDef
    # ]
    additionalModelRequestFields: typing.NotRequired[typing.Mapping[str, typing.Any]]
    promptVariables: typing.NotRequired[
        typing.Mapping[str, bedrock_types.PromptVariableValuesTypeDef]
    ]
    additionalModelResponseFieldPaths: typing.NotRequired[typing.Sequence[str]]
    requestMetadata: typing.NotRequired[typing.Mapping[str, str]]
    performanceConfig: typing.NotRequired[bedrock_types.PerformanceConfigurationTypeDef]


def convert_request(request: types_chat.ChatRequest) -> ConverseRequestTypeDef:
    """リクエストをAWS Bedrockの形式に変換します。"""
    messages, system = _format_messages(request.messages)
    inference_config = _make_inference_config(request)
    tool_config = _make_tool_config(request.tools, request.tool_choice)
    kwargs: ConverseRequestTypeDef = {
        "modelId": request.model,
        "messages": messages,
        "system": system,
        "inferenceConfig": inference_config,
        # "additionalModelRequestFields": {},
        # "guardrailConfig": {"guardrailVersion": "", "guardrailIdentifier": ""},
    }
    if tool_config is not None:
        kwargs["toolConfig"] = tool_config
    return kwargs


def _make_inference_config(
    request: types_chat.ChatRequest,
) -> bedrock_types.InferenceConfigurationTypeDef:
    """推論設定を作成。"""
    # 未サポートのパラメータをチェック
    if not isinstance(request.response_format, NotGiven):
        logger.warning("response_format is not supported in AWS Bedrock implementation")
    if not isinstance(request.seed, NotGiven):
        logger.warning("seed is not supported in AWS Bedrock implementation")
    if not isinstance(request.web_search_options, NotGiven):
        logger.warning(
            "web_search_options is not supported in AWS Bedrock implementation"
        )
    if not isinstance(request.presence_penalty, NotGiven):
        logger.warning(
            "presence_penalty is not supported in AWS Bedrock implementation"
        )
    if not isinstance(request.frequency_penalty, NotGiven):
        logger.warning(
            "frequency_penalty is not supported in AWS Bedrock implementation"
        )
    if not isinstance(request.logprobs, NotGiven):
        logger.warning("logprobs is not supported in AWS Bedrock implementation")
    if not isinstance(request.top_logprobs, NotGiven):
        logger.warning("top_logprobs is not supported in AWS Bedrock implementation")

    inference_config: bedrock_types.InferenceConfigurationTypeDef = {}
    if (
        not isinstance(request.max_completion_tokens, NotGiven)
        and request.max_completion_tokens is not None
    ):
        inference_config["maxTokens"] = request.max_completion_tokens
    if not isinstance(request.stop, NotGiven) and request.stop is not None:
        inference_config["stopSequences"] = (
            request.stop if isinstance(request.stop, list) else [request.stop]
        )
    if (
        not isinstance(request.temperature, NotGiven)
        and request.temperature is not None
    ):
        inference_config["temperature"] = request.temperature
    if not isinstance(request.top_p, NotGiven) and request.top_p is not None:
        inference_config["topP"] = float(request.top_p)
    return inference_config


def _format_messages(
    messages: typing.Iterable[openai.types.chat.ChatCompletionMessageParam],
) -> tuple[
    list[bedrock_types.MessageUnionTypeDef],
    list[bedrock_types.SystemContentBlockTypeDef],
]:
    """メッセージをBedrockの形式に変換します。"""
    formatted_messages: list[bedrock_types.MessageUnionTypeDef] = []
    for message in messages:
        if message["role"] == "tool":
            tool_result = _to_bedrock_tool_result(message)

            # 複数あったらまとめる
            if (
                len(formatted_messages) > 0
                and formatted_messages[-1].get("role") == "user"
            ):
                # 最後のメッセージに追加
                assert isinstance(formatted_messages[-1]["content"], list)
                formatted_messages[-1]["content"].append(tool_result)
            else:
                # 新しいメッセージとして追加
                formatted_messages.append({"role": "user", "content": [tool_result]})

        elif message["role"] in ("user", "assistant"):
            formatted_messages.append(_to_bedrock_userassistant_message(message))
        else:
            continue  # 'developer', 'system', 'function' は無視

    system = sum(
        [
            typing.cast(
                list[bedrock_types.SystemContentBlockTypeDef],
                _to_bedrock_content_blocks(message["content"]),
            )
            for message in messages
            if message["role"] in ("developer", "system")
        ],
        [],
    )

    return formatted_messages, system


def _make_tool_config(
    tools: typing.Iterable[openai.types.chat.ChatCompletionToolParam] | NotGiven,
    tool_choice: openai.types.chat.ChatCompletionToolChoiceOptionParam | NotGiven,
) -> bedrock_types.ToolConfigurationTypeDef | None:
    """ツール設定を作成。"""
    if isinstance(tools, NotGiven):
        return None
    tools = list(tools)  # pydantic_core._pydantic_core.ValidatorIterator対策(?)
    if len(tools) == 0:
        return None  # ツールなし

    bedrock_tools: list[bedrock_types.ToolTypeDef] = []
    for tool in tools:
        if not isinstance(tool, dict):
            raise ValueError(f"Each tool must be a dict. {tool=}")
        function = tool.get("function")
        if function is None:
            raise ValueError(f"Tool must have 'function'. {tool=}")

        bedrock_tools.append(
            {
                "toolSpec": {
                    "name": function.get("name", ""),
                    "description": function.get("description", ""),
                    "inputSchema": {"json": function.get("parameters", {})},
                }
            }
        )

    bedrock_tool_choice: bedrock_types.ToolChoiceTypeDef
    if isinstance(tool_choice, NotGiven):
        bedrock_tool_choice = {"auto": {}}
    elif isinstance(tool_choice, dict):
        if tool_choice.get("type") != "function":
            raise ValueError(f"Invalid tool_choice: {tool_choice=}")
        bedrock_tool_choice = {
            "tool": {"name": tool_choice.get("function", {}).get("name", "")}
        }
    elif tool_choice == "auto":
        bedrock_tool_choice = {"auto": {}}
    elif tool_choice == "none":
        bedrock_tool_choice = {"auto": {}}  # not supported
    elif tool_choice == "required":
        bedrock_tool_choice = {"any": {}}
    else:
        raise ValueError(f"Invalid tool_choice: {tool_choice}.")

    return {"tools": bedrock_tools, "toolChoice": bedrock_tool_choice}


def _to_bedrock_userassistant_message(
    message: openai.types.chat.ChatCompletionMessageParam,
) -> bedrock_types.MessageUnionTypeDef:
    """ユーザー/アシスタントメッセージをBedrock形式に変換。"""
    role = typing.cast(typing.Literal["assistant", "user"], message["role"])
    bedrock_content: list[
        bedrock_types.ContentBlockUnionTypeDef | bedrock_types.ContentBlockOutputTypeDef
    ] = []

    content = message.get("content")
    if content is not None:
        bedrock_content.extend(_to_bedrock_content_blocks(content))

    # audio
    audio = message.get("audio")
    if audio is not None:
        logger.warning(
            f"Audio content is not supported in this implementation. {audio=}"
        )

    # tool_calls
    tool_calls = message.get("tool_calls")
    if tool_calls is not None:
        tool_calls = typing.cast(
            list[openai.types.chat.ChatCompletionMessageToolCallParam], tool_calls
        )
        for tool_call in tool_calls:
            if tool_call.get("type") != "function":
                raise ValueError(
                    f"Unsupported tool call type: {tool_call=}. " "Expected 'function'."
                )
            function = tool_call.get("function")
            if function is None:
                raise ValueError(f"Tool call must have 'function'. {tool_call=}")

            # argumentsはJSONパースする必要がある
            arguments = function.get("arguments", "")
            try:
                if arguments:
                    parsed_args = json.loads(arguments)
                else:
                    parsed_args = {}
            except json.JSONDecodeError:
                parsed_args = {}

            bedrock_content.append(
                {
                    "toolUse": {
                        "toolUseId": tool_call["id"],
                        "name": function.get("name", ""),
                        "input": parsed_args,
                    }
                }
            )

    # refusal
    refusal = message.get("refusal")
    if refusal is not None:
        refusal = typing.cast(str, refusal)
        bedrock_content.append({"guardContent": {"text": {"text": refusal}}})
    return {"role": role, "content": bedrock_content}


def _to_bedrock_tool_result(
    message: openai.types.chat.ChatCompletionToolMessageParam,
) -> bedrock_types.ContentBlockTypeDef:
    """ツール結果メッセージをBedrock形式に変換。"""
    tool_call_id = message.get("tool_call_id")
    if tool_call_id is None:
        raise ValueError(f"Tool message must have 'tool_call_id'. {message=}")

    bedrock_content: list[bedrock_types.ToolResultContentBlockTypeDef] = []
    content: (
        str
        | typing.Iterable[openai.types.chat.ChatCompletionContentPartTextParam]
        | None
    ) = message.get("content")
    if content is not None:
        if isinstance(content, str):
            # 文字列ならそのまま
            bedrock_content.append({"text": content})
        elif isinstance(content, typing.Iterable):
            content = typing.cast(
                typing.Iterable[openai.types.chat.ChatCompletionContentPartTextParam],
                content,
            )
            for c in content:
                if c.get("type") == "text":
                    c_text = c.get("text", "")
                    bedrock_content.append({"text": c_text})
                else:
                    raise ValueError(f"Invalid content: {c}.")
        else:
            raise ValueError(f"Invalid content: {content}.")

    # 文字列がJSONっぽければJSON扱いにしちゃう
    for i, bc in enumerate(bedrock_content):
        try:
            data = json.loads(bc["text"])
            if isinstance(data, dict):
                bedrock_content[i] = {"json": data}
        except json.JSONDecodeError:
            pass

    return {
        "toolResult": {
            "toolUseId": tool_call_id,
            "content": bedrock_content,
            # "status": "error"  ⇒ OpenAI APIにはこれに相当するものが無い…
        }
    }


def _to_bedrock_content_blocks(
    value: (
        str
        | typing.Iterable[
            openai.types.chat.chat_completion_content_part_param.ChatCompletionContentPartParam
        ]
        | typing.Iterable[
            openai.types.chat.chat_completion_assistant_message_param.ContentArrayOfContentPart
        ]
        | None
    ),
) -> list[bedrock_types.ContentBlockTypeDef]:
    """コンテンツをBedrock形式のブロックリストに変換。"""
    if value is None:
        return []
    if isinstance(value, str):
        return [{"text": value}]
    if isinstance(value, typing.Iterable):
        return [_to_bedrock_content_block(v) for v in value]  # type: ignore[arg-type]
    raise ValueError(
        f"Invalid content: {value=}. "
        "Expected str or iterable of OpenAI content parts."
    )


def _to_bedrock_content_block(
    value: (
        openai.types.chat.chat_completion_content_part_param.ChatCompletionContentPartParam
        | openai.types.chat.chat_completion_assistant_message_param.ContentArrayOfContentPart
    ),
) -> bedrock_types.ContentBlockTypeDef:
    """OpenAIのコンテンツをBedrockの形式に変換します。"""

    # class ContentBlockTypeDef(TypedDict):
    #     text: NotRequired[str]
    #     image: NotRequired[ImageBlockUnionTypeDef]
    #     document: NotRequired[DocumentBlockUnionTypeDef]
    #     video: NotRequired[VideoBlockUnionTypeDef]
    #     toolUse: NotRequired[ToolUseBlockUnionTypeDef]
    #     toolResult: NotRequired[ToolResultBlockUnionTypeDef]
    #     guardContent: NotRequired[GuardrailConverseContentBlockUnionTypeDef]
    #     reasoningContent: NotRequired[ReasoningContentBlockUnionTypeDef]

    match value.get("type"):
        case "text":
            text = value.get("text")
            if text is None:
                raise ValueError(f"Text content must have 'text' field. {value=}")
            text = str(text)
            return {"text": text}
        case "image_url":
            image_url = value.get("image_url")
            if image_url is None:
                raise ValueError(
                    f"Image URL content must have 'image_url' field. {value=}"
                )
            image_url = typing.cast(
                openai.types.chat.chat_completion_content_part_image_param.ImageURL,
                image_url,
            )
            url = image_url.get("url")
            if url is None:
                raise ValueError(
                    f"Image URL content must have 'url' field. {image_url=}"
                )
            # detail = image_url.get("detail")
            data_url = pytilpack.data_url.parse(url)
            if data_url.mime_type.startswith("image/"):
                # ["gif", "jpeg", "png", "webp"]
                image_format = {
                    "image/gif": "gif",
                    "image/jpeg": "jpeg",
                    "image/webp": "webp",
                    "image/png": "png",
                }.get(data_url.mime_type)
                if image_format is None:
                    raise ValueError(
                        f"Unsupported MIME type: {data_url.mime_type}. "
                        "Expected image."
                    )
                image_format = typing.cast(
                    bedrock_literals.ImageFormatType, image_format
                )
                return {
                    "image": {
                        "format": image_format,
                        "source": {"bytes": data_url.data},
                    }
                }
            elif data_url.mime_type.startswith("video/"):
                # ["flv", "mkv", "mov", "mp4", "mpeg", "mpg", "three_gp", "webm", "wmv"]
                video_format = {
                    "video/x-flv": "flv",
                    "video/x-matroska": "mkv",
                    "video/quicktime": "mov",
                    "video/mp4": "mp4",
                    "video/mpeg": "mpeg",
                    "video/3gpp": "three_gp",
                    "video/webm": "webm",
                    "video/x-ms-wmv": "wmv",
                    "video/x-msvideo": "avi",
                    "video/ogg": "ogg",
                }.get(data_url.mime_type)
                if video_format is None:
                    raise ValueError(
                        f"Unsupported MIME type: {data_url.mime_type}. "
                        "Expected video."
                    )
                video_format = typing.cast(
                    bedrock_literals.VideoFormatType, video_format
                )
                return {
                    "video": {
                        "format": video_format,
                        "source": {"bytes": data_url.data},
                    }
                }
            else:
                # ["csv", "doc", "docx", "html", "md", "pdf", "txt", "xls", "xlsx"]
                doc_format = {
                    "text/csv": "csv",
                    "application/msword": "doc",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
                    "text/html": "html",
                    "text/markdown": "md",
                    "application/pdf": "pdf",
                    "text/plain": "txt",
                    "application/vnd.ms-excel": "xls",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
                }.get(data_url.mime_type)
                doc_format = typing.cast(
                    bedrock_literals.DocumentFormatType, doc_format
                )
                if doc_format is None:
                    raise ValueError(
                        f"Unsupported MIME type: {data_url.mime_type}. "
                        "Expected image/video/document."
                    )
                return {
                    "document": {
                        "name": f"data.{doc_format}",  # dummy
                        "format": doc_format,
                        "source": {"bytes": data_url.data},
                    }
                }
        case "input_audio":
            # format_ = value["input_audio"].get("format")  # "wav" or "mp3"
            # data = value["input_audio"].get("data", "")  # Base64 encoded data.
            raise ValueError("Input audio is not supported in this implementation.")
        case "file":
            # .venv/lib/python3.12/site-packages/openai/types/chat/chat_completion_content_part_param.py
            file = typing.cast(
                openai.types.chat.chat_completion_content_part_param.FileFile,
                value.get("file", {}),
            )
            file_data = file.get("file_data", "")  # Base64 encoded data.
            # file_id = file.get("file_id")  # uploaded file ID.
            filename = file.get("filename")  # Original filename.
            if file_data is None:
                # 手抜き: とりあえずファイルデータ必須にする
                raise ValueError(
                    "File data is required for file content in this implementation."
                )
            suffix = pathlib.Path(filename).suffix if filename else ""
            # ["gif", "jpeg", "jpg", "png", "webp"]
            # ["flv", "mkv", "mov", "mp4", "mpeg", "mpg", "three_gp", "webm", "wmv"]
            # ["pdf", "txt", "csv", "html", "doc", "docx", "xls", "xlsx"]
            type_ = {
                ".gif": "image",
                ".jpeg": "image",
                ".jpg": "image",
                ".png": "image",
                ".webp": "image",
                ".flv": "video",
                ".mkv": "video",
                ".mov": "video",
                ".mp4": "video",
                ".avi": "video",
                ".mpeg": "video",
                ".mpg": "video",
                ".3gp": "video",
                ".webm": "video",
                ".wmv": "video",
                ".pdf": "document",
                ".txt": "document",
                ".csv": "document",
                ".html": "document",
                ".doc": "document",
                ".docx": "document",
                ".xls": "document",
                ".xlsx": "document",
            }.get(suffix.lower())
            format_ = suffix.removeprefix(".")
            format_ = {
                # ほぼ拡張子のままいけるがダメなやつだけ置換
                "3gp": "three_gp"
            }.get(format_, format_)
            if type_ == "image":
                format_ = typing.cast(bedrock_literals.ImageFormatType, format_)
                return {
                    "image": {
                        "format": format_,
                        "source": {"bytes": base64.b64decode(file_data)},
                    }
                }
            elif type_ == "video":
                format_ = typing.cast(bedrock_literals.VideoFormatType, format_)
                return {
                    "video": {
                        "format": format_,
                        "source": {"bytes": base64.b64decode(file_data)},
                    }
                }
            elif type_ == "document":
                format_ = typing.cast(bedrock_literals.DocumentFormatType, format_)
                return {
                    "document": {
                        "name": f"data.{format_}",  # dummy
                        "format": format_,
                        "source": {"bytes": base64.b64decode(file_data)},
                    }
                }
            else:
                raise ValueError(
                    f"Unsupported file type: {filename}. "
                    "Expected image/video/document."
                )
        case "refusal":
            refusal = value.get("refusal")
            refusal = typing.cast(str, refusal)
            return {"guardContent": {"text": {"text": refusal}}}
        case _:
            raise ValueError(
                f"Unsupported content type: {value.get('type')}. "
                "Expected 'text', 'image_url', 'file_url', or 'refusal'."
            )
