#!/usr/bin/env python3
"""検証用コード。

```
下記をよく読み、vertexai.pyの実装を完成させてください。

- aws.py
- Google Gen AI SDK
    - <https://googleapis.github.io/python-genai/>
    - .venv/lib/python3.12/site-packages/google/genai/models.py など
- OpenAI API
    - <https://platform.openai.com/docs/api-reference/chat/create>
    - .venv/lib/python3.12/site-packages/openai/types/chat/chat_completion.py など
```

"""

import asyncio
import collections.abc
import json
import logging
import time
import typing
import uuid

import google.genai
import google.genai.types
import openai.types.chat
import openai.types.completion_usage
from openai._types import NOT_GIVEN

import config
import types_chat

logger = logging.getLogger(__name__)


class VertexAIClient:
    """VertexAIのクライアント。"""

    def __init__(self) -> None:
        """初期化処理。"""

    async def chat(
        self, request: types_chat.ChatRequest
    ) -> openai.types.chat.ChatCompletion:
        """OpenAIのChat Completions API互換API。"""
        assert not request.stream

        client = google.genai.Client(
            vertexai=True,
            project=config.GOOGLE_PROJECT_ID,
            location="us-central1",  # config.GOOGLE_REGION,
            http_options=google.genai.types.HttpOptions(api_version="v1"),
        )

        # メッセージの変換
        formatted_messages, system_instruction = self._format_messages(request.messages)

        # システムメッセージがある場合は、最初のメッセージとして追加
        if system_instruction:
            formatted_messages.insert(
                0,
                google.genai.types.Content(
                    role="user",
                    parts=[google.genai.types.Part(text=system_instruction)],
                ),
            )

        # 生成設定の作成
        generation_config = self._make_generation_config(request)

        # Vertex AIでチャット生成を実行
        response = await client.aio.models.generate_content(
            model=request.model, contents=formatted_messages, config=generation_config
        )

        return self._process_non_streaming_response(request, response)

    async def chat_stream(
        self, request: types_chat.ChatRequest
    ) -> collections.abc.AsyncGenerator[openai.types.chat.ChatCompletionChunk, None]:
        """OpenAIのChat Completions API互換API。(ストリーミング版)"""
        assert request.stream

        client = google.genai.Client(
            vertexai=True,
            project=config.GOOGLE_PROJECT_ID,
            location="us-central1",  # config.GOOGLE_REGION,
            http_options=google.genai.types.HttpOptions(api_version="v1"),
        )

        # メッセージの変換
        formatted_messages, system_instruction = self._format_messages(request.messages)

        # システムメッセージがある場合は、最初のメッセージとして追加
        if system_instruction:
            formatted_messages.insert(
                0,
                google.genai.types.Content(
                    role="user",
                    parts=[google.genai.types.Part(text=system_instruction)],
                ),
            )

        # 生成設定の作成
        generation_config = self._make_generation_config(request)

        # Vertex AIでストリーミングチャット生成を実行
        stream = await client.aio.models.generate_content_stream(
            model=request.model, contents=formatted_messages, config=generation_config
        )

        async for response_chunk in stream:
            if chunk := self._process_stream_chunk(request, response_chunk):
                yield chunk

    def _format_messages(
        self, messages: typing.Iterable[openai.types.chat.ChatCompletionMessageParam]
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
                content = message.get("content")

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
                    tool_calls = message.get("tool_calls", [])
                    for tool_call in tool_calls:
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

                if parts:
                    formatted_messages.append(
                        google.genai.types.Content(role=message["role"], parts=parts)
                    )

        return formatted_messages, system_instruction

    def _make_generation_config(
        self, request: types_chat.ChatRequest
    ) -> google.genai.types.GenerateContentConfigOrDict:
        """生成設定を作成します。"""
        generation_config = google.genai.types.GenerateContentConfig()

        if request.temperature is not NOT_GIVEN:
            generation_config.temperature = request.temperature

        if request.max_tokens is not NOT_GIVEN:
            generation_config.max_output_tokens = request.max_tokens

        if request.top_p is not NOT_GIVEN:
            generation_config.top_p = request.top_p

        if request.stop is not NOT_GIVEN:
            if isinstance(request.stop, str):
                generation_config.stop_sequences = [request.stop]
            elif isinstance(request.stop, list):
                generation_config.stop_sequences = request.stop

        return generation_config

    def _process_non_streaming_response(
        self,
        request: types_chat.ChatRequest,
        response: google.genai.types.GenerateContentResponse,
    ) -> openai.types.chat.ChatCompletion:
        """非ストリーミングレスポンスをOpenAI形式に変換します。"""

        # usageの処理
        usage = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = openai.types.completion_usage.CompletionUsage.model_construct(
                prompt_tokens=getattr(response.usage_metadata, "prompt_token_count", 0),
                completion_tokens=getattr(
                    response.usage_metadata, "candidates_token_count", 0
                ),
                total_tokens=getattr(response.usage_metadata, "total_token_count", 0),
            )

        # レスポンスの処理
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
                                    "arguments": json.dumps(
                                        dict(part.function_call.args)
                                    ),
                                },
                            }
                            tool_calls.append(tool_call)

                message = {
                    "role": "assistant",
                    "content": " ".join(content_parts) if content_parts else None,
                }

                if tool_calls:
                    message["tool_calls"] = tool_calls

                finish_reason = self._get_finish_reason(candidate.finish_reason)

                choices.append(
                    {"index": i, "message": message, "finish_reason": finish_reason}
                )

        return openai.types.chat.ChatCompletion.model_construct(
            id=str(uuid.uuid4()),
            choices=choices,
            created=int(time.time()),
            model=request.model,
            object="chat.completion",
            usage=usage,
        )

    def _process_stream_chunk(
        self,
        request: types_chat.ChatRequest,
        chunk: google.genai.types.GenerateContentResponse,
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
                                    "arguments": json.dumps(
                                        dict(part.function_call.args)
                                    ),
                                },
                            }
                            tool_calls.append(tool_call)

                    if content_parts:
                        delta["content"] = " ".join(content_parts)

                    if tool_calls:
                        delta["tool_calls"] = tool_calls

                finish_reason = (
                    self._get_finish_reason(candidate.finish_reason)
                    if candidate.finish_reason
                    else None
                )

                choices.append(
                    {"index": i, "delta": delta, "finish_reason": finish_reason}
                )

        if not choices:
            # 空のチャンクの場合はスキップ
            return None

        return openai.types.chat.ChatCompletionChunk.model_construct(
            id=str(uuid.uuid4()),
            choices=choices,
            created=int(time.time()),
            model=request.model,
            object="chat.completion.chunk",
        )

    def _get_finish_reason(
        self, finish_reason: typing.Any
    ) -> typing.Literal["stop", "length", "tool_calls", "content_filter"] | None:
        """finish_reasonをOpenAI形式に変換します。"""
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


async def main() -> None:
    """動作確認用コード。"""
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)

    client = VertexAIClient()

    # テストメッセージの作成
    messages = [
        {"role": "system", "content": "あなたは親切なアシスタントです。"},
        {"role": "user", "content": "こんにちは！"},
    ]

    # 非ストリーミングモードでのテスト
    if True:
        response = await client.chat(
            types_chat.ChatRequest(
                messages=messages,
                model="gemini-2.5-flash-preview-05-20",
                temperature=0.7,
                max_tokens=500,
                stream=False,
            )
        )
        print("Response:", response.choices[0].message.content)

    # ストリーミングモードでのテスト
    if True:
        stream = client.chat_stream(
            types_chat.ChatRequest(
                messages=messages,
                model="gemini-2.5-flash-preview-05-20",
                temperature=0.7,
                max_tokens=500,
                stream=True,
            )
        )
        async for chunk in stream:
            print("Chunk:", chunk.choices[0].delta.content)


if __name__ == "__main__":
    asyncio.run(main())
