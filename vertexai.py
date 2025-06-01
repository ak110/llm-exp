#!/usr/bin/env python3
"""VertexAIのChat Completions API互換の実装。"""

import asyncio
import collections.abc
import logging

import google.genai
import openai.types.chat
import openai.types.completion_usage

import config
import types_chat
import vertexai_request
import vertexai_response

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
        formatted_messages, system_instruction = vertexai_request.format_messages(
            request.messages
        )
        # 生成設定の作成
        generation_config = vertexai_request.make_generation_config(request)
        if system_instruction is not None:
            generation_config.system_instruction = system_instruction

        # Vertex AIでチャット生成を実行
        response = await client.aio.models.generate_content(
            model=request.model, contents=formatted_messages, config=generation_config
        )

        return vertexai_response.process_non_streaming_response(request, response)

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
        formatted_messages, system_instruction = vertexai_request.format_messages(
            request.messages
        )

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
        generation_config = vertexai_request.make_generation_config(request)

        # Vertex AIでストリーミングチャット生成を実行
        stream = await client.aio.models.generate_content_stream(
            model=request.model, contents=formatted_messages, config=generation_config
        )

        async for response_chunk in stream:
            if chunk := vertexai_response.process_stream_chunk(request, response_chunk):
                yield chunk


async def main() -> None:
    """動作確認用コード。"""
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)

    client = VertexAIClient()
    model = "gemini-2.5-flash-preview-05-20"

    # 非ストリーミングモードでのテスト
    if False:
        response = await client.chat(
            types_chat.ChatRequest(
                messages=[
                    {"role": "system", "content": "あなたは親切なアシスタントです。"},
                    {"role": "user", "content": "こんにちは！"},
                ],
                model=model,
                temperature=0.7,
                max_completion_tokens=500,
                stream=False,
            )
        )
        print("Response:", response.choices[0].message.content)

    # ストリーミングモードでのTool Callingテスト
    if True:
        stream = client.chat_stream(
            types_chat.ChatRequest(
                messages=[
                    {"role": "system", "content": "あなたは親切なアシスタントです。"},
                    {"role": "user", "content": "東京の天気を教えてください"},
                ],
                model=model,
                temperature=0.7,
                max_completion_tokens=500,
                stream=True,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "指定された場所の現在の天気を取得する",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "location": {
                                        "type": "string",
                                        "description": "天気を知りたい場所（例：東京、大阪）",
                                    }
                                },
                                "required": ["location"],
                            },
                        },
                    }
                ],
            )
        )
        async for chunk in stream:
            if len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta.content is not None:
                    print("delta.content:", delta.content)
                if delta.tool_calls is not None:
                    print(
                        "delta.tool_calls:",
                        [
                            tool_call.model_dump(exclude_none=True)
                            for tool_call in delta.tool_calls
                        ],
                    )
            if chunk.usage is not None:
                print("usage:", chunk.usage)

    # ストリーミングモードでのTool Callingテスト2
    if True:
        stream = client.chat_stream(
            types_chat.ChatRequest(
                messages=[
                    {"role": "system", "content": "あなたは親切なアシスタントです。"},
                    {"role": "user", "content": "東京の天気を教えてください"},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_qaxHS0I7vEg7aBTahgrq2754",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "東京"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_qaxHS0I7vEg7aBTahgrq2754",
                        "content": "晴れ",
                    },
                ],
                model=model,
                temperature=0.7,
                max_completion_tokens=500,
                stream=True,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "指定された場所の現在の天気を取得する",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "location": {
                                        "type": "string",
                                        "description": "天気を知りたい場所（例：東京、大阪）",
                                    }
                                },
                                "required": ["location"],
                            },
                        },
                    }
                ],
            )
        )
        async for chunk in stream:
            if len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta.content is not None:
                    print("delta.content:", delta.content)
                if delta.tool_calls is not None:
                    print(
                        "delta.tool_calls:",
                        [
                            tool_call.model_dump(exclude_none=True)
                            for tool_call in delta.tool_calls
                        ],
                    )
            if chunk.usage is not None:
                print("usage:", chunk.usage)


if __name__ == "__main__":
    asyncio.run(main())
