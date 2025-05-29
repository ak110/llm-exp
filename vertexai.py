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

    # テストメッセージの作成
    messages: list[openai.types.chat.ChatCompletionMessageParam] = [
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
            if len(chunk.choices) > 0:
                print("Chunk:", chunk.choices[0].delta.content)
            if chunk.usage is not None:
                print("Usage:", chunk.usage)


if __name__ == "__main__":
    asyncio.run(main())
