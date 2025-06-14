#!/usr/bin/env python3
"""VertexAIのOpenAI API互換の実装。"""

import argparse
import asyncio
import collections.abc
import logging

import google.genai
import openai.types.chat

import config
import errors
import types_chat
import types_embedding
import types_image
import vertexai_chat_request
import vertexai_chat_response
import vertexai_embedding
import vertexai_image

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

        try:
            generation_config, formatted_messages = (
                vertexai_chat_request.convert_request(request)
            )
            response = await client.aio.models.generate_content(
                model=request.model,
                contents=formatted_messages,
                config=generation_config,
            )
            return vertexai_chat_response.process_non_streaming_response(
                request, response
            )
        except Exception as e:
            raise errors.map_exception(e) from e

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

        try:
            generation_config, formatted_messages = (
                vertexai_chat_request.convert_request(request)
            )
            stream = await client.aio.models.generate_content_stream(
                model=request.model,
                contents=formatted_messages,
                config=generation_config,
            )
            async for response_chunk in stream:
                chunk = vertexai_chat_response.process_stream_chunk(
                    request, response_chunk
                )
                if chunk is not None:
                    yield chunk
        except Exception as e:
            raise errors.map_exception(e) from e

    async def images_generate(
        self, request: types_image.ImageRequest
    ) -> openai.types.ImagesResponse:
        """OpenAIのImage Creation API互換API。"""
        client = google.genai.Client(
            vertexai=True,
            project=config.GOOGLE_PROJECT_ID,
            location="us-central1",
            http_options=google.genai.types.HttpOptions(api_version="v1"),
        )

        try:
            generation_config = vertexai_image.convert_request(request)
            response = await client.aio.models.generate_images(
                model=request.model, prompt=request.prompt, config=generation_config
            )
            return vertexai_image.convert_response(request, response)
        except Exception as e:
            raise errors.map_exception(e) from e

    async def embeddings(
        self, request: types_embedding.EmbeddingRequest
    ) -> openai.types.CreateEmbeddingResponse:
        """OpenAIのEmbeddings API互換API。"""
        client = google.genai.Client(
            vertexai=True,
            project=config.GOOGLE_PROJECT_ID,
            location="us-central1",
            http_options=google.genai.types.HttpOptions(api_version="v1"),
        )

        try:
            embed_config, contents = vertexai_embedding.convert_request(request)
            response = await client.aio.models.embed_content(
                model=request.model, contents=contents, config=embed_config
            )
            return vertexai_embedding.convert_response(request, response)
        except Exception as e:
            raise errors.map_exception(e) from e


async def main() -> None:
    """動作確認用コード。"""
    parser = argparse.ArgumentParser(description="VertexAIのOpenAI API互換の実装")
    parser.add_argument(
        "mode", choices=["chat", "chat-stream", "image", "embedding"], help="実行モード"
    )
    args = parser.parse_args()
    mode = args.mode

    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)

    client = VertexAIClient()
    chat_model = "gemini-2.5-flash-preview-05-20"

    if mode == "chat":
        response = await client.chat(
            types_chat.ChatRequest(
                messages=[
                    {"role": "system", "content": "あなたは親切なアシスタントです。"},
                    {"role": "user", "content": "こんにちは！"},
                ],
                model=chat_model,
                temperature=0.7,
                max_completion_tokens=500,
                stream=False,
            )
        )
        print("Response:", response.choices[0].message.content)

    elif mode == "chat-stream":
        stream = client.chat_stream(
            types_chat.ChatRequest(
                messages=[
                    {"role": "system", "content": "あなたは親切なアシスタントです。"},
                    {"role": "user", "content": "東京の天気を教えてください"},
                ],
                model=chat_model,
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
                print("usage:", chunk.usage.model_dump(exclude_none=True))

    elif mode == "image":
        image_response = await client.images_generate(
            types_image.ImageRequest(
                # model="imagen-3.0-generate-002",
                model="imagen-3.0-fast-generate-001",
                # prompt="A cute cat sitting on a table",
                prompt="どら焼き",
                n=1,
            )
        )
        assert image_response.data is not None
        for i, image in enumerate(image_response.data):
            print(f"Image {i}:")
            if image.url is not None:
                print("  URL:", image.url)
            if image.b64_json is not None:
                print("  Base64 JSON:", image.b64_json[:50] + "...")

    elif mode == "embedding":
        embedding_response = await client.embeddings(
            types_embedding.EmbeddingRequest(
                model="text-multilingual-embedding-002",
                input=["こんにちは、世界！", "hello, world!"],
            )
        )
        for i, embedding in enumerate(embedding_response.data):
            print(f"Embedding {i}: {embedding.embedding[:5]}...")


if __name__ == "__main__":
    asyncio.run(main())
    asyncio.run(main())
