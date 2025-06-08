#!/usr/bin/env python3
"""AWSのOpenAI API互換の実装。"""

import argparse
import asyncio
import collections.abc
import json
import logging
import os
import typing

import aiobotocore.session
import iam_rolesanywhere_session
import openai.types.chat
import types_aiobotocore_bedrock_runtime.client

import aws_chat_request
import aws_chat_response
import aws_embedding
import aws_image
import config
import types_chat
import types_embedding
import types_image

logger = logging.getLogger(__name__)


class AWSClient:
    """AWSのクライアント。"""

    def __init__(self) -> None:
        """初期化処理。"""
        self.session = (
            iam_rolesanywhere_session.IAMRolesAnywhereSession(
                profile_arn=config.AWS_IAMRA_PROFILE_ARN,
                role_arn=config.AWS_IAMRA_ROLE_ARN,
                trust_anchor_arn=config.AWS_IAMRA_TRUST_ANCHOR_ARN,
                certificate=config.AWS_IAMRA_CERTIFICATE_PATH,
                private_key=config.AWS_IAMRA_PRIVATE_PATH,
                region=config.AWS_IAMRA_REGION,
                proxies={
                    "http": os.environ.get("http_proxy"),
                    "https": os.environ.get("https_proxy"),
                },
            ).get_session()
            if config.AWS_IAMRA_CERTIFICATE_PATH.exists()
            else None
        )

    async def chat(
        self, request: types_chat.ChatRequest
    ) -> openai.types.chat.ChatCompletion:
        """OpenAIのChat Completions API互換API。"""
        assert self.session is not None
        assert not request.stream
        # リクエストの変換
        kwargs = aws_chat_request.convert_request(request)
        # API呼び出し
        async with self._create_client() as bedrock:
            bedrock = typing.cast(
                types_aiobotocore_bedrock_runtime.client.BedrockRuntimeClient, bedrock
            )
            response = await bedrock.converse(**kwargs)
            return aws_chat_response.process_non_streaming_response(request, response)

    async def chat_stream(
        self, request: types_chat.ChatRequest
    ) -> collections.abc.AsyncGenerator[openai.types.chat.ChatCompletionChunk, None]:
        """OpenAIのChat Completions API互換API。(ストリーミング版)"""
        assert self.session is not None
        assert request.stream
        # リクエストの変換
        kwargs = aws_chat_request.convert_request(request)
        # API呼び出し
        async with self._create_client() as bedrock:
            bedrock = typing.cast(
                types_aiobotocore_bedrock_runtime.client.BedrockRuntimeClient, bedrock
            )
            response = await bedrock.converse_stream(**kwargs)
            async for event in response["stream"]:
                chunk = aws_chat_response.process_stream_event(request, event)
                if chunk is not None:
                    yield chunk

    async def images_generate(
        self, request: types_image.ImageRequest
    ) -> openai.types.ImagesResponse:
        """OpenAIのImage Creation API互換API。"""
        assert self.session is not None
        # リクエストの変換
        request_body = aws_image.convert_request(request)
        # API呼び出し
        async with self._create_client() as bedrock:
            bedrock = typing.cast(
                types_aiobotocore_bedrock_runtime.client.BedrockRuntimeClient, bedrock
            )
            response_body = await self._invoke_model(
                bedrock, request.model, request_body
            )
            return aws_image.convert_response(request, response_body)

    async def embeddings(
        self, request: types_embedding.EmbeddingRequest
    ) -> openai.types.CreateEmbeddingResponse:
        """OpenAIのImage Creation API互換API。"""
        assert self.session is not None
        # リクエストの変換
        request_body = aws_embedding.convert_request(request)
        # API呼び出し
        async with self._create_client() as bedrock:
            bedrock = typing.cast(
                types_aiobotocore_bedrock_runtime.client.BedrockRuntimeClient, bedrock
            )
            response_body = await self._invoke_model(
                bedrock, request.model, request_body
            )
            return aws_embedding.convert_response(request, response_body)

    async def _invoke_model(
        self,
        bedrock: types_aiobotocore_bedrock_runtime.client.BedrockRuntimeClient,
        model_id: str,
        body: dict[str, typing.Any],
    ) -> dict[str, typing.Any]:
        """bedrockのinvoke_modelを呼び出す。"""
        response = await bedrock.invoke_model(
            body=json.dumps(body),
            modelId=model_id,
            accept="application/json",
            contentType="application/json",
        )
        response_body = json.loads(await response["body"].read())
        return response_body

    def _create_client(self) -> aiobotocore.session.ClientCreatorContext:
        """aiobotocoreのクライアントを作成する。"""
        assert self.session is not None
        credentials = self.session.get_credentials()
        session = aiobotocore.session.get_session()
        return session.create_client(
            service_name="bedrock-runtime",
            region_name="ap-northeast-1",
            aws_access_key_id=credentials.access_key,
            aws_secret_access_key=credentials.secret_key,
            aws_session_token=credentials.token,
            aws_account_id=credentials.account_id,
        )


async def main() -> None:
    """動作確認用コード。"""
    parser = argparse.ArgumentParser(description="AWSのOpenAI API互換の実装")
    parser.add_argument(
        "mode", choices=["chat", "chat-stream", "image", "embedding"], help="実行モード"
    )
    args = parser.parse_args()
    mode = args.mode

    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)

    client = AWSClient()
    chat_model = "anthropic.claude-3-haiku-20240307-v1:0"

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
                model="amazon.nova-canvas-v1:0",
                prompt="A cute cat sitting on a table",
                n=1,
                size="1024x1024",
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
                model="cohere.embed-multilingual-v3",
                input=["こんにちは、世界！", "hello, world!"],
            )
        )
        for i, embedding in enumerate(embedding_response.data):
            print(f"Embedding {i}: {embedding.embedding[:5]}...")


if __name__ == "__main__":
    asyncio.run(main())
