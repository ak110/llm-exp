#!/usr/bin/env python3
"""OpenAI API。"""

import argparse
import asyncio
import binascii
import collections.abc
import logging

import cryptography
import cryptography.hazmat.backends
import cryptography.hazmat.primitives.hashes
import cryptography.x509
import msal
import openai
import openai.types.chat
from openai._types import NOT_GIVEN

import config
import errors
import types_chat
import types_embedding
import types_image

logger = logging.getLogger(__name__)


class AzureClient:
    """Azure OpenAI Serviceのクライアント。"""

    def __init__(self) -> None:
        pass

    async def chat(
        self, request: types_chat.ChatRequest
    ) -> openai.types.chat.ChatCompletion:
        """OpenAIのChat Completions API互換API。"""
        assert not request.stream

        try:
            async with openai.AsyncAzureOpenAI(
                azure_endpoint="https://privchat-eu.openai.azure.com",
                api_version="2025-01-01-preview",
                azure_ad_token=acquire_access_token(
                    ["https://cognitiveservices.azure.com/.default"]
                ),
            ) as client:
                return await client.chat.completions.create(
                    model=request.model,
                    messages=request.messages,
                    stream=False,
                    audio=request.audio,
                    frequency_penalty=request.frequency_penalty,
                    logit_bias=request.logit_bias,
                    logprobs=request.logprobs,
                    max_completion_tokens=request.max_completion_tokens,
                    metadata=request.metadata,
                    modalities=request.modalities,
                    n=request.n,
                    parallel_tool_calls=request.parallel_tool_calls,
                    prediction=request.prediction,
                    presence_penalty=request.presence_penalty,
                    reasoning_effort=request.reasoning_effort,
                    response_format=request.response_format,
                    seed=request.seed,
                    service_tier=request.service_tier,
                    stop=request.stop,
                    store=request.store,
                    # stream_options=request.stream_options,
                    temperature=request.temperature,
                    tool_choice=request.tool_choice,
                    tools=request.tools,
                    top_logprobs=request.top_logprobs,
                    top_p=request.top_p,
                    user=request.user,
                    web_search_options=request.web_search_options,
                )
        except Exception as e:
            raise errors.map_exception(e) from e

    async def chat_stream(
        self, request: types_chat.ChatRequest
    ) -> collections.abc.AsyncGenerator[openai.types.chat.ChatCompletionChunk, None]:
        """OpenAIのChat Completions API互換API。(ストリーミング版)"""
        assert request.stream
        if request.stream_options is NOT_GIVEN:
            request.stream_options = {"include_usage": True}

        try:
            async with openai.AsyncAzureOpenAI(
                azure_endpoint="https://privchat-eu.openai.azure.com",
                api_version="2025-01-01-preview",
                azure_ad_token=acquire_access_token(
                    ["https://cognitiveservices.azure.com/.default"]
                ),
            ) as client:
                stream = await client.chat.completions.create(
                    model=request.model,
                    messages=request.messages,
                    stream=True,
                    audio=request.audio,
                    frequency_penalty=request.frequency_penalty,
                    logit_bias=request.logit_bias,
                    logprobs=request.logprobs,
                    max_completion_tokens=request.max_completion_tokens,
                    metadata=request.metadata,
                    modalities=request.modalities,
                    n=request.n,
                    parallel_tool_calls=request.parallel_tool_calls,
                    prediction=request.prediction,
                    presence_penalty=request.presence_penalty,
                    reasoning_effort=request.reasoning_effort,
                    response_format=request.response_format,
                    seed=request.seed,
                    service_tier=request.service_tier,
                    stop=request.stop,
                    store=request.store,
                    stream_options=request.stream_options,
                    temperature=request.temperature,
                    tool_choice=request.tool_choice,
                    tools=request.tools,
                    top_logprobs=request.top_logprobs,
                    top_p=request.top_p,
                    user=request.user,
                    web_search_options=request.web_search_options,
                )
                async for chunk in stream:
                    yield chunk
        except Exception as e:
            raise errors.map_exception(e) from e

    async def images(
        self, request: types_image.ImageRequest
    ) -> openai.types.ImagesResponse:
        """OpenAIのImage Generation API互換API。"""
        try:
            async with openai.AsyncAzureOpenAI(
                azure_endpoint="https://privchat-eu.openai.azure.com",
                api_version="2025-01-01-preview",
                azure_ad_token=acquire_access_token(
                    ["https://cognitiveservices.azure.com/.default"]
                ),
            ) as client:
                return await client.images.generate(
                    prompt=request.prompt,
                    background=request.background,
                    model=request.model,
                    moderation=request.moderation,
                    n=request.n,
                    output_compression=request.output_compression,
                    output_format=request.output_format,
                    quality=request.quality,
                    response_format=request.response_format,
                    size=request.size,
                    style=request.style,
                    user=request.user,
                )
        except Exception as e:
            raise errors.map_exception(e) from e

    async def embeddings(
        self, request: types_embedding.EmbeddingRequest
    ) -> openai.types.CreateEmbeddingResponse:
        """OpenAIのEmbedding API互換API。"""
        try:
            async with openai.AsyncAzureOpenAI(
                azure_endpoint="https://privchat-eu.openai.azure.com",
                api_version="2025-01-01-preview",
                azure_ad_token=acquire_access_token(
                    ["https://cognitiveservices.azure.com/.default"]
                ),
            ) as client:
                return await client.embeddings.create(
                    input=request.input,
                    model=request.model,
                    dimensions=request.dimensions,
                    encoding_format=request.encoding_format,
                    user=request.user,
                )
        except Exception as e:
            raise errors.map_exception(e) from e


def acquire_access_token(scopes: list[str]) -> str:
    """Azure ADのトークンを取得する。"""
    app = create_msal_app()
    token = app.acquire_token_for_client(scopes)
    if "error" in token:
        raise errors.APIError(
            f"認証処理でエラーが発生しました。エラーコード:{token.get('error')}",
            code="authentication_error",
            status_code=401,
            type_="server_error",
            details=token.get("error_description"),
        )
    assert "access_token" in token, f"トークンが取得できませんでした: {token}"
    return token["access_token"]


def create_msal_app() -> msal.ConfidentialClientApplication:
    """msal.ConfidentialClientApplicationを作成する。"""
    client_credential = _load_pem_certificate(
        config.AZURE_CLIENT_CERTIFICATE_PATH.read_bytes()
    )
    return msal.ConfidentialClientApplication(
        config.AZURE_CLIENT_ID,
        authority=f"https://login.microsoftonline.com/{config.AZURE_TENANT_ID}",
        client_credential=client_credential,
    )


def _load_pem_certificate(certificate_data: bytes) -> dict:
    cert = cryptography.x509.load_pem_x509_certificate(
        certificate_data, cryptography.hazmat.backends.default_backend()
    )
    fingerprint = cert.fingerprint(cryptography.hazmat.primitives.hashes.SHA1())
    return {
        "private_key": certificate_data,
        "thumbprint": binascii.hexlify(fingerprint).decode("utf-8"),
    }


async def main() -> None:
    """動作確認用コード。"""
    parser = argparse.ArgumentParser(description="Azure OpenAI API")
    parser.add_argument(
        "mode", choices=["chat", "chat-stream", "image", "embedding"], help="実行モード"
    )
    args = parser.parse_args()
    mode = args.mode

    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)

    client = AzureClient()
    chat_model = "gpt-4o-mini-2024-07-18"

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
        image_response = await client.images(
            types_image.ImageRequest(
                model="dall-e-3",
                prompt="赤いバラの花束",
                n=1,
                quality="standard",
                size="1024x1024",
                style="natural",
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
                model="text-embedding-ada-002",
                input=["こんにちは、世界！", "hello, world!"],
                encoding_format="float",
            )
        )
        for i, embedding in enumerate(embedding_response.data):
            print(f"Embedding {i}: {embedding.embedding[:5]}...")


if __name__ == "__main__":
    asyncio.run(main())
