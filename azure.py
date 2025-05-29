#!/usr/bin/env python3
"""OpenAI API。"""

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
from openai._types import NOT_GIVEN

import config
import types_chat


class AzureClient:
    """Azure OpenAI Serviceのクライアント。"""

    def __init__(self) -> None:
        pass

    async def chat(
        self, request: types_chat.ChatRequest
    ) -> openai.types.chat.ChatCompletion:
        """OpenAIのChat Completions API互換API。"""
        assert not request.stream

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

    async def chat_stream(
        self, request: types_chat.ChatRequest
    ) -> collections.abc.AsyncGenerator[openai.types.chat.ChatCompletionChunk, None]:
        """OpenAIのChat Completions API互換API。(ストリーミング版)"""
        assert request.stream
        if request.stream_options is NOT_GIVEN:
            request.stream_options = {"include_usage": True}

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


def acquire_access_token(scopes: list[str]) -> str:
    """Azure ADのトークンを取得する。"""
    app = create_msal_app()
    token = app.acquire_token_for_client(scopes)
    if "error" in token:
        raise ValueError(
            f"認証処理でエラーが発生しました。"
            f"エラーコード:{token.get('error')}"
            f" 詳細:{token.get('error_description')}"
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
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)

    client = AzureClient()

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
                model="gpt-4o-mini-2024-07-18",
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
                model="gpt-4o-mini-2024-07-18",
                temperature=0.7,
                max_tokens=500,
                stream=True,
            )
        )
        async for chunk in stream:
            if len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                print("Chunk:", chunk.choices[0].delta.content)
            if chunk.usage is not None:
                print("Usage:", chunk.usage)


if __name__ == "__main__":
    asyncio.run(main())
