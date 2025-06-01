#!/usr/bin/env python3
"""AWSのChat Completions API互換の実装。"""

import asyncio
import collections.abc
import logging
import os

import aiobotocore.session
import iam_rolesanywhere_session
import openai.types.chat

import aws_request
import aws_response
import config
import types_chat

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
        kwargs = aws_request.convert_request(request)

        # Converse APIを呼び出し
        credentials = self.session.get_credentials()
        session = aiobotocore.session.get_session()
        async with session.create_client(
            service_name="bedrock-runtime",
            region_name="ap-northeast-1",
            aws_access_key_id=credentials.access_key,
            aws_secret_access_key=credentials.secret_key,
            aws_session_token=credentials.token,
            aws_account_id=credentials.account_id,
        ) as bedrock:
            response = await bedrock.converse(**kwargs)
            return aws_response.process_non_streaming_response(request, response)

    async def chat_stream(
        self, request: types_chat.ChatRequest
    ) -> collections.abc.AsyncGenerator[openai.types.chat.ChatCompletionChunk, None]:
        """OpenAIのChat Completions API互換API。(ストリーミング版)"""
        assert self.session is not None
        assert request.stream
        kwargs = aws_request.convert_request(request)

        # Converse APIを呼び出し
        credentials = self.session.get_credentials()
        session = aiobotocore.session.get_session()
        async with session.create_client(
            service_name="bedrock-runtime",
            region_name="ap-northeast-1",
            aws_access_key_id=credentials.access_key,
            aws_secret_access_key=credentials.secret_key,
            aws_session_token=credentials.token,
            aws_account_id=credentials.account_id,
        ) as bedrock:
            response = await bedrock.converse_stream(**kwargs)
            async for event in response["stream"]:
                chunk = aws_response.process_stream_event(request, event)
                if chunk is not None:
                    yield chunk


async def main() -> None:
    """動作確認用コード。"""
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)

    client = AWSClient()
    model = "anthropic.claude-3-haiku-20240307-v1:0"

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
                print("usage:", chunk.usage.model_dump(exclude_none=True))

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
                                "id": "tooluse_EpJP3Wr9SIOCe0_Yz2k_OA",
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
                        "tool_call_id": "tooluse_EpJP3Wr9SIOCe0_Yz2k_OA",
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
                print("usage:", chunk.usage.model_dump(exclude_none=True))


if __name__ == "__main__":
    asyncio.run(main())
