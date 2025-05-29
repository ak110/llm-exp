#!/usr/bin/env python3
"""AWSのChat Completions API互換の実装。"""

import asyncio
import collections.abc
import logging
import os

import aiobotocore.session
import iam_rolesanywhere_session
import openai.types.chat
import openai.types.completion_usage

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
        messages, system = aws_request.format_messages(request.messages)
        inference_config = aws_request.make_inference_config(request)
        tool_config = aws_request.make_tool_config(request.tools, request.tool_choice)

        # Converse APIを呼び出し
        credentials = self.session.get_credentials()
        print(f"{credentials=}")  # TODO: 仮
        print(f"{credentials.account_id=}")
        session = aiobotocore.session.get_session()
        async with session.create_client(
            service_name="bedrock-runtime",
            region_name="ap-northeast-1",
            aws_access_key_id=credentials.access_key,
            aws_secret_access_key=credentials.secret_key,
            aws_session_token=credentials.token,
            aws_account_id=credentials.account_id,
        ) as bedrock:
            response = await bedrock.converse(
                modelId=request.model,
                messages=messages,
                system=system,
                inferenceConfig=inference_config,
                additionalModelRequestFields={},
                # guardrailConfig={"guardrailVersion": "", "guardrailIdentifier": ""},
                toolConfig=tool_config,
            )
            return await aws_response.process_non_streaming_response(request, response)

    async def chat_stream(
        self, request: types_chat.ChatRequest
    ) -> collections.abc.AsyncGenerator[openai.types.chat.ChatCompletionChunk, None]:
        """OpenAIのChat Completions API互換API。(ストリーミング版)"""
        assert self.session is not None
        assert request.stream
        messages, system = aws_request.format_messages(request.messages)
        inference_config = aws_request.make_inference_config(request)
        tool_config = aws_request.make_tool_config(request.tools, request.tool_choice)

        # Converse APIを呼び出し
        credentials = self.session.get_credentials()
        print(f"{credentials=}")  # TODO: 仮
        print(f"{credentials.account_id=}")
        session = aiobotocore.session.get_session()
        async with session.create_client(
            service_name="bedrock-runtime",
            region_name="ap-northeast-1",
            aws_access_key_id=credentials.access_key,
            aws_secret_access_key=credentials.secret_key,
            aws_session_token=credentials.token,
            aws_account_id=credentials.account_id,
        ) as bedrock:
            response = await bedrock.converse_stream(
                modelId=request.model,
                messages=messages,
                system=system,
                inferenceConfig=inference_config,
                additionalModelRequestFields={},
                # guardrailConfig={"guardrailVersion": "", "guardrailIdentifier": ""},
                toolConfig=tool_config,
            )
            async for event in response["stream"]:
                if chunk := aws_response.process_stream_event(request, event):
                    yield chunk


async def main() -> None:
    """動作確認用コード。"""
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)

    client = AWSClient()

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
                model="anthropic.claude-3-haiku-20240307-v1:0",
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
                model="anthropic.claude-3-haiku-20240307-v1:0",
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
