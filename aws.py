#!/usr/bin/env python3
"""検証用コード。

```
下記のURLを参考にaws.pyを実装してください。

Converse API
https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStream.html

OpenAI API
https://platform.openai.com/docs/api-reference/chat/create
```

"""

import asyncio
import json
import logging
import os
import time
import typing
import uuid

import httpx
import iam_rolesanywhere_session
from openai._streaming import AsyncStream
from openai._types import NOT_GIVEN, Body, Headers, NotGiven, Query
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAudioParam,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionPredictionContentParam,
    ChatCompletionStreamOptionsParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
    completion_create_params,
)
from openai.types.shared.metadata import Metadata
from openai.types.shared.reasoning_effort import ReasoningEffort

import config


class AWSClient:
    """AWSのクライアント。"""

    def __init__(self) -> None:
        """初期化処理。"""
        self.roles_anywhere_session = iam_rolesanywhere_session.IAMRolesAnywhereSession(
            profile_arn=config.AWS_IAMRA_PROFILE_ARN,
            role_arn=config.AWS_IAMRA_ROLE_ARN,
            trust_anchor_arn=config.AWS_IAMRA_TRUST_ANCHOR_ARN,
            certificate=config.AWS_IAMRA_CERTIFICATE_PATH,
            private_key=config.AWS_IAMRA_PRIVATE_PATH,
            region="ap-northeast-1",
            proxies={
                "http": os.environ.get("http_proxy"),
                "https": os.environ.get("https_proxy"),
            },
        ).get_session()
        self.bedrock_runtime = self.roles_anywhere_session.client(
            service_name="bedrock-runtime", region_name="ap-northeast-1"
        )

    async def chat_completion(
        self,
        messages: typing.Iterable[ChatCompletionMessageParam],
        model: str,
        audio: ChatCompletionAudioParam | None | NotGiven = NOT_GIVEN,
        frequency_penalty: float | None | NotGiven = NOT_GIVEN,
        function_call: completion_create_params.FunctionCall | NotGiven = NOT_GIVEN,
        functions: (
            typing.Iterable[completion_create_params.Function] | NotGiven
        ) = NOT_GIVEN,
        logit_bias: dict[str, int] | None | NotGiven = NOT_GIVEN,
        logprobs: bool | None | NotGiven = NOT_GIVEN,
        max_completion_tokens: int | None | NotGiven = NOT_GIVEN,
        max_tokens: int | None | NotGiven = NOT_GIVEN,
        metadata: Metadata | None | NotGiven = NOT_GIVEN,
        modalities: list[typing.Literal["text", "audio"]] | None | NotGiven = NOT_GIVEN,
        n: int | None | NotGiven = NOT_GIVEN,
        parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
        prediction: ChatCompletionPredictionContentParam | None | NotGiven = NOT_GIVEN,
        presence_penalty: float | None | NotGiven = NOT_GIVEN,
        reasoning_effort: ReasoningEffort | None | NotGiven = NOT_GIVEN,
        response_format: completion_create_params.ResponseFormat | NotGiven = NOT_GIVEN,
        seed: int | None | NotGiven = NOT_GIVEN,
        service_tier: (
            typing.Literal["auto", "default", "flex"] | None | NotGiven
        ) = NOT_GIVEN,
        stop: str | None | list[str] | None | NotGiven = NOT_GIVEN,
        store: bool | None | NotGiven = NOT_GIVEN,
        stream: typing.Literal[False] | None | NotGiven = NOT_GIVEN,
        stream_options: ChatCompletionStreamOptionsParam | None | NotGiven = NOT_GIVEN,
        temperature: float | None | NotGiven = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
        tools: typing.Iterable[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        top_logprobs: int | None | NotGiven = NOT_GIVEN,
        top_p: float | None | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        web_search_options: (
            completion_create_params.WebSearchOptions | NotGiven
        ) = NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> ChatCompletion | AsyncStream[ChatCompletionChunk]:
        """OpenAIのChat Completions APIを使用してチャット応答を生成します。

        Returns:
            ChatCompletion | AsyncStream[ChatCompletionChunk]: ストリーミングが無効の場合はChatCompletion、
            有効の場合はAsyncStream[ChatCompletionChunk]を返します。
        """
        # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
        # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStream.html
        # https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference-call.html
        # https://docs.aws.amazon.com/bedrock/latest/userguide/tool-use-inference-call.html
        # https://docs.aws.amazon.com/bedrock/latest/userguide/tool-use-examples.html

        # メッセージをBedrockの形式に変換
        formatted_messages = []
        for message in messages:
            formatted_message = {"role": message["role"]}
            if isinstance(message["content"], list):
                formatted_message["content"] = message["content"]
            elif isinstance(message["content"], str):
                formatted_message["content"] = [{"text": message["content"]}]
            else:
                raise ValueError(
                    f"Invalid content type: {type(message['content'])}. "
                    "Expected str or list."
                )
            formatted_messages.append(formatted_message)

        system = [
            {"text": message["content"]}
            for message in messages
            if message["role"] == "system"
        ]

        # 推論設定の追加
        inference_config = {}
        if max_tokens is not NOT_GIVEN:
            inference_config["maxTokens"] = max_tokens
        if stop is not NOT_GIVEN:
            inference_config["stopSequences"] = (
                stop if isinstance(stop, list) else [stop]
            )
        if temperature is not NOT_GIVEN:
            inference_config["temperature"] = temperature
        if top_p is not NOT_GIVEN:
            inference_config["topP"] = top_p

        # Converse APIを呼び出し
        if stream:
            response = self.bedrock_runtime.converse_stream(
                modelId=model,
                messages=formatted_messages,
                system=system,
                inferenceConfig=inference_config,
                additionalmodelRequestFields={},
                guardrailConfig={},
                # toolConfig={"toolChoice": {...}, "tools": [{...}]},
            )
            return await self._process_streaming_response(response, model)
        else:
            response = self.bedrock_runtime.converse(
                modelId=model,
                messages=formatted_messages,
                system=system,
                inferenceConfig=inference_config,
                additionalmodelRequestFields={},
                guardrailConfig={},
                # toolConfig={"toolChoice": {...}, "tools": [{...}]},
            )
            return await self._process_non_streaming_response(response, model)

    async def _process_non_streaming_response(
        self, response: dict, model: str
    ) -> ChatCompletion:
        """非ストリーミングレスポンスをOpenAI形式に変換します。

        Args:
            response: Bedrockからのレスポンス
            model: モデル名

        Returns:
            ChatCompletion: OpenAI形式のレスポンス
        """
        response_body = json.loads(response["body"].read())
        return ChatCompletion(
            id=str(uuid.uuid4()),
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": response_body["output"]["message"]["content"][0][
                            "text"
                        ],
                    },
                    "finish_reason": response_body.get("stopReason", "stop"),
                    "index": 0,
                }
            ],
            created=int(time.time()),
            model=model,
            object="chat.completion",
            system_fingerprint=None,
            usage=response_body.get(
                "usage", {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}
            ),
        )

    async def _process_streaming_response(
        self, response: dict, model: str
    ) -> AsyncStream[ChatCompletionChunk]:
        """ストリーミングレスポンスを処理します。

        Args:
            response: Bedrockからのレスポンス
            model: モデル名

        Returns:
            AsyncStream[ChatCompletionChunk]: ストリーミングレスポンス
        """

        async def process_stream():
            """ストリームイベントを処理します。"""
            async for event in response["body"]:
                event_data = json.loads(event["chunk"]["bytes"].decode())
                if chunk := self._process_stream_event(event_data, model):
                    yield chunk

        return AsyncStream[ChatCompletionChunk](
            cast_to=ChatCompletionChunk,
            response=response,
            client=self,
            stream_cls=lambda x: None,  # 不要だが、APIの要件で必要
            stream=process_stream(),
        )

    def _process_stream_event(
        self, event_data: dict, model: str
    ) -> ChatCompletionChunk | None:
        """ストリームイベントをOpenAI形式のチャンクに変換します。

        Args:
            event_data: イベントデータ
            model: モデル名

        Returns:
            ChatCompletionChunk | None: OpenAI形式のチャンク。イベントが処理不要な場合はNone。
        """
        if "contentBlockDelta" in event_data:
            delta_text = event_data["contentBlockDelta"]["delta"].get("text", "")
            return ChatCompletionChunk(
                id=str(uuid.uuid4()),
                choices=[
                    {
                        "delta": {"content": delta_text},
                        "finish_reason": None,
                        "index": 0,
                    }
                ],
                created=int(time.time()),
                model=model,
                object="chat.completion.chunk",
            )
        elif "messageStop" in event_data:
            return ChatCompletionChunk(
                id=str(uuid.uuid4()),
                choices=[
                    {
                        "delta": {},
                        "finish_reason": event_data["messageStop"].get(
                            "stopReason", "stop"
                        ),
                        "index": 0,
                    }
                ],
                created=int(time.time()),
                model=model,
                object="chat.completion.chunk",
            )
        return None


async def main() -> None:
    """動作確認用コード。"""
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)

    client = AWSClient()

    # テストメッセージの作成
    messages = [
        {"role": "system", "content": "あなたは親切なアシスタントです。"},
        {"role": "user", "content": "こんにちは！"},
    ]

    # 非ストリーミングモードでのテスト
    response = await client.chat_completion(
        messages=messages,
        model="anthropic.claude-3-haiku-20240307-v1:0",
        temperature=0.7,
        max_tokens=500,
    )
    print("Response:", response.choices[0].message.content)

    # ストリーミングモードでのテスト
    stream = await client.chat_completion(
        messages=messages,
        model="anthropic.claude-3-haiku-20240307-v1:0",
        temperature=0.7,
        max_tokens=500,
        stream=True,
    )
    async for chunk in stream:
        print("Chunk:", chunk.choices[0].delta.content, end="")


if __name__ == "__main__":
    asyncio.run(main())
