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
import collections.abc
import json
import logging
import os
import time
import typing
import uuid

import aiobotocore.session
import iam_rolesanywhere_session
import openai.types.chat
import openai.types.completion_usage
import types_aiobotocore_bedrock_runtime.type_defs
from openai._types import NOT_GIVEN

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
        """OpenAIのChat Completions APIを使用してチャット応答を生成します。

        Returns:
            ChatCompletion | AsyncStream[ChatCompletionChunk]: ストリーミングが無効の場合はChatCompletion、
            有効の場合はAsyncStream[ChatCompletionChunk]を返します。
        """
        assert self.session is not None
        assert not request.stream
        messages, system = self._format_messages(request.messages)
        inference_config = self._make_inference_config(request)

        # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
        # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStream.html
        # https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference-call.html
        # https://docs.aws.amazon.com/bedrock/latest/userguide/tool-use-inference-call.html
        # https://docs.aws.amazon.com/bedrock/latest/userguide/tool-use-examples.html

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
                # toolConfig={"toolChoice": {...}, "tools": [{...}]},
            )
            return await self._process_non_streaming_response(request, response)

    async def chat_stream(
        self, request: types_chat.ChatRequest
    ) -> collections.abc.AsyncGenerator[openai.types.chat.ChatCompletionChunk, None]:
        """OpenAIのChat Completions APIを使用してチャット応答を生成します。

        Returns:
            ChatCompletion | AsyncStream[ChatCompletionChunk]: ストリーミングが無効の場合はChatCompletion、
            有効の場合はAsyncStream[ChatCompletionChunk]を返します。
        """
        assert self.session is not None
        assert request.stream
        messages, system = self._format_messages(request.messages)
        inference_config = self._make_inference_config(request)

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
                # toolConfig={"toolChoice": {...}, "tools": [{...}]},
            )
            async for event in response["stream"]:
                if chunk := self._process_stream_event(request, event):
                    yield chunk

    def _format_messages(
        self, messages: typing.Iterable[openai.types.chat.ChatCompletionMessageParam]
    ) -> tuple[
        list[types_aiobotocore_bedrock_runtime.type_defs.MessageTypeDef],
        list[types_aiobotocore_bedrock_runtime.type_defs.SystemContentBlockTypeDef],
    ]:
        """メッセージをBedrockの形式に変換します。"""
        formatted_messages: list[
            types_aiobotocore_bedrock_runtime.type_defs.MessageTypeDef
        ] = []
        for message in messages:
            role: typing.Literal["assistant", "user"]
            content: list[
                types_aiobotocore_bedrock_runtime.type_defs.ContentBlockTypeDef
            ]
            # TODO: role
            # 'developer', 'system', 'user', 'assistant', 'tool', 'function'
            if message["role"] == "tool":
                role = "user"

                content_data = message["content"]
                try:
                    content_data = {"json": json.loads(content_data)}
                except json.JSONDecodeError:
                    content_data = {"text": content_data}

                content = [
                    {
                        "toolResult": {
                            "toolUseId": message["tool_call_id"],
                            "content": [content_data],
                            # "status": "error"  ⇒ OpenAI APIにはこれに相当するものが無い…
                        }
                    }
                ]

            elif message["role"] in ("user", "assistant"):
                role: typing.Literal["assistant", "user"] = message["role"]
                if isinstance(message["content"], list):
                    # TODO: image_urlなど
                    content = message["content"]
                elif isinstance(message["content"], str):
                    content = [{"text": message["content"]}]
                else:
                    raise ValueError(
                        f"Invalid content type: {type(message['content'])}. "
                        "Expected str or list."
                    )
            else:
                continue  # 'developer', 'system', 'function' は無視

            formatted_messages.append({"role": role, "content": content})

        system = [
            {"text": message["content"]}
            for message in messages
            if message["role"] in ("developer", "system")
        ]

        return formatted_messages, system

    def _make_inference_config(
        self, request: types_chat.ChatRequest
    ) -> types_aiobotocore_bedrock_runtime.type_defs.InferenceConfigurationTypeDef:
        """推論設定を作成。"""
        inference_config = {}
        if request.max_tokens is not NOT_GIVEN:
            inference_config["maxTokens"] = request.max_tokens
        if request.stop is not NOT_GIVEN:
            inference_config["stopSequences"] = (
                request.stop if isinstance(request.stop, list) else [request.stop]
            )
        if request.temperature is not NOT_GIVEN:
            inference_config["temperature"] = request.temperature
        if request.top_p is not NOT_GIVEN:
            inference_config["topP"] = request.top_p
        return inference_config

    async def _process_non_streaming_response(
        self,
        request: types_chat.ChatRequest,
        response: types_aiobotocore_bedrock_runtime.type_defs.ConverseResponseTypeDef,
    ) -> openai.types.chat.ChatCompletion:
        """非ストリーミングレスポンスをOpenAI形式に変換します。

        Args:
            response: Bedrockからのレスポンス
            model: モデル名

        Returns:
            ChatCompletion: OpenAI形式のレスポンス
        """
        logger.debug(f"{response=}")
        print(f"{response=}")  # TODO: 仮

        # response:
        #   output: ConverseOutputTypeDef
        #   stopReason: StopReasonType
        #   usage: TokenUsageTypeDef
        #   metrics: ConverseMetricsTypeDef
        #   additionalModelResponseFields: Dict[str, Any]
        #   trace: ConverseTraceTypeDef
        #   performanceConfig: PerformanceConfigurationTypeDef
        #   ResponseMetadata: ResponseMetadataTypeDef
        # 例：
        # {
        #     "ResponseMetadata": {
        #         "RequestId": "35a78c10-4d44-4f22-98bd-7e90ac7baf8b",
        #         "HTTPStatusCode": 200,
        #         "HTTPHeaders": {
        #             "date": "Tue, 20 May 2025 14:37:39 GMT",
        #             "content-type": "application/json",
        #             "content-length": "366",
        #             "connection": "keep-alive",
        #             "x-amzn-requestid": "35a78c10-4d44-4f22-98bd-7e90ac7baf8b",
        #         },
        #         "RetryAttempts": 0,
        #     },
        #     "output": {
        #         "message": {
        #             "role": "assistant",
        #             "content": [
        #                 {
        #                     "text": "こんにちは!どうぞよろしくお願いいたします。何かお手伝いできることはありますか?私はあなたのお役に立てるよう最善を尽くします。"
        #                 }
        #             ],
        #         }
        #     },
        #     "stopReason": "end_turn",
        #     "usage": {"inputTokens": 27, "outputTokens": 60, "totalTokens": 87},
        #     "metrics": {"latencyMs": 726},
        # }

        usage: openai.types.completion_usage.CompletionUsage | None = None
        if (bedrock_usage := response.get("usage")) is not None:
            usage = openai.types.completion_usage.CompletionUsage.model_construct(
                completion_tokens=bedrock_usage.get("inputTokens", 0),
                prompt_tokens=bedrock_usage.get("outputTokens", 0),
                total_tokens=bedrock_usage.get("totalTokens", 0),
                completion_tokens_details=openai.types.completion_usage.CompletionTokensDetails.model_construct(
                    # TODO: cacheWriteが無いっぽい
                    accepted_prediction_tokens=bedrock_usage.get(
                        "cacheWriteInputTokens", 0
                    )
                ),
                prompt_tokens_details=openai.types.completion_usage.PromptTokensDetails.model_construct(
                    cached_tokens=bedrock_usage.get("cacheReadInputTokens", 0)
                ),
            )

        return openai.types.chat.ChatCompletion.model_construct(
            id=str(uuid.uuid4()),
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": response["output"]["message"]["content"][0]["text"],
                    },
                    "finish_reason": self._get_finish_reason(
                        response.get("stopReason")
                    ),
                    "index": 0,
                }
            ],
            created=int(time.time()),
            model=request.model,
            object="chat.completion",
            system_fingerprint=None,
            usage=usage,
        )

    def _process_stream_event(
        self,
        request: types_chat.ChatRequest,
        event_data: types_aiobotocore_bedrock_runtime.type_defs.ConverseStreamOutputTypeDef,
    ) -> openai.types.chat.ChatCompletionChunk | None:
        """ストリームイベントをOpenAI形式のチャンクに変換します。

        Args:
            event_data: イベントデータ
            model: モデル名

        Returns:
            ChatCompletionChunk | None: OpenAI形式のチャンク。イベントが処理不要な場合はNone。
        """
        # class ConverseStreamOutputTypeDef(TypedDict):
        #     messageStart: NotRequired[MessageStartEventTypeDef]
        #     contentBlockStart: NotRequired[ContentBlockStartEventTypeDef]
        #     contentBlockDelta: NotRequired[ContentBlockDeltaEventTypeDef]
        #     contentBlockStop: NotRequired[ContentBlockStopEventTypeDef]
        #     messageStop: NotRequired[MessageStopEventTypeDef]
        #     metadata: NotRequired[ConverseStreamMetadataEventTypeDef]
        #     internalServerException: NotRequired[InternalServerExceptionTypeDef]
        #     modelStreamErrorException: NotRequired[ModelStreamErrorExceptionTypeDef]
        #     validationException: NotRequired[ValidationExceptionTypeDef]
        #     throttlingException: NotRequired[ThrottlingExceptionTypeDef]
        #     serviceUnavailableException: NotRequired[ServiceUnavailableExceptionTypeDef]

        if "contentBlockDelta" in event_data:
            delta_text = event_data["contentBlockDelta"]["delta"].get("text", "")
            return openai.types.chat.ChatCompletionChunk(
                id=str(uuid.uuid4()),
                choices=[
                    {
                        "delta": {"content": delta_text},
                        "finish_reason": None,
                        "index": 0,
                    }
                ],
                created=int(time.time()),
                model=request.model,
                object="chat.completion.chunk",
            )
        elif "messageStop" in event_data:
            return openai.types.chat.ChatCompletionChunk(
                id=str(uuid.uuid4()),
                choices=[
                    {
                        "delta": {},
                        "finish_reason": self._get_finish_reason(
                            event_data["messageStop"].get("stopReason")
                        ),
                        "index": 0,
                    }
                ],
                created=int(time.time()),
                model=request.model,
                object="chat.completion.chunk",
            )
        return None

    def _get_finish_reason(
        self, stop_reason: str | None
    ) -> typing.Literal["stop", "length", "tool_calls", "content_filter"] | None:
        """stopReasonをOpenAIのfinish_reasonに変換。"""
        if stop_reason is None:
            return None
        return {
            "end_turn": "stop",
            "tool_use": "tool_calls",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "guardrail_intervened": "content_filter",
            "content_filtered": "content_filter",
        }[stop_reason]


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
    if False:
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
            print("Chunk:", chunk.choices[0].delta.content)


if __name__ == "__main__":
    asyncio.run(main())
