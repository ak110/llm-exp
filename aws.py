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
import logging
import os
import time
import typing
import uuid

import httpx
import iam_rolesanywhere_session
import mypy_boto3_bedrock_runtime.type_defs
import openai._streaming
import openai.types.chat
import openai.types.completion_usage
import openai.types.shared.metadata
import openai.types.shared.reasoning_effort
from openai._types import NOT_GIVEN, Body, Headers, NotGiven, Query

import config

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

    async def chat_completion(
        self,
        messages: typing.Iterable[openai.types.chat.ChatCompletionMessageParam],
        model: str,
        audio: openai.types.chat.ChatCompletionAudioParam | None | NotGiven = NOT_GIVEN,
        frequency_penalty: float | None | NotGiven = NOT_GIVEN,
        function_call: (
            openai.types.chat.completion_create_params.FunctionCall | NotGiven
        ) = NOT_GIVEN,
        functions: (
            typing.Iterable[openai.types.chat.completion_create_params.Function]
            | NotGiven
        ) = NOT_GIVEN,
        logit_bias: dict[str, int] | None | NotGiven = NOT_GIVEN,
        logprobs: bool | None | NotGiven = NOT_GIVEN,
        max_completion_tokens: int | None | NotGiven = NOT_GIVEN,
        max_tokens: int | None | NotGiven = NOT_GIVEN,
        metadata: openai.types.shared.metadata.Metadata | None | NotGiven = NOT_GIVEN,
        modalities: list[typing.Literal["text", "audio"]] | None | NotGiven = NOT_GIVEN,
        n: int | None | NotGiven = NOT_GIVEN,
        parallel_tool_calls: bool | NotGiven = NOT_GIVEN,
        prediction: (
            openai.types.chat.ChatCompletionPredictionContentParam | None | NotGiven
        ) = NOT_GIVEN,
        presence_penalty: float | None | NotGiven = NOT_GIVEN,
        reasoning_effort: (
            openai.types.shared.reasoning_effort.ReasoningEffort | None | NotGiven
        ) = NOT_GIVEN,
        response_format: (
            openai.types.chat.completion_create_params.ResponseFormat | NotGiven
        ) = NOT_GIVEN,
        seed: int | None | NotGiven = NOT_GIVEN,
        service_tier: (
            typing.Literal["auto", "default", "flex"] | None | NotGiven
        ) = NOT_GIVEN,
        stop: str | None | list[str] | None | NotGiven = NOT_GIVEN,
        store: bool | None | NotGiven = NOT_GIVEN,
        stream: typing.Literal[False] | None | NotGiven = NOT_GIVEN,
        stream_options: (
            openai.types.chat.ChatCompletionStreamOptionsParam | None | NotGiven
        ) = NOT_GIVEN,
        temperature: float | None | NotGiven = NOT_GIVEN,
        tool_choice: (
            openai.types.chat.ChatCompletionToolChoiceOptionParam | NotGiven
        ) = NOT_GIVEN,
        tools: (
            typing.Iterable[openai.types.chat.ChatCompletionToolParam] | NotGiven
        ) = NOT_GIVEN,
        top_logprobs: int | None | NotGiven = NOT_GIVEN,
        top_p: float | None | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        web_search_options: (
            openai.types.chat.completion_create_params.WebSearchOptions | NotGiven
        ) = NOT_GIVEN,
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> (
        openai.types.chat.ChatCompletion
        | typing.AsyncGenerator[openai.types.chat.ChatCompletionChunk]
    ):
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
            if message["role"] not in ("user", "assistant"):
                continue
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
        bedrock = self.session.client(
            service_name="bedrock-runtime", region_name="ap-northeast-1"
        )
        if stream:
            response = bedrock.converse_stream(
                modelId=model,
                messages=formatted_messages,
                system=system,
                inferenceConfig=inference_config,
                additionalModelRequestFields={},
                # guardrailConfig={"guardrailVersion": "", "guardrailIdentifier": ""},
                # toolConfig={"toolChoice": {...}, "tools": [{...}]},
            )
            return await self._process_streaming_response(response, model)
        else:
            response = bedrock.converse(
                modelId=model,
                messages=formatted_messages,
                system=system,
                inferenceConfig=inference_config,
                additionalModelRequestFields={},
                # guardrailConfig={"guardrailVersion": "", "guardrailIdentifier": ""},
                # toolConfig={"toolChoice": {...}, "tools": [{...}]},
            )
            return await self._process_non_streaming_response(response, model)

    async def _process_non_streaming_response(
        self,
        response: mypy_boto3_bedrock_runtime.type_defs.ConverseResponseTypeDef,
        model: str,
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

        finish_reason: (
            typing.Literal[
                "stop", "length", "tool_calls", "content_filter", "function_call"
            ]
            | None
        ) = {
            "end_turn": "stop",
            "tool_use": "tool_calls",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "guardrail_intervened": "content_filter",
            "content_filtered": "content_filter",
            None: None,
        }[
            response.get("stopReason")
        ]
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
                    "finish_reason": finish_reason,
                    "index": 0,
                }
            ],
            created=int(time.time()),
            model=model,
            object="chat.completion",
            system_fingerprint=None,
            usage=usage,
        )

    async def _process_streaming_response(
        self,
        response: mypy_boto3_bedrock_runtime.type_defs.ConverseStreamResponseTypeDef,
        model: str,
    ) -> typing.AsyncGenerator[openai.types.chat.ChatCompletionChunk]:
        """ストリーミングレスポンスを処理します。

        Args:
            response: Bedrockからのレスポンス
            model: モデル名

        Returns:
            ストリーミングレスポンス
        """
        for event in response["stream"]:
            if chunk := self._process_stream_event(event, model):
                yield chunk

    def _process_stream_event(
        self,
        event_data: mypy_boto3_bedrock_runtime.type_defs.ConverseStreamOutputTypeDef,
        model: str,
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
                model=model,
                object="chat.completion.chunk",
            )
        elif "messageStop" in event_data:
            return openai.types.chat.ChatCompletionChunk(
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
    asyncio.run(main())
