"""AWSのテキスト埋め込みAPI関連の実装。"""

import logging
import typing

import openai.types

import types_embedding

logger = logging.getLogger(__name__)


def convert_request(request: types_embedding.EmbeddingRequest) -> dict[str, typing.Any]:
    """OpenAIのリクエストをBedrockのリクエストに変換。"""
    # TODO: 実装
    del request
    return {}


def convert_response(
    request: types_embedding.EmbeddingRequest, response_body: dict[str, typing.Any]
) -> openai.types.CreateEmbeddingResponse:
    """BedrockのレスポンスをOpenAIのレスポンスに変換。"""
    # TODO: 実装
    del response_body
    return openai.types.CreateEmbeddingResponse(
        object="list",
        data=[
            openai.types.Embedding(object="embedding", index=0, embedding=[0.0] * 1536)
        ],
        model=request.model,
        usage=openai.types.create_embedding_response.Usage(
            prompt_tokens=0, total_tokens=0
        ),
    )
