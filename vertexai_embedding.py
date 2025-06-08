"""VertexAIのテキスト埋め込みAPI関連の実装。

参考:

- https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api

"""

import logging
import typing

import google.genai.types
import openai.types
from openai._types import NotGiven

import types_embedding

logger = logging.getLogger(__name__)


def convert_request(
    request: types_embedding.EmbeddingRequest,
) -> tuple[google.genai.types.EmbedContentConfig, google.genai.types.ContentListUnion]:
    """OpenAIのリクエストをVertexAIのリクエストに変換。"""
    contents = request.get_input()
    if len(contents) > 0 and isinstance(contents[0], list):
        raise ValueError("Vertex AI Embedding API does not support token arrays.")

    config = google.genai.types.EmbedContentConfig()
    if not isinstance(request.dimensions, NotGiven):
        config.output_dimensionality = request.dimensions

    config.auto_truncate = True

    return config, typing.cast(google.genai.types.ContentListUnion, contents)


def convert_response(
    request: types_embedding.EmbeddingRequest,
    response: google.genai.types.EmbedContentResponse,
) -> openai.types.CreateEmbeddingResponse:
    """VertexAIのレスポンスをOpenAIのレスポンスに変換。"""
    embeddings_list = []
    total_tokens = 0
    inputs = request.get_input()

    if response.embeddings:
        for i, embedding in enumerate(response.embeddings):
            if embedding.values:
                embeddings_list.append(embedding.values)
                # 簡易的なトークン数計算
                if i < len(inputs):
                    input_text = inputs[i]
                    if isinstance(input_text, list):
                        input_text = " ".join(map(str, input_text))
                    total_tokens += (
                        len(input_text.split())
                        if isinstance(input_text, str)
                        else len(input_text)
                    )

    return types_embedding.make_embedding_response(
        request=request, embedding_list=embeddings_list, prompt_tokens=total_tokens
    )
