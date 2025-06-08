"""VertexAIのテキスト埋め込みAPI関連の実装。"""

import logging
import typing

import openai.types

import types_embedding

logger = logging.getLogger(__name__)


def convert_response(
    request: types_embedding.EmbeddingRequest, response: typing.Any
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
