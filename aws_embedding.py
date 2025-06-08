"""AWSのテキスト埋め込みAPI関連の実装。

Amazon Titan Text Embeddingsや、Cohere Embedモデルなどに対応する

参考:
- https://docs.aws.amazon.com/ja_jp/bedrock/latest/userguide/model-parameters-titan-embed-text.html
- https://docs.aws.amazon.com/ja_jp/bedrock/latest/userguide/model-parameters-embed.html

"""

import logging
import typing

import openai.types
from openai._types import NotGiven

import types_embedding

logger = logging.getLogger(__name__)


def convert_request(request: types_embedding.EmbeddingRequest) -> dict[str, typing.Any]:
    """OpenAIのリクエストをBedrockのリクエストに変換。"""
    input_data = request.get_input()

    if request.model.startswith("amazon.titan-embed-text"):
        # Amazon Titan Text Embeddings V2
        if len(input_data) > 1:
            raise ValueError("Titan Embed Text V2 supports only single input")

        text = input_data[0]
        if isinstance(text, list):
            raise ValueError("Titan Embed Text V2 does not support token arrays")

        body = {"inputText": text}

        if not isinstance(request.dimensions, NotGiven):
            if request.dimensions not in [256, 512, 1024]:
                raise ValueError(
                    "Dimensions must be 256, 512, or 1024 for Titan Embed Text"
                )
            body["dimensions"] = request.dimensions

        return body

    elif request.model.startswith("cohere.embed"):
        # Cohere Embed
        texts = []
        for item in input_data:
            if isinstance(item, list):
                raise ValueError("Cohere Embed does not support token arrays")
            texts.append(item)

        if len(texts) > 96:
            raise ValueError("Cohere Embed supports maximum 96 texts per request")

        body = {
            "texts": texts,
            "input_type": "search_document",  # デフォルトとして使用
            "truncate": "END",
        }

        return body

    else:
        raise ValueError(f"Unsupported model: {request.model}")


def convert_response(
    request: types_embedding.EmbeddingRequest, response_body: dict[str, typing.Any]
) -> openai.types.CreateEmbeddingResponse:
    """BedrockのレスポンスをOpenAIのレスポンスに変換。"""

    if request.model.startswith("amazon.titan-embed-text"):
        # Amazon Titan Text Embeddings V2
        embedding_data = response_body.get("embedding", [])
        prompt_tokens = response_body.get("inputTextTokenCount", 0)

        return types_embedding.make_embedding_response(
            request=request,
            embedding_list=[embedding_data],
            prompt_tokens=prompt_tokens,
        )

    elif request.model.startswith("cohere.embed"):
        # Cohere Embed
        embeddings = response_body.get("embeddings", [])
        # Cohereは入力テキストのトークン数を返さないため、推定値を使用
        input_data = request.get_input()
        estimated_tokens = sum(
            len(text.split()) if isinstance(text, str) else len(text)
            for text in input_data
        )

        return types_embedding.make_embedding_response(
            request=request, embedding_list=embeddings, prompt_tokens=estimated_tokens
        )

    else:
        raise ValueError(f"Unsupported model: {request.model}")
