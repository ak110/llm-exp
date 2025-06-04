"""AWSの画像生成APIの実装。"""

import logging
import typing

import openai.types

import types_image

logger = logging.getLogger(__name__)


def convert_request(request: types_image.ImageRequest) -> dict[str, typing.Any]:
    """OpenAIのリクエストをBedrockのリクエストに変換。"""
    # TODO: 実装
    del request
    return {}


def convert_response(
    request: types_image.ImageRequest, response_body: dict[str, typing.Any]
) -> openai.types.ImagesResponse:
    """BedrockのレスポンスをOpenAIのレスポンスに変換。"""
    # TODO: 実装
    del request, response_body
    return openai.types.ImagesResponse(created=0, data=[openai.types.Image()])
