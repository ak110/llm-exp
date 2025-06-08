"""VertexAIの画像生成API関連の実装。

参考:
- https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/imagen-api

"""

import base64
import logging
import time

import google.genai.types
import openai.types

import types_image

logger = logging.getLogger(__name__)


def convert_request(
    request: types_image.ImageRequest,
) -> google.genai.types.GenerateImagesConfig:
    """OpenAIのリクエストをVertexAIのリクエストに変換。"""
    generation_config = google.genai.types.GenerateImagesConfig()
    if isinstance(request.n, int):
        generation_config.number_of_images = request.n
    return generation_config


def convert_response(
    request: types_image.ImageRequest,  # pylint: disable=unused-argument
    response: google.genai.types.GenerateImagesResponse,
) -> openai.types.ImagesResponse:
    """VertexAIのレスポンスをOpenAIのレスポンスに変換。"""
    images = []

    if response.generated_images:
        for generated_image in response.generated_images:
            b64_json = None
            if generated_image.image and generated_image.image.image_bytes:
                b64_json = base64.b64encode(generated_image.image.image_bytes).decode(
                    "utf-8"
                )
            images.append(openai.types.Image(b64_json=b64_json))

    return openai.types.ImagesResponse(created=int(time.time()), data=images)
