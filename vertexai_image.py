"""VertexAIの画像生成API関連の実装。

参考:
- https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/imagen-api

"""

import base64
import logging
import time

import google.genai.types
import openai.types
from openai._types import NotGiven

import types_image

logger = logging.getLogger(__name__)


def convert_request(
    request: types_image.ImageRequest,
) -> google.genai.types.GenerateImagesConfig:
    """OpenAIのリクエストをVertexAIのリクエストに変換。"""
    generation_config = google.genai.types.GenerateImagesConfig()

    if not isinstance(request.n, NotGiven):
        generation_config.number_of_images = request.n

    if not isinstance(request.size, NotGiven):
        width, height = map(int, str(request.size).split("x"))
        # アスペクト比へ変換する
        while width % 2 == 0 and height % 2 == 0:
            width //= 2
            height //= 2
        # Imagen3 の場合、"1:1"、"3:4"、"4:3"、"9:16"、"16:9" がサポートされている
        # https://ai.google.dev/gemini-api/docs/imagen?hl=ja
        # ここでは気にせず渡してしまう
        generation_config.aspect_ratio = f"{width}:{height}"

    generation_config.safety_filter_level = (
        google.genai.types.SafetyFilterLevel.BLOCK_ONLY_HIGH
    )
    generation_config.include_safety_attributes = True
    generation_config.include_rai_reason = True
    return generation_config


def convert_response(
    request: types_image.ImageRequest,  # pylint: disable=unused-argument
    response: google.genai.types.GenerateImagesResponse,
) -> openai.types.ImagesResponse:
    """VertexAIのレスポンスをOpenAIのレスポンスに変換。"""
    if response.generated_images is None:
        raise ValueError(
            "No images were generated in the response."
            f" Check the request parameters. {response.positive_prompt_safety_attributes}"
        )

    data: list[openai.types.Image] = []
    for image in response.generated_images:
        # 画像が生成されなかった場合の処理
        if image.image is None and image.rai_filtered_reason is not None:
            raise ValueError(f"Image generation failed: {image.rai_filtered_reason}")

        # 画像が生成された場合の処理
        b64_json = None
        if image.image and image.image.image_bytes:
            b64_json = base64.b64encode(image.image.image_bytes).decode("utf-8")
        data.append(
            openai.types.Image(b64_json=b64_json, revised_prompt=image.enhanced_prompt)
        )

    return openai.types.ImagesResponse(created=int(time.time()), data=data)
