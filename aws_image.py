"""AWSの画像生成APIの実装。

Stable DiffusionやNova Canvasなどに対応する。

参考:
- https://docs.aws.amazon.com/ja_jp/bedrock/latest/userguide/bedrock-runtime_example_bedrock-runtime_InvokeModel_StableDiffusion_section.html
- https://docs.aws.amazon.com/ja_jp/bedrock/latest/userguide/bedrock-runtime_example_bedrock-runtime_InvokeModel_AmazonNovaImageGeneration_section.html

"""

import logging
import random
import typing

import openai.types
from openai._types import NotGiven

import types_image

logger = logging.getLogger(__name__)


def convert_request(request: types_image.ImageRequest) -> dict[str, typing.Any]:
    """OpenAIのリクエストをBedrockのリクエストに変換。"""
    body: dict[str, typing.Any]

    if request.model.startswith("stability."):
        # Stable Diffusion
        body = {
            "text_prompts": [{"text": request.prompt}],
            "seed": random.randint(0, 2**32 - 1),
        }

        if not isinstance(request.size, NotGiven):
            width, height = map(int, str(request.size).split("x"))
            body["width"] = width
            body["height"] = height

        if not isinstance(request.style, NotGiven):
            style_map = {"vivid": "photographic", "natural": "digital-art"}
            body["style_preset"] = style_map.get(str(request.style), "photographic")

        return body

    elif request.model.startswith("amazon.nova-canvas"):
        # Amazon Nova Canvas
        body = {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {"text": request.prompt},
            "imageGenerationConfig": {
                "seed": random.randint(0, 858993459),
                "quality": "standard",
                "numberOfImages": 1 if isinstance(request.n, NotGiven) else request.n,
            },
        }

        config = typing.cast(dict[str, typing.Any], body["imageGenerationConfig"])
        if not isinstance(request.size, NotGiven):
            width, height = map(int, str(request.size).split("x"))
            config["width"] = width
            config["height"] = height
        else:
            config["width"] = 1024
            config["height"] = 1024

        if not isinstance(request.quality, NotGiven):
            config["quality"] = str(request.quality)

        return body

    else:
        raise ValueError(f"Unsupported model: {request.model}")


def convert_response(
    request: types_image.ImageRequest, response_body: dict[str, typing.Any]
) -> openai.types.ImagesResponse:
    """BedrockのレスポンスをOpenAIのレスポンスに変換。"""
    import time

    images = []

    if request.model.startswith("stability."):
        # Stable Diffusion
        artifacts = response_body.get("artifacts", [])
        for artifact in artifacts:
            b64_json = artifact.get("base64", "")
            images.append(openai.types.Image(b64_json=b64_json))

    elif request.model.startswith("amazon.nova-canvas"):
        # Amazon Nova Canvas
        image_data_list = response_body.get("images", [])
        for image_data in image_data_list:
            images.append(openai.types.Image(b64_json=image_data))

    else:
        raise ValueError(f"Unsupported model: {request.model}")

    return openai.types.ImagesResponse(created=int(time.time()), data=images)
