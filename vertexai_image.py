"""VertexAIの画像生成API関連の実装。"""

import logging
import typing

import google.genai.types
import openai.types
from openai._types import NOT_GIVEN

import types_image

logger = logging.getLogger(__name__)


def convert_request(
    request: types_image.ImageRequest,
) -> google.genai.types.GenerateContentConfig:
    """OpenAIのリクエストをVertexAIのリクエストに変換。"""
    generation_config = google.genai.types.GenerateContentConfig()
    if isinstance(request.n, int):
        generation_config.candidate_count = request.n
    return generation_config, request.prompt


def convert_response(
    request: types_image.ImageRequest, response: typing.Any
) -> openai.types.ImagesResponse:
    """VertexAIのレスポンスをOpenAIのレスポンスに変換。"""
    images = []
    if response.candidates:
        for candidate in response.candidates:
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, "inline_data") and part.inline_data:
                        if request.response_format in ("b64_json", NOT_GIVEN):
                            images.append(
                                openai.types.Image(
                                    b64_json=(
                                        part.inline_data.data.decode("utf-8")
                                        if isinstance(part.inline_data.data, bytes)
                                        else part.inline_data.data
                                    ),
                                    revised_prompt=request.prompt,
                                )
                            )
                        else:
                            data_str = (
                                part.inline_data.data.decode("utf-8")
                                if isinstance(part.inline_data.data, bytes)
                                else part.inline_data.data
                            )
                            images.append(
                                openai.types.Image(
                                    url=f"data:{part.inline_data.mime_type};base64,{data_str}",
                                    revised_prompt=request.prompt,
                                )
                            )

    return openai.types.ImagesResponse(
        created=int(response.model_dump().get("created", 0)), data=images
    )
