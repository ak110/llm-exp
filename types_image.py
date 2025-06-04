"""OpenAI Image Generation API互換API用pydantic。"""

import typing

import pydantic
from openai._types import NOT_GIVEN, NotGiven


class ImageRequest(pydantic.BaseModel):
    """画像生成APIのリクエスト。"""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    """pydanticの設定。"""

    model: str
    """OpenAIのImage Generation APIはDALL·E-3やDALL·E-2など、
    複数のモデルバリエーションを提供しています。

    利用可能なモデルの最新のリスト、それらの能力と違いについては、
    Modelsのドキュメントを参照してください。
    """

    prompt: str
    """画像生成のプロンプト。

    英語で指定する必要があります。空白やNULLを指定することはできず、
    最大1000文字に制限されています。
    """

    n: int | NotGiven = NOT_GIVEN
    """生成する画像の数。1-10の整数で指定します。デフォルトは1です。"""

    quality: typing.Literal["standard", "hd"] | NotGiven = NOT_GIVEN
    """生成される画像の品質を'standard'または'hd'で指定します。

    hdは標準よりもより詳細で、より正確な画像を生成しますが、
    生成に時間がかかり、より高価です。デフォルトは'standard'です。
    """

    response_format: typing.Literal["url", "b64_json"] | NotGiven = NOT_GIVEN
    """生成される画像の返却形式を指定します。

    'url'(デフォルト): 一時的なURLを返します。
    'b64_json': Base64エンコードされたJSON文字列を返します。
    """

    size: typing.Literal["1024x1024", "1792x1024", "1024x1792"] | NotGiven = NOT_GIVEN
    """生成される画像のサイズを指定します。

    以下の3つのオプションから選択できます：
    - 1024x1024 (正方形)
    - 1792x1024 (横長)
    - 1024x1792 (縦長)

    デフォルトは1024x1024です。
    """

    style: typing.Literal["vivid", "natural"] | NotGiven = NOT_GIVEN
    """生成される画像のスタイルを指定します。

    'vivid': より劇的で、印象的な画像を生成します
    'natural': より忠実で自然な画像を生成します

    デフォルトは'vivid'です。
    """

    user: str | NotGiven = NOT_GIVEN
    """エンドユーザーの安定した識別子。

    不正利用の検出と防止に役立ちます。
    """

    # 以下は提供しない
    # extra_headers: Send extra headers
    # extra_query: Add additional query parameters to the request
    # extra_body: Add additional JSON properties to the request
    # timeout: Override the client-level default timeout for this request, in seconds
