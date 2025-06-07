"""OpenAI Image Generation API互換API用pydantic。"""

import typing

import pydantic
from openai._types import NOT_GIVEN, NotGiven


class ImageRequest(pydantic.BaseModel):
    """画像生成APIのリクエスト。"""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    """pydanticの設定。"""

    prompt: str
    """画像生成のプロンプト。

    英語で指定する必要があります。空白やNULLを指定することはできず、
    最大1000文字に制限されています。
    """

    background: typing.Literal["transparent", "opaque", "auto"] | None | NotGiven = (
        NOT_GIVEN
    )
    """生成画像の背景透過設定を可能にするパラメータです。このパラメータは `gpt-image-1` モデルでのみ有効です。

    値は `transparent`、`opaque`、または `auto`（デフォルト値）のいずれかを指定します。
    `auto` を選択した場合、モデルが自動的に最適な背景を選択します。
    `transparent` を選択した場合、出力形式は透過をサポートしている必要があるため、`png`（デフォルト値）または `webp` のいずれかを指定してください。
    """

    model: str
    """OpenAIのImage Generation APIはDALL·E-3やDALL·E-2など、
    複数のモデルバリエーションを提供しています。

    利用可能なモデルの最新のリスト、それらの能力と違いについては、
    Modelsのドキュメントを参照してください。
    """

    moderation: typing.Literal["low", "auto"] | None | NotGiven = NOT_GIVEN
    """`gpt-image-1` が生成する画像に対するコンテンツ検閲レベルを制御します。制限の少ないフィルタリングを行う場合は `low` を、デフォルト値である `auto` を指定してください。"""

    n: int | NotGiven = NOT_GIVEN
    """生成する画像の数。1-10の整数で指定します。デフォルトは1です。"""

    quality: typing.Literal["standard", "hd"] | NotGiven = NOT_GIVEN
    """生成される画像の品質を'standard'または'hd'で指定します。

    hdは標準よりもより詳細で、より正確な画像を生成しますが、
    生成に時間がかかり、より高価です。デフォルトは'standard'です。
    """

    output_compression: int | None | NotGiven = NOT_GIVEN
    """生成画像の圧縮レベル（0～100%）を指定します。このパラメータは `gpt-image-1` モデルにおいて `webp` または `jpeg` 出力形式を選択した場合にのみ有効で、デフォルト値は 100 です。
    """

    output_format: typing.Literal["png", "jpeg", "webp"] | None | NotGiven = NOT_GIVEN
    """生成画像の出力形式を指定します。このパラメータは `gpt-image-1` モデルでのみ有効です。指定可能な形式は `png`、`jpeg`、または `webp` のいずれかです。"""

    response_format: typing.Literal["url", "b64_json"] | NotGiven = NOT_GIVEN
    """`dall-e-2` および `dall-e-3` で生成された画像の返却形式。

    `url` または `b64_json` のいずれかを指定する必要があります。URL は画像生成後60分間のみ有効です。
    このパラメータは `gpt-image-1` ではサポートされておらず、常に base64 エンコードされた画像が返されます。
    """

    size: typing.Literal["1024x1024", "1792x1024", "1024x1792"] | NotGiven = NOT_GIVEN
    """生成される画像のサイズ。

    `gpt-image-1` の場合は `1024x1024`、`1536x1024`（横長）、`1024x1536`（縦長）、またはデフォルト値の `auto` のいずれかを指定する必要があります。
    `dall-e-2` の場合は `256x256`、`512x512`、または `1024x1024`、
    `dall-e-3` の場合は `1024x1024`、`1792x1024`、または `1024x1792` のいずれかを指定してください。
    """

    style: typing.Literal["vivid", "natural"] | NotGiven = NOT_GIVEN
    """生成画像のスタイル。このパラメータは `dall-e-3` モデルでのみ有効です。

    `vivid` または `natural` のいずれかを指定してください。
    `vivid` を指定するとモデルは超写実的で劇的な画像を生成する傾向が強くなります。
    `natural` を指定すると、より自然で過度な写実性を抑えた画像が生成されます。
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
