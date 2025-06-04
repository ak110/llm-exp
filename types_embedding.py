"""OpenAI Embedding API互換API用pydantic。"""

import base64
import typing

import numpy as np
import numpy.typing as npt
import openai.types
import pydantic
from openai._types import NOT_GIVEN, NotGiven


class EmbeddingRequest(pydantic.BaseModel):
    """埋め込みAPIのリクエスト。"""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    """pydanticの設定。"""

    model: str
    """利用可能なモデルを確認するには、List ModelsのAPIを使用するか、
    Model overviewでモデルの説明を参照してください。
    """

    input: str | list[str] | list[int] | list[list[int]]
    """埋め込むテキストで、文字列またはトークンの配列としてエンコードされます。

    1回のリクエストで複数の入力を埋め込むには、文字列の配列またはトークン配列の配列を渡します。
    入力は、モデルの最大入力トークン（すべての埋め込みモデルで8,192トークン）を超えてはならず、
    空の文字列であってはならず、配列は2048次元以下である必要があります。

    また、すべての埋め込みモデルは、単一のリクエストですべての入力を合計して300,000トークンの制限を適用します。
    """

    dimensions: int | NotGiven = NOT_GIVEN
    """生成される埋め込みベクトルの次元数を指定します。

    text-embedding-3以降のモデルでのみサポートされています。
    """

    encoding_format: typing.Literal["float", "base64"] | NotGiven = NOT_GIVEN
    """埋め込みを返す形式を指定します。

    'float'(デフォルト)または'base64'を指定できます。
    """

    user: str | NotGiven = NOT_GIVEN
    """エンドユーザーを表す一意の識別子。

    OpenAIが不正利用を監視および検出するのに役立ちます。
    """

    # 以下は提供しない
    # extra_headers: Send extra headers
    # extra_query: Add additional query parameters to the request
    # extra_body: Add additional JSON properties to the request
    # timeout: Override the client-level default timeout for this request, in seconds

    def get_input(self) -> list[str] | list[list[int]]:
        """入力を取得する。(型の種類を減らすためのもの)"""
        if isinstance(self.input, str):
            return [self.input]
        elif isinstance(self.input, list) and all(
            isinstance(i, int) for i in self.input
        ):
            return [self.input]
        else:
            return self.input


def make_embedding_response(
    embedding_list: list[npt.NDArray] | list[list[float]],
    model: str,
    prompt_tokens: int,
    encoding_format: typing.Literal["float", "base64"] = "float",
) -> openai.types.CreateEmbeddingResponse:
    """テキスト埋め込みAPIのレスポンスを作成して返す。"""
    return openai.types.CreateEmbeddingResponse(
        data=[
            openai.types.embedding.Embedding.model_construct(
                embedding=encode_embedding(embedding, encoding_format),  # type: ignore[arg-type]
                index=i,
                object=embedding,
            )
            for i, embedding in enumerate(embedding_list)
        ],
        model=model,
        object="list",
        usage=openai.types.create_embedding_response.Usage(
            prompt_tokens=prompt_tokens, total_tokens=prompt_tokens
        ),
    )


def encode_embedding(
    embedding: npt.NDArray | list[float],
    encoding_format: typing.Literal["float", "base64"],
) -> list[float] | str:
    """埋め込みを指定された形式でエンコードする。"""
    if encoding_format == "float":
        if isinstance(embedding, list):
            return embedding
        return embedding.tolist()
    embedding = np.asarray(embedding, dtype=np.float32)
    return base64.b64encode(embedding.tobytes()).decode("utf-8")
