"""OpenAI互換API用pydantic。"""

import typing
import warnings

import openai.types.chat
import openai.types.shared.metadata
import openai.types.shared.reasoning_effort
import pydantic
from openai._types import NOT_GIVEN, NotGiven


class ChatRequest(pydantic.BaseModel):
    """チャット補完APIのリクエスト。"""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    """pydanticの設定。"""

    messages: typing.Iterable[openai.types.chat.ChatCompletionMessageParam]
    """会話を構成するメッセージのリスト。

    使用するモデルによって、テキスト、画像、音声など、異なるメッセージタイプ（モダリティ）がサポートされています。
    """

    model: str
    """レスポンスの生成に使用するモデルID（例：`gpt-4o`、`o3`）。"""

    audio: openai.types.chat.ChatCompletionAudioParam | None | NotGiven = NOT_GIVEN
    """音声出力のためのパラメータ。

    `modalities: ["audio"]`で音声出力を要求する場合に必要です。
    """

    frequency_penalty: float | None | NotGiven = NOT_GIVEN
    """-2.0から2.0の間の数値。

    正の値は、テキスト内での既存の出現頻度に基づいて新しいトークンにペナルティを与え、
    モデルが同じ文を逐語的に繰り返す可能性を減少させます。
    """

    logit_bias: dict[str, int] | None | NotGiven = NOT_GIVEN
    """指定されたトークンが補完に出現する可能性を修正します。

    トークナイザー内のトークンIDからバイアス値（-100から100）へのマッピングを含むJSONオブジェクトを受け付けます。
    """

    logprobs: bool | None | NotGiven = NOT_GIVEN
    """出力トークンのログ確率を返すかどうか。

    trueの場合、`message`の`content`で返される各出力トークンのログ確率を返します。
    """

    max_completion_tokens: int | None | NotGiven = NOT_GIVEN
    """補完で生成できるトークンの上限。

    表示される出力トークンと推論トークンの両方を含みます。
    """

    max_tokens: int | None | NotGiven = NOT_GIVEN
    """チャット補完で生成できるトークンの最大数。

    APIを介して生成されるテキストのコストを制御するために使用できます。
    このパラメータは`max_completion_tokens`に置き換えられ、o-seriesモデルとは互換性がありません。
    """

    metadata: openai.types.shared.metadata.Metadata | None | NotGiven = NOT_GIVEN
    """オブジェクトに添付できる16個のキーと値のペアのセット。

    構造化された形式でオブジェクトに関する追加情報を保存し、APIまたはダッシュボードを介してオブジェクトを照会するのに役立ちます。
    """

    modalities: list[typing.Literal["text", "audio"]] | None | NotGiven = NOT_GIVEN
    """モデルに生成させたい出力タイプ。

    ほとんどのモデルはデフォルトでテキスト生成が可能です：`["text"]`。
    gpt-4o-audio-previewモデルは音声も生成できます：`["text", "audio"]`。
    """

    n: int | None | NotGiven = NOT_GIVEN
    """各入力メッセージに対して生成するチャット補完の選択肢の数。

    生成されたトークンの数に基づいて課金されることに注意してください。コストを最小限に抑えるために`n`を`1`に保ちます。
    """

    parallel_tool_calls: bool | NotGiven = NOT_GIVEN
    """ツール使用中の並列関数呼び出しを有効にするかどうか。"""

    prediction: (
        openai.types.chat.ChatCompletionPredictionContentParam | None | NotGiven
    ) = NOT_GIVEN
    """テキストファイルの内容など、再生成される静的な予測出力。"""

    presence_penalty: float | None | NotGiven = NOT_GIVEN
    """-2.0から2.0の間の数値。

    正の値は、トークンがテキストに既に出現しているかどうかに基づいて新しいトークンにペナルティを与え、
    モデルが新しいトピックについて話す可能性を高めます。
    """

    reasoning_effort: (
        openai.types.shared.reasoning_effort.ReasoningEffort | None | NotGiven
    ) = NOT_GIVEN
    """o-seriesモデルのみ。低、中、高の値をサポートし、推論の労力を制限します。

    推論の労力を減らすと、応答が速くなり、使用されるトークンが少なくなる可能性があります。
    """

    response_format: (
        openai.types.chat.completion_create_params.ResponseFormat | NotGiven
    ) = NOT_GIVEN
    """モデルが出力する必要のある形式を指定するオブジェクト。JSON出力を強制するために使用できます。"""

    seed: int | None | NotGiven = NOT_GIVEN
    """ベータ機能。指定された場合、システムは決定論的なサンプリングを行うよう努めます。"""

    service_tier: typing.Literal["auto", "default", "flex"] | None | NotGiven = (
        NOT_GIVEN
    )
    """リクエスト処理のレイテンシーティアを指定します。

    スケールティアサービスに登録しているお客様向けのパラメータです。
    """

    stop: str | list[str] | None | NotGiven = NOT_GIVEN
    """モデルがそれ以上のトークンを生成するのを停止する最大4つのシーケンス。

    最新の推論モデル`o3`と`o4-mini`ではサポートされていません。
    """

    store: bool | None | NotGiven = NOT_GIVEN
    """このチャット補完リクエストの出力をモデル蒸留またはevals製品で使用するために保存するかどうか。"""

    stream: bool = False
    """trueに設定すると、モデルのレスポンスデータは生成時にサーバー送信イベントを使用してクライアントにストリーミングされます。"""

    stream_options: (
        openai.types.chat.ChatCompletionStreamOptionsParam | None | NotGiven
    ) = NOT_GIVEN
    """ストリーミングレスポンスのオプション。`stream: true`の場合のみ設定します。"""

    temperature: float | None | NotGiven = NOT_GIVEN
    """0から2の間のサンプリング温度。0.8のような高い値は出力をよりランダムにし、0.2のような低い値はより集中的で決定論的にします。"""

    tool_choice: openai.types.chat.ChatCompletionToolChoiceOptionParam | NotGiven = (
        NOT_GIVEN
    )
    """モデルが呼び出すツール（存在する場合）を制御します。"""

    tools: typing.Iterable[openai.types.chat.ChatCompletionToolParam] | NotGiven = (
        NOT_GIVEN
    )
    """モデルが呼び出す可能性のあるツールのリスト。現在、関数のみがツールとしてサポートされています。"""

    top_logprobs: int | None | NotGiven = NOT_GIVEN
    """各トークン位置で返す最も可能性の高いトークンの数（0から20の整数）。

    このパラメータを使用する場合は`logprobs`をtrueに設定する必要があります。
    """

    top_p: float | None | NotGiven = NOT_GIVEN
    """温度によるサンプリングの代替として、モデルがtop_p確率質量を持つトークンの結果を考慮するnucleusサンプリング。"""

    user: str | NotGiven = NOT_GIVEN
    """エンドユーザーの安定した識別子。

    類似したリクエストをより適切にバケット化することでキャッシュヒット率を向上させ、
    OpenAIが悪用を検出および防止するのに役立ちます。
    """

    web_search_options: (
        openai.types.chat.completion_create_params.WebSearchOptions | NotGiven
    ) = NOT_GIVEN
    """レスポンスで使用する関連結果をウェブから検索するツール。"""

    @pydantic.model_validator(mode="after")
    def validate_after(self) -> typing.Self:
        # max_tokens / max_completion_tokens
        has_max_tokens = not isinstance(self.max_tokens, NotGiven)
        has_max_completion_tokens = not isinstance(self.max_completion_tokens, NotGiven)
        if has_max_tokens and has_max_completion_tokens:
            raise ValueError(
                "max_tokens と max_completion_tokens は同時に指定できません"
            )
        if has_max_tokens:
            warnings.warn(
                "max_tokens は非推奨です。代わりに max_completion_tokens を使用してください。",
                DeprecationWarning,
                stacklevel=3,
            )
            self.max_completion_tokens = self.max_tokens
            self.max_tokens = NOT_GIVEN

        # Iterable to list
        if isinstance(self.messages, typing.Iterable):
            self.messages = list(self.messages)
            for message in self.messages:
                tool_calls = message.get("tool_calls")
                if tool_calls is not None and isinstance(tool_calls, typing.Iterable):
                    message["tool_calls"] = list(message["tool_calls"])  # type: ignore
        if isinstance(self.tools, typing.Iterable):
            self.tools = list(self.tools)

        return self
