"""エラー処理関連。"""

import httpx
import openai


class APIError(openai.APIStatusError):
    """エラー。"""

    def __init__(
        self,
        message: str,
        code: str | None = None,
        status_code: int | None = None,
        param: str | None = None,
        type_: str | None = None,
        details: str | None = None,
    ) -> None:
        if code is None:
            code = "internal_server_error"
        if status_code is None:
            status_code = 500
        if type_ is None:
            type_ = "api_error"
        if details is not None:
            # 詳細メッセージはmessageに1行あけて追加する
            message += f"\n\n{details}"
        super().__init__(
            message,
            response=httpx.Response(
                status_code=status_code,
                request=httpx.Request(method="POST", url="https://api.openai.com/v1"),
            ),
            body={"message": message, "type": type_, "code": code, "param": param},
        )


class InvalidRequestError(APIError):
    """無効なリクエストエラー。"""

    def __init__(
        self,
        message: str,
        code: str = "invalid_request_error",
        param: str | None = None,
        details: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            status_code=400,
            param=param,
            type_="invalid_request_error",
            details=details,
        )


class MissingRequiredParameter(InvalidRequestError):
    """必須パラメータが欠落しているエラー。"""

    def __init__(self, param: str) -> None:
        super().__init__(
            message=f"必須パラメータ '{param}' がリクエストに含まれません。",
            code="missing_required_parameter",
            param=param,
        )


class InvalidParameterValue(InvalidRequestError):
    """無効なパラメータ値エラー。"""

    def __init__(self, param: str, value: str) -> None:
        super().__init__(
            message=f"パラメータ '{param}' の値 '{value}' は無効です。",
            code="invalid_parameter_value",
            param=param,
        )


class ContextWindowExceededError(InvalidRequestError):
    """コンテキストウィンドウを超えたエラー。"""

    def __init__(self, details: str | None = None) -> None:
        super().__init__(
            message="入力データ長が大きすぎるため処理できませんでした。"
            "新しいセッションで再試行するか、より最大入力長の長いモデルを使用してください。",
            code="context_window_exceeded",
            details=details,
        )


class ContentPolicyViolationError(InvalidRequestError):
    """コンテンツポリシー違反エラー。"""

    def __init__(self, details: str | None = None) -> None:
        super().__init__(
            message="コンテンツフィルターによりエラーになりました。"
            "許可されていないテキストが含まれている可能性があります。"
            "誤検出と思われる場合は、プロンプトを調整するか、他のモデルを使用してみてください。",
            code="content_policy_violation",
            details=details,
        )


class RateLimitError(APIError):
    """レート制限エラー。"""

    def __init__(self, details: str | None = None) -> None:
        super().__init__(
            message="API提供元のレート制限を超過しました。"
            "他のモデルを使用するか、しばらく待ってから再度お試しください。",
            code="rate_limit_exceeded",
            status_code=429,
            type_="rate_limit_error",
            details=details,
        )


# 以上で足りないエラーは基本的には特殊なエラーのはずなのでAPIErrorを使う


def map_exception(e: Exception) -> APIError:
    """例外を適切なエラーにマッピングする。"""
    if isinstance(e, APIError):
        return e

    error_str = str(e)

    # OpenAI例外の処理
    if isinstance(e, openai.BadRequestError):
        if (
            "maximum context length" in error_str
            or "context_length_exceeded" in error_str
        ):
            return ContextWindowExceededError(details=error_str)
        if "content_policy_violation" in error_str or "content filter" in error_str:
            return ContentPolicyViolationError(details=error_str)
        return InvalidRequestError(
            message="リクエスト内容に何らかのエラーがありました",
            code="bad_request",
            details=error_str,
        )
    if isinstance(e, openai.AuthenticationError):
        return InvalidRequestError(
            message="APIキーを確認してください。",
            code="authentication_error",
            details=error_str,
        )
    if isinstance(e, openai.PermissionDeniedError):
        return InvalidRequestError(
            message="APIキーの権限を確認してください。",
            code="permission_denied",
            details=error_str,
        )
    if isinstance(e, openai.NotFoundError):
        return InvalidRequestError(
            message="リソースが見つかりません。モデル名などを確認してください。",
            code="not_found",
            details=error_str,
        )
    if isinstance(e, openai.RateLimitError):
        return RateLimitError(details=error_str)
    if isinstance(e, openai.InternalServerError):
        return APIError(
            message="内部サーバーエラーが発生しました。",
            code="internal_server_error",
            status_code=500,
            details=error_str,
        )
    if isinstance(e, openai.UnprocessableEntityError):
        return InvalidRequestError(
            message="リクエスト内容を処理できませんでした。",
            code="unprocessable_entity",
            details=error_str,
        )
    if isinstance(e, openai.APITimeoutError):
        return APIError(
            message="APIリクエストがタイムアウトしました。",
            code="timeout_error",
            status_code=408,
            details=error_str,
        )
    if isinstance(e, openai.APIConnectionError):
        return APIError(
            message="APIへの接続でエラーが発生しました。",
            code="connection_error",
            status_code=503,
            details=error_str,
        )
    if isinstance(e, (openai.APIError, openai.APIStatusError)):
        # 汎用的なOpenAI APIエラー
        if hasattr(e, "status_code"):
            status_code = e.status_code
            if status_code == 400:
                return InvalidRequestError(
                    message="リクエスト内容に何らかのエラーがありました。",
                    code="bad_request",
                    details=error_str,
                )
            if status_code == 401:
                return InvalidRequestError(
                    message="APIキーを確認してください。",
                    code="authentication_error",
                    details=error_str,
                )
            if status_code == 403:
                return InvalidRequestError(
                    message="APIキーの権限を確認してください。",
                    code="permission_denied",
                    details=error_str,
                )
            if status_code == 404:
                return InvalidRequestError(
                    message="リソースが見つかりません。",
                    code="not_found",
                    details=error_str,
                )
            if status_code == 429:
                return RateLimitError(details=error_str)
            if status_code >= 500:
                return APIError(
                    message="サーバーエラーが発生しました。",
                    code="internal_server_error",
                    status_code=status_code,
                    details=error_str,
                )
        return APIError(
            message="APIエラーが発生しました。", code="api_error", details=error_str
        )

    exception_type = str(type(e)).lower()

    # Azure例外の処理
    if (
        "azure" in exception_type
        or "DeploymentNotFound" in error_str
        or "content_filter_policy" in error_str
    ):
        if "This model's maximum context length is" in error_str:
            return ContextWindowExceededError(details=error_str)
        if "DeploymentNotFound" in error_str:
            return InvalidRequestError(
                message="指定されたモデルまたはデプロイメントが見つかりません。",
                code="model_not_found",
                details=error_str,
            )
        if (
            "content_filter_policy" in error_str
            or "content management" in error_str
            or "safety system" in error_str
        ):
            return ContentPolicyViolationError(details=error_str)
        if "invalid_request_error" in error_str:
            return InvalidRequestError(
                message="リクエスト内容に何らかのエラーがありました。",
                code="invalid_request_error",
                details=error_str,
            )
        if hasattr(e, "status_code"):
            if e.status_code == 401:
                return InvalidRequestError(
                    message="APIキーまたは認証情報を確認してください。",
                    code="authentication_error",
                    details=error_str,
                )
            if e.status_code == 429:
                return RateLimitError(details=error_str)
            if e.status_code >= 500:
                return APIError(
                    message="内部サーバーエラーが発生しました。",
                    code="internal_server_error",
                    status_code=500,
                    details=error_str,
                )
        return APIError(
            message="APIエラーが発生しました。", code="api_error", details=error_str
        )

    # AWS例外の処理
    if (
        "boto" in exception_type
        or "bedrock" in exception_type
        or "ValidationException" in error_str
        or "throttlingException" in error_str
        or "AccessDeniedException" in error_str
    ):
        if (
            "too many tokens" in error_str
            or "Input is too long" in error_str
            or "prompt is too long" in error_str
        ):
            return ContextWindowExceededError(details=error_str)
        if "AccessDeniedException" in error_str:
            return InvalidRequestError(
                message="IAMロールまたは権限を確認してください。",
                code="permission_denied",
                details=error_str,
            )
        if "throttlingException" in error_str or "ThrottlingException" in error_str:
            return RateLimitError(details=error_str)
        if "ValidationException" in error_str:
            return InvalidRequestError(
                message="リクエスト内容に検証エラーがありました。",
                code="validation_error",
                details=error_str,
            )
        if "Unable to locate credentials" in error_str:
            return InvalidRequestError(
                message="AWS認証情報が見つかりません。AWS認証設定を確認してください。",
                code="authentication_error",
                details=error_str,
            )
        if hasattr(e, "status_code") or hasattr(e, "response"):
            response = getattr(e, "response", {})
            status_code = getattr(
                e,
                "status_code",
                response.get("ResponseMetadata", {}).get("HTTPStatusCode"),
            )
            if status_code == 500:
                return APIError(
                    message="内部サーバーエラーが発生しました。",
                    code="internal_server_error",
                    status_code=500,
                    details=error_str,
                )
        return APIError(
            message="APIエラーが発生しました。", code="api_error", details=error_str
        )

    # VertexAI例外の処理
    if (
        "google" in exception_type
        or "vertex" in exception_type
        or "genai" in exception_type
        or "Quota exceeded for" in error_str
        or "The response was blocked" in error_str
    ):
        if "Quota exceeded for" in error_str or "out of capacity" in error_str:
            return RateLimitError(details=error_str)
        if (
            "The response was blocked" in error_str
            or "content filtering policy" in error_str
        ):
            return ContentPolicyViolationError(details=error_str)
        if "API key not valid" in error_str:
            return InvalidRequestError(
                message="APIキーまたはサービスアカウントを確認してください。",
                code="authentication_error",
                details=error_str,
            )
        if "Unable to find your project" in error_str:
            return InvalidRequestError(
                message="Google Cloudプロジェクトが見つかりません。プロジェクトIDを確認してください。",
                code="project_not_found",
                details=error_str,
            )
        if "400 Request payload size exceeds" in error_str:
            return ContextWindowExceededError(details=error_str)
        if hasattr(e, "status_code"):
            if e.status_code == 403:
                return InvalidRequestError(
                    message="権限またはAPI有効化を確認してください。",
                    code="permission_denied",
                    details=error_str,
                )
            if e.status_code >= 500:
                return APIError(
                    message="内部サーバーエラーが発生しました。",
                    code="internal_server_error",
                    status_code=500,
                    details=error_str,
                )
        return APIError(
            message="APIエラーが発生しました。", code="api_error", details=error_str
        )

    # HTTP例外の汎用処理
    if hasattr(e, "status_code"):
        status_code = e.status_code
        if status_code == 400:
            return InvalidRequestError(
                message="リクエスト内容に何らかのエラーがありました。",
                code="bad_request",
                details=error_str,
            )
        if status_code == 401:
            return InvalidRequestError(
                message="認証エラーが発生しました。",
                code="authentication_error",
                details=error_str,
            )
        if status_code == 403:
            return InvalidRequestError(
                message="権限エラーが発生しました。",
                code="permission_denied",
                details=error_str,
            )
        if status_code == 404:
            return InvalidRequestError(
                message="リソースが見つかりません。",
                code="not_found",
                details=error_str,
            )
        if status_code == 429:
            return RateLimitError(details=error_str)
        if status_code >= 500:
            return APIError(
                message="サーバーエラーが発生しました。",
                code="internal_server_error",
                status_code=status_code,
                details=error_str,
            )
        return APIError(
            message="HTTPエラーが発生しました。",
            code="api_error",
            status_code=status_code,
            details=error_str,
        )

    # その他の例外は汎用APIエラーとして処理
    return APIError(
        message="予期しないエラーが発生しました。",
        code="internal_server_error",
        status_code=500,
        details=error_str,
    )
