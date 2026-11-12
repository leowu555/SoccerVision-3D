"""Request body size limit, enforced while the body streams in (not after it is spooled)."""

import contextlib
import json

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from soccervision_api.errors import error_body


class BodySizeLimitMiddleware:
    def __init__(self, app: ASGIApp, *, max_bytes: int, path_prefixes: tuple[str, ...]) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.path_prefixes = path_prefixes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or not str(scope.get("path", "")).startswith(self.path_prefixes)
        ):
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        declared = headers.get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
            await self._reject(send)
            return

        received = 0
        exceeded = False
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received, exceeded
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    exceeded = True
                    raise _BodyTooLargeError
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal response_started
            if exceeded:
                return
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        with contextlib.suppress(_BodyTooLargeError):
            await self.app(scope, limited_receive, guarded_send)
        if exceeded and not response_started:
            await self._reject(send)

    async def _reject(self, send: Send) -> None:
        mib = self.max_bytes / (1024 * 1024)
        body = json.dumps(
            error_body("upload_too_large", f"Request body exceeds the {mib:.0f} MiB upload limit.")
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


class _BodyTooLargeError(Exception):
    pass
