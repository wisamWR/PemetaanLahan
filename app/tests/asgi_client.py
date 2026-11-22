"""Klien ASGI minimal untuk tes API (BUKAN bagian aplikasi).

`starlette.testclient.TestClient` butuh `httpx`, yang tidak termasuk stack yang diizinkan; klien ini
memanggil aplikasi ASGI langsung, tanpa jaringan. Mendukung badan JSON, query, dan aliran penuh
(SSE dibaca sampai server menutup respons).
"""

from __future__ import annotations

import asyncio
import json as jsonlib
import time
from dataclasses import dataclass
from urllib.parse import unquote, urlencode


@dataclass
class Response:
    status: int
    headers: dict[str, str]
    body: bytes

    @property
    def text(self) -> str:
        return self.body.decode("utf-8")

    def json(self):
        return jsonlib.loads(self.body)

    def events(self) -> list[tuple[str, dict | None]]:
        """Bingkai SSE (event, data) dari badan respons; komentar dan `retry` dilewati."""
        out: list[tuple[str, dict | None]] = []
        for block in self.text.split("\n\n"):
            event, data = None, None
            for line in block.splitlines():
                if line.startswith("event: "):
                    event = line[len("event: "):]
                elif line.startswith("data: "):
                    data = jsonlib.loads(line[len("data: "):])
            if event is not None:
                out.append((event, data))
        return out


async def _call(app, method: str, path: str, query: str, headers: list, body: bytes) -> Response:
    raw_path, _, _ = path.partition("?")
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method,
        "scheme": "http", "path": unquote(raw_path), "raw_path": raw_path.encode(), "query_string": query.encode(),
        "root_path": "", "headers": headers, "client": ("testclient", 50000), "server": ("testserver", 80),
    }
    request_sent = False
    never = asyncio.Event()

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await never.wait()                      # respons berjalan; tak ada pemutusan dari klien

    status, response_headers, chunks = 0, {}, []

    async def send(message):
        nonlocal status, response_headers
        if message["type"] == "http.response.start":
            status = message["status"]
            response_headers = {k.decode().lower(): v.decode() for k, v in message["headers"]}
        elif message["type"] == "http.response.body":
            chunks.append(message.get("body", b""))

    await app(scope, receive, send)
    return Response(status, response_headers, b"".join(chunks))


def request(
    app, method: str, path: str, *, params: dict | None = None, json=None, headers: dict | None = None,
    body: bytes | None = None, content_type: str = "application/octet-stream",
) -> Response:
    raw_headers = [(b"host", b"testserver")]
    if body is not None:                        # badan mentah (unggahan PNG)
        raw_headers += [(b"content-type", content_type.encode()), (b"content-length", str(len(body)).encode())]
    else:
        body = b""
    if json is not None:
        body = jsonlib.dumps(json).encode()
        raw_headers += [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
    for key, value in (headers or {}).items():
        raw_headers.append((key.lower().encode(), value.encode()))
    query = urlencode({k: v for k, v in (params or {}).items() if v is not None})
    return asyncio.run(_call(app, method.upper(), path, query, raw_headers, body))


def get(app, path: str, **params) -> Response:
    return request(app, "GET", path, params=params)


def post(
    app, path: str, json=None, *, params: dict | None = None, body: bytes | None = None,
    content_type: str = "application/octet-stream",
) -> Response:
    return request(app, "POST", path, json=json, params=params, body=body, content_type=content_type)


def wait_job(app, job_id: str, timeout: float = 60.0) -> dict:
    """Tunggu pekerjaan selesai (done/error); kembalikan cuplikan terakhirnya."""
    deadline = time.monotonic() + timeout
    while True:
        snap = get(app, f"/api/jobs/{job_id}").json()
        if snap["status"] in ("done", "error"):
            return snap
        if time.monotonic() > deadline:
            raise TimeoutError(f"pekerjaan {job_id} belum selesai setelah {timeout} detik: {snap}")
        time.sleep(0.03)
