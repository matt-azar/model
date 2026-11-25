"""Serve a checkpoint-backed chat window on the local machine."""

from __future__ import annotations

import argparse
import json
import traceback
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

from src.chat import resolve_best_ckpt
from src.generate import Generator
from src.template import format_chat_messages

ASSETS = Path(__file__).with_name("gui_assets")
ASSET_ROUTES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/chat.js": ("chat.js", "text/javascript; charset=utf-8"),
}
MAX_BODY_BYTES = 1024 * 1024
MAX_MESSAGES = 21
MAX_PROMPT_CHARACTERS = 8000
MAX_CONTEXT_TOKENS = 4096
CONTENT_POLICY = "default-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"


def parse_messages(body: object) -> list[dict[str, str]]:
    raw = body.get("messages") if isinstance(body, dict) else None
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_MESSAGES or len(raw) % 2 == 0:
        raise ValueError("Send a conversation ending with your question.")

    messages = []
    for index, item in enumerate(raw):
        role = "assistant" if index % 2 else "user"
        if not isinstance(item, dict) or item.get("role") != role:
            raise ValueError("Conversation turns must alternate between you and the model.")
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Each turn needs a message.")
        messages.append({"role": role, "content": content})

    if len(messages[-1]["content"]) > MAX_PROMPT_CHARACTERS:
        raise ValueError(f"Keep the latest message to {MAX_PROMPT_CHARACTERS:,} characters.")
    return messages


class ChatSession:
    def __init__(self, generator: Generator, checkpoint: str, max_new_tokens: int) -> None:
        self.generator = generator
        self.checkpoint = checkpoint
        self.max_new_tokens = max_new_tokens
        self.lock = Lock()

    def describe(self) -> dict:
        return {"checkpoint": self.checkpoint, "device": str(self.generator.device)}

    def reply(self, messages: list[dict[str, str]]) -> dict:
        tokenizer = self.generator.tokenizer
        context = min(self.generator.model.cfg.block_size, MAX_CONTEXT_TOKENS)
        for start in range(0, len(messages), 2):
            prompt = format_chat_messages(messages[start:], tokenizer)
            space = context - len(tokenizer.encode(prompt, add_bos=True, add_eos=False))
            if space >= self.max_new_tokens or start == len(messages) - 1:
                break

        if space < 1:
            raise ValueError("The latest message is too long for the model. Shorten it and try again.")
        reply = self.generator.generate(
            [prompt], max_new_tokens=min(space, self.max_new_tokens), temperature=0.2
        )[0]
        if not reply.text.strip():
            raise ValueError("The model returned an empty reply. Try rephrasing the question.")
        return {
            "text": reply.text,
            "finish_reason": reply.finish_reason,
            "omitted_turns": start // 2,
        }


class ChatHandler(BaseHTTPRequestHandler):
    server: ChatServer

    def _send(self, status: HTTPStatus, payload: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", CONTENT_POLICY)
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, status: HTTPStatus, payload: dict) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def _local_host(self) -> bool:
        return self.headers.get("Host") in {
            f"127.0.0.1:{self.server.server_port}",
            f"localhost:{self.server.server_port}",
        }

    def do_GET(self) -> None:
        if not self._local_host():
            self._json(HTTPStatus.FORBIDDEN, {"error": "Open the local address printed in the terminal."})
            return
        if self.path == "/api/model":
            self._json(HTTPStatus.OK, self.server.session.describe())
            return

        asset = ASSET_ROUTES.get(self.path)
        if asset is None:
            self._json(HTTPStatus.NOT_FOUND, {"error": "Page not found."})
            return
        name, content_type = asset
        self._send(HTTPStatus.OK, (ASSETS / name).read_bytes(), content_type)

    def do_POST(self) -> None:
        origin = f"http://{self.headers.get('Host')}"
        if not self._local_host() or self.headers.get("Origin") != origin:
            self._json(HTTPStatus.FORBIDDEN, {"error": "Send questions from the local chat window."})
            return
        if self.path != "/api/chat":
            self._json(HTTPStatus.NOT_FOUND, {"error": "Page not found."})
            return
        if self.headers.get_content_type() != "application/json":
            self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "Expected a JSON request."})
            return

        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            size = 0
        if not 0 < size <= MAX_BODY_BYTES:
            self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "The conversation request is too large or empty."})
            return

        try:
            body = json.loads(self.rfile.read(size).decode("utf-8"))
        except (UnicodeError, ValueError, RecursionError):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "Invalid conversation request."})
            return

        session = self.server.session
        if not session.lock.acquire(blocking=False):
            self._json(HTTPStatus.CONFLICT, {"error": "The model is already answering another question."})
            return
        try:
            messages = parse_messages(body)
            reply = session.reply(messages)
        except ValueError as error:
            status, reply = HTTPStatus.BAD_REQUEST, {"error": str(error)}
        except Exception:
            traceback.print_exc()
            status, reply = HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Generation failed. See the terminal for details."}
        else:
            status = HTTPStatus.OK
        finally:
            session.lock.release()
        self._json(status, reply)

    def log_message(self, format: str, *args: object) -> None:
        # Keep progress and model errors visible in the launcher terminal.
        pass


class ChatServer(ThreadingHTTPServer):
    def __init__(self, port: int, session: ChatSession) -> None:
        super().__init__(("127.0.0.1", port), ChatHandler)
        self.session = session


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", help="trained export directory or checkpoint")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--no-browser", action="store_true", help="print the local URL without opening it")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535 or args.max_new_tokens < 1:
        parser.error("choose a valid port and a positive token limit")

    ckpt = resolve_best_ckpt(args.ckpt)
    print(f"Loading {ckpt}…", flush=True)
    generator = Generator.from_checkpoint(ckpt, args.device)
    session = ChatSession(generator, ckpt, args.max_new_tokens)
    try:
        server = ChatServer(args.port, session)
    except OSError as error:
        parser.exit(1, f"Could not open local port {args.port}: {error}. Try --port 7861.\n")

    with server:
        url = f"http://127.0.0.1:{server.server_port}"
        print(f"Chat ready at {url} ({generator.device}). Press Ctrl+C to close.", flush=True)
        if not args.no_browser:
            try:
                webbrowser.open(url)
            except webbrowser.Error:
                pass
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
