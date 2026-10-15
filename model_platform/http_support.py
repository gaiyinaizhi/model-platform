"""Small bounded local HTTP transport, not a production internet gateway."""
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .common import MAX_BODY, Problem, encode


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


class JsonHandler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(3)

    def log_message(self, *args):
        pass

    def body(self):
        if self.headers.get("Transfer-Encoding"):
            raise Problem(400, "invalid_input", "Chunked requests are unsupported")
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1:
            raise Problem(400, "invalid_input", "One Content-Length required")
        try:
            size = int(lengths[0])
        except ValueError:
            raise Problem(400, "invalid_input", "Invalid Content-Length") from None
        if not 0 < size <= MAX_BODY:
            raise Problem(413, "body_limit", "Request body exceeds limit or is empty")
        if self.headers.get_content_type() != "application/json":
            raise Problem(400, "invalid_input", "application/json required")
        try:
            raw = self.rfile.read(size)
            if len(raw) != size:
                raise ValueError("Truncated body")
            return json.loads(raw, object_pairs_hook=unique_pairs,
                              parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
        except (ValueError, UnicodeDecodeError, socket.timeout, RecursionError):
            raise Problem(400, "invalid_json", "Invalid JSON body") from None

    def respond(self, status, body):
        content = encode(body)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(content)

    def handle_problem(self, exc):
        self.respond(exc.status, {"error": {"code": exc.code, "message": exc.message}})


class BoundedHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 16
    allow_reuse_address = True

    def __init__(self, address, handler):
        self.slots = threading.BoundedSemaphore(16)
        super().__init__(address, handler)

    def process_request(self, request, address):
        if not self.slots.acquire(blocking=False):
            try:
                request.sendall(b'HTTP/1.0 429 Too Many Requests\r\nContent-Length: 0\r\nConnection: close\r\n\r\n')
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, address)
        except Exception:
            self.slots.release()
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self.slots.release()
