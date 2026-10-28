from __future__ import annotations

import contextlib
import selectors
import socket
import socketserver
import sys
from urllib.parse import urlsplit

from ..llm import LOCAL_HOSTS
from .allow import SCHEME_PORTS, allowed

HOST_ALIAS = "host.docker.internal"
REFUSED = "refused"
CHUNK = 65536
HEAD_LIMIT = 65536
END = b"\r\n\r\n"
OPENED = b"HTTP/1.1 200 Connection established\r\n\r\n"
FORBIDDEN = b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
UNREACHABLE = b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
LOOPBACK = tuple(host.strip("[]") for host in LOCAL_HOSTS)


def target(line: str) -> tuple[str, int]:
    method, _, rest = line.partition(" ")
    where = rest.partition(" ")[0]
    if method == "CONNECT":
        host, _, port = where.rpartition(":")
        return host.strip("[]"), int(port)
    url = urlsplit(where)
    return url.hostname or "", url.port or SCHEME_PORTS["http"]


def _head(conn: socket.socket) -> bytes:
    data = b""
    while END not in data and len(data) < HEAD_LIMIT:
        chunk = conn.recv(CHUNK)
        if not chunk:
            break
        data += chunk
    return data


def _relay(client: socket.socket, upstream: socket.socket) -> None:
    with selectors.DefaultSelector() as sel:
        sel.register(client, selectors.EVENT_READ, upstream)
        sel.register(upstream, selectors.EVENT_READ, client)
        while True:
            for key, _ in sel.select():
                data = key.fileobj.recv(CHUNK)
                if not data:
                    return
                key.data.sendall(data)


class Tunnel(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        head = _head(self.request)
        line = head.partition(b"\r\n")[0].decode("latin-1")
        try:
            host, port = target(line)
        except ValueError:
            host, port = "", 0
        if not allowed(host, port, self.server.rules):
            print(f"{REFUSED} {host}:{port}", file=sys.stderr, flush=True)
            self.request.sendall(FORBIDDEN)
            return
        try:
            upstream = socket.create_connection((HOST_ALIAS if host in LOOPBACK else host, port))
        except OSError:
            self.request.sendall(UNREACHABLE)
            return
        with upstream, contextlib.suppress(OSError):
            if line.startswith("CONNECT "):
                self.request.sendall(OPENED)
                upstream.sendall(head.partition(END)[2])
            else:
                upstream.sendall(head)
            _relay(self.request, upstream)


class Egress(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], rules: list[str]) -> None:
        super().__init__(address, Tunnel)
        self.rules = rules


if __name__ == "__main__":
    with Egress(("", int(sys.argv[1])), sys.argv[2:]) as server:
        server.serve_forever()
