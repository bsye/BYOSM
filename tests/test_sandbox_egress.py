import socket
import socketserver
import threading

import pytest

from byosm.sandbox import egress
from byosm.sandbox.egress import FORBIDDEN, OPENED, Egress, target

LOOPBACK = "127.0.0.1"


class Echo(socketserver.BaseRequestHandler):
    def handle(self):
        while data := self.request.recv(1024):
            self.request.sendall(data)


@pytest.fixture
def servers(monkeypatch):
    monkeypatch.setattr(egress, "HOST_ALIAS", LOOPBACK)
    upstream = socketserver.ThreadingTCPServer((LOOPBACK, 0), Echo)
    upstream.daemon_threads = True
    port = upstream.server_address[1]
    proxy = Egress((LOOPBACK, 0), [f"localhost:{port}"])
    for server in (upstream, proxy):
        threading.Thread(target=server.serve_forever, daemon=True).start()
    yield proxy.server_address[1], port
    for server in (upstream, proxy):
        server.shutdown()
        server.server_close()


def ask(proxy, request, read):
    with socket.create_connection((LOOPBACK, proxy), timeout=5) as conn:
        conn.sendall(request)
        data = b""
        while len(data) < read and (chunk := conn.recv(1024)):
            data += chunk
        return data


def test_requests_name_their_host_and_port():
    assert target("CONNECT huggingface.co:443 HTTP/1.1") == ("huggingface.co", 443)
    assert target("CONNECT [::1]:8080 HTTP/1.1") == ("::1", 8080)
    assert target("POST http://localhost:11434/v1/chat HTTP/1.1") == ("localhost", 11434)
    assert target("GET http://example.com/ HTTP/1.1") == ("example.com", 80)


def test_an_allowed_host_is_tunnelled_and_localhost_means_the_machine(servers):
    proxy, port = servers
    opened = f"CONNECT localhost:{port} HTTP/1.1\r\n\r\n".encode()
    with socket.create_connection((LOOPBACK, proxy), timeout=5) as conn:
        conn.sendall(opened)
        assert conn.recv(len(OPENED)) == OPENED
        conn.sendall(b"ping")
        assert conn.recv(4) == b"ping"


def test_a_plain_http_request_reaches_the_allowed_host_with_its_body(servers):
    proxy, port = servers
    request = f"POST http://localhost:{port}/v1 HTTP/1.1\r\nContent-Length: 4\r\n\r\nbody".encode()
    assert ask(proxy, request, len(request)) == request


def test_any_other_host_is_refused_and_logged(servers, capsys):
    proxy, port = servers
    assert ask(proxy, b"CONNECT example.com:443 HTTP/1.1\r\n\r\n", len(FORBIDDEN)) == FORBIDDEN
    assert ask(proxy, f"CONNECT localhost:{port + 1} HTTP/1.1\r\n\r\n".encode(), len(FORBIDDEN)) == FORBIDDEN
    assert "refused example.com:443" in capsys.readouterr().err
