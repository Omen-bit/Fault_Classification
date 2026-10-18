"""Shared test setup: the whole suite runs with the network blocked."""
import os
import socket
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


LOOPBACK = ("127.0.0.1", "::1", "localhost")


@pytest.fixture(autouse=True, scope="session")
def no_network():
    """Any connection that leaves this machine raises. Loopback stays open because the
    event loop on Windows uses a local socket pair internally."""
    real_connect, real_create = socket.socket.connect, socket.create_connection

    def guarded_connect(self, address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else address
        if host not in LOOPBACK:
            raise RuntimeError("network blocked in tests")
        return real_connect(self, address, *args, **kwargs)

    def guarded_create(address, *args, **kwargs):
        if address[0] not in LOOPBACK:
            raise RuntimeError("network blocked in tests")
        return real_create(address, *args, **kwargs)

    socket.socket.connect = guarded_connect
    socket.create_connection = guarded_create
    yield
    socket.socket.connect, socket.create_connection = real_connect, real_create


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    from app import app
    with TestClient(app) as c:
        yield c
