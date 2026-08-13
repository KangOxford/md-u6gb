from __future__ import annotations

import contextlib
import getpass
import importlib.machinery
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import types
import unittest
from unittest import mock


SCRIPT = Path("/projects/public/u6gb/.local/bin/sgpur")


def load_sgpur() -> types.ModuleType:
    name = f"sgpur_web_refresh_test_{os.getpid()}_{id(object())}"
    loader = importlib.machinery.SourceFileLoader(name, str(SCRIPT))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


class FakeResponse:
    def __init__(self, body: str, status: int = 200):
        self.body = body.encode()
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return self.body


def patch_urlopen(body: str = "0"):
    return mock.patch("urllib.request.urlopen", return_value=FakeResponse(body))


class WebRefreshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_sgpur()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.old_inbox_path = self.module.INBOX_PATH
        self.module.INBOX_PATH = str(Path(self.tmp.name) / "sgpur" / "inbox.json")
        self.addCleanup(setattr, self.module, "INBOX_PATH", self.old_inbox_path)

    def write_config(self, data: dict, mode: int = 0o600) -> Path:
        path = Path(self.module.INBOX_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data) + "\n")
        path.chmod(mode)
        return path

    def test_browser_post_uses_header_not_query(self):
        source = SCRIPT.read_text()
        self.assertIn("'x-token':tok", source)
        self.assertNotIn("?t=", source)
        self.assertNotIn("mode:'no-cors'", source)

    def test_set_inbox_preserves_existing_token_and_never_prints_it(self):
        token = "sentinel-secret"
        path = self.write_config({"url": "https://old.example", "token": token})
        output = io.StringIO()
        with patch_urlopen("0"), contextlib.redirect_stdout(output):
            rc = self.module.cmd_set_inbox("https://new.example")
        self.assertEqual(0, rc)
        self.assertEqual(
            {"url": "https://new.example", "token": token},
            json.loads(path.read_text()),
        )
        self.assertNotIn(token, output.getvalue())
        self.assertEqual(0o600, stat.S_IMODE(path.stat().st_mode))

    def test_set_inbox_prompts_for_missing_token_without_echo(self):
        output = io.StringIO()
        with mock.patch.object(getpass, "getpass", return_value="prompted-secret"), \
             patch_urlopen("0"), contextlib.redirect_stdout(output):
            rc = self.module.cmd_set_inbox("https://new.example")
        self.assertEqual(0, rc)
        path = Path(self.module.INBOX_PATH)
        self.assertEqual(
            "prompted-secret", json.loads(path.read_text())["token"]
        )
        self.assertNotIn("prompted-secret", output.getvalue())
        self.assertEqual(0o600, stat.S_IMODE(path.stat().st_mode))

    def test_inbox_get_rejects_malformed_counter(self):
        with patch_urlopen("not-a-timestamp"):
            value = self.module.inbox_get(
                {"url": "https://worker.example", "token": "s"}
            )
        self.assertIsNone(value)

    def test_inbox_get_network_error_is_unavailable(self):
        with mock.patch("urllib.request.urlopen", side_effect=OSError("offline")):
            value = self.module.inbox_get(
                {"url": "https://worker.example", "token": "s"}
            )
        self.assertIsNone(value)

    def test_inbox_get_uses_cloudflare_compatible_user_agent(self):
        seen = {}

        def open_request(request, timeout):
            seen["user_agent"] = request.get_header("User-agent")
            return FakeResponse("0")

        with mock.patch("urllib.request.urlopen", side_effect=open_request):
            value = self.module.inbox_get(
                {"url": "https://worker.example", "token": "s"}
            )
        self.assertEqual("0", value)
        self.assertIn("sgpur/", seen["user_agent"])

    def test_read_inbox_rejects_invalid_shape(self):
        self.write_config({"url": "http://worker.example", "token": "s"})
        self.assertEqual({}, self.module.read_inbox())
        self.write_config({"url": "https://worker.example", "token": ""})
        self.assertEqual({}, self.module.read_inbox())


if __name__ == "__main__":
    unittest.main()
