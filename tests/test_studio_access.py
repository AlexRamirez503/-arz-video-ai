"""CPU-only regression tests for private Studio access in Safari-like browsers."""
import ast
import hmac
import tempfile
import unittest
from pathlib import Path
from typing import Optional
from types import SimpleNamespace
from unittest.mock import Mock
import time
import uuid

from fastapi import Cookie, HTTPException, Query
from fastapi.responses import HTMLResponse


ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "server" / "main.py"
STUDIO_PATH = ROOT / "server" / "studio.html"


class StudioAccessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.html = Path(self.temp.name) / "studio.html"
        self.html.write_text("<html>studio</html>", encoding="utf-8")
        module = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
        fn = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "studio"
        )
        fn.decorator_list = []
        self.ns = {
            "Optional": Optional,
            "hmac": hmac,
            "HTMLResponse": HTMLResponse,
            "HTTPException": HTTPException,
            "Query": Query,
            "Cookie": Cookie,
            "STUDIO_ACCESS_KEY": "private-link-key",
            "STUDIO_SESSION_COOKIE": "aztv_studio_session",
            "STUDIO_SESSION_MAX_AGE_SECONDS": 60,
            "STUDIO_HTML_PATH": self.html,
            "has_studio_access": lambda value: bool(
                value and hmac.compare_digest(value, "private-link-key")
            ),
        }
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "main.py", "exec"), self.ns)

    def test_valid_private_link_serves_studio_without_redirect(self):
        response = self.ns["studio"](access_key="private-link-key", studio_session=None)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.body, b"<html>studio</html>")
        self.assertIn("aztv_studio_session=private-link-key", response.headers["set-cookie"])

    def test_cookie_session_still_opens_studio(self):
        response = self.ns["studio"](access_key=None, studio_session="private-link-key")
        self.assertEqual(response.status_code, 200)

    def test_missing_private_access_is_rejected(self):
        with self.assertRaises(HTTPException) as raised:
            self.ns["studio"](access_key=None, studio_session=None)
        self.assertEqual(raised.exception.status_code, 401)

    def test_page_keeps_access_for_api_calls_then_hides_url(self):
        page = STUDIO_PATH.read_text(encoding="utf-8")
        self.assertIn("sessionStorage.setItem('aztv_studio_access',privateAccess)", page)
        self.assertIn("history.replaceState(null,document.title,window.location.pathname)", page)
        self.assertIn("h['X-Studio-Access']=privateAccess", page)


class StudioScriptTests(unittest.TestCase):
    def test_studio_queues_the_exact_user_script_with_male_voice(self):
        module = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
        fn = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "studio_request"
        )
        fn.decorator_list = []
        job_queue = Mock()
        ns = {
            "StudioRequest": object,
            "uuid": uuid,
            "time": time,
            "FAST_PROMO_AVATAR_ID": "promo3d",
            "set_job": Mock(),
            "job_queue": job_queue,
        }
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "main.py", "exec"), ns)
        script = "Hoy tienes una oferta especial. Descarga AZTV y disfruta tus favoritos."
        request = SimpleNamespace(
            message=script,
            avatar_url=None,
            brand_text="AZTV",
            cta_text="Descárgala hoy",
            voice="male",
            voice_speed="fast",
            visual_mode="cinematic",
        )

        ns["studio_request"](request)
        _, payload = job_queue.put.call_args.args[0]
        self.assertEqual(payload["text"], script)
        self.assertEqual(payload["voice"], "male")
        self.assertEqual(payload["visual_mode"], "cinematic")


if __name__ == "__main__":
    unittest.main()
