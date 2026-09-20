"""CPU-only checks for the unrestricted local Wan video path."""
import ast
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "server" / "main.py"


class FreeVideoPromptTests(unittest.TestCase):
    def test_prompt_keeps_arbitrary_requested_scene(self):
        module = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
        fn = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "build_free_video_prompt"
        )
        ns = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "main.py", "exec"), ns)
        requested = "Una casa cantando bajo la lluvia con luces de neón"
        prompt = ns["build_free_video_prompt"](requested, "cartoon")
        self.assertIn(requested, prompt)
        self.assertIn("animated cartoon video", prompt)
        self.assertNotIn("presenter", prompt.lower())


class FreeVideoWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.jobs_dir = self.root / "jobs"
        self.jobs_dir.mkdir()
        self.states = {}
        self.generated = self.root / "wan.mp4"
        self.generated.write_bytes(b"generated clip")
        module = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
        fn = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_free_video_job"
        )
        self.run_wan = Mock(return_value=self.generated)
        self.ns = {
            "Path": Path,
            "time": time,
            "shutil": shutil,
            "JOBS_DIR": self.jobs_dir,
            "set_job": lambda job_id, **changes: self.states.update(changes),
            "build_free_video_prompt": lambda prompt, style: f"{style}: {prompt}",
            "run_wan_video": self.run_wan,
            "cleanup_outputs": Mock(),
        }
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "main.py", "exec"), self.ns)

    def test_arbitrary_prompt_generates_without_avatar_or_voice(self):
        payload = {
            "prompt": "Un niño en bicicleta por un parque soleado",
            "free_style": "cinematic",
            "orientation": "vertical",
            "steps": 24,
            "seed": -1,
        }
        self.ns["run_free_video_job"]("freevideo123", payload)
        self.run_wan.assert_called_once()
        kwargs = self.run_wan.call_args.kwargs
        self.assertEqual(kwargs["prompt"], "cinematic: Un niño en bicicleta por un parque soleado")
        self.assertEqual(kwargs["orientation"], "vertical")
        self.assertEqual(self.states["status"], "done")
        self.assertTrue((self.jobs_dir / "freevideo123.mp4").is_file())

    def test_generation_error_does_not_create_fake_video(self):
        self.run_wan.side_effect = RuntimeError("GPU error")
        payload = {
            "prompt": "Dibujos animados coloridos",
            "free_style": "cartoon",
            "orientation": "landscape",
            "steps": 24,
            "seed": -1,
        }
        self.ns["run_free_video_job"]("freevideo123", payload)
        self.assertEqual(self.states["status"], "error")
        self.assertFalse((self.jobs_dir / "freevideo123.mp4").exists())


if __name__ == "__main__":
    unittest.main()
