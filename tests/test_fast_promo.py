"""CPU-only checks for the fast avatar promotion path."""
import ast
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "server" / "main.py"


class FastPromoWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.job_id = "fastpromo12345678"
        self.base = self.root / "avatars" / "promo3d"
        self.events = []
        self.states = {}

        def synthesize(text, wav, length_scale, voice_model=None):
            self.events.append(("speech", length_scale, text))
            Path(wav).write_bytes(b"voice")

        def inference(**kwargs):
            self.events.append("lipsync")
            output = self.base / "vid_output" / f"{self.job_id}.mp4"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"animated avatar with lipsync")

        def branding(source, output, **kwargs):
            self.events.append("branding")
            shutil.copy2(source, output)

        self.ns = {
            "Path": Path,
            "shutil": shutil,
            "time": time,
            "JOBS_DIR": self.root,
            "FPS": 25,
            "FAST_VOICE_LENGTH_SCALE": 0.84,
            "NORMAL_VOICE_LENGTH_SCALE": 1.0,
            "MALE_VOICE_MODEL": self.root / "male.onnx",
            "STUDIO_VOICE_MODELS": {"male": self.root / "male.onnx"},
            "studio_voice_text": lambda text: text,
            "set_job": lambda job, **changes: self.states.update(changes),
            "init_engine": lambda: self.events.append("engine"),
            "ensure_promo_avatar": lambda payload: self._avatar,
            "synthesize_speech": synthesize,
            "avatar_base": lambda avatar_id: self.base,
            "apply_branding": branding,
            "cleanup_outputs": lambda: self.events.append("cleanup"),
        }
        self._avatar = Mock(inference=inference)
        (self.root / "male.onnx").write_bytes(b"voice")
        module = ast.parse(MAIN_PATH.read_text())
        fn = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_promo_job"
        )
        self.assertNotIn("run_wan_video", ast.unparse(fn))
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "main.py", "exec"), self.ns)

    def test_fast_path_uses_avatar_and_local_voice_without_wan(self):
        payload = {
            "avatar_id": "promo3d",
            "text": "Descarga AZTV hoy.",
            "brand_text": "AZTV",
            "cta_text": "Descárgala hoy",
            "voice": "male",
            "voice_speed": "fast",
        }
        self.ns["run_promo_job"](self.job_id, payload)
        self.assertEqual(self.events[0], "engine")
        self.assertEqual(self.events[1], ("speech", 0.84, "Descarga AZTV hoy."))
        self.assertEqual(self.events[2:], ["lipsync", "branding", "cleanup"])
        self.assertEqual(self.states["status"], "done")
        self.assertEqual(self.states["stage"], "done")
        self.assertTrue((self.root / f"{self.job_id}.mp4").is_file())
        self.assertFalse((self.root / f"{self.job_id}.wav").exists())
        self.assertFalse((self.root / f"{self.job_id}-lipsync.mp4").exists())

    def test_normal_voice_speed_uses_normal_length_scale(self):
        payload = {
            "avatar_id": "promo3d",
            "text": "Hola.",
            "brand_text": "AZTV",
            "cta_text": "",
            "voice": "male",
            "voice_speed": "normal",
        }
        self.ns["run_promo_job"](self.job_id, payload)
        self.assertEqual(self.events[1], ("speech", 1.0, "Hola."))


if __name__ == "__main__":
    unittest.main()
