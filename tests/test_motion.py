"""CPU-only regression checks for orchestration, not model-quality validation."""
import ast
import io
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))
from motion import save_upload


class UploadTests(unittest.TestCase):
    def test_rejects_oversized_and_empty_media(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "upload"
            for data in (b"", b"12345"):
                with self.assertRaises(ValueError):
                    save_upload(io.BytesIO(data), path, 4)

    def test_preserves_uploaded_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "upload"
            save_upload(io.BytesIO(b"1234"), path, 4)
            self.assertEqual(path.read_bytes(), b"1234")


class MotionFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.work = self.root / "work"
        self.work.mkdir()
        self.job_id = "abcdef1234567890"
        self.avatar_id = "motion-" + self.job_id
        self.base = self.root / self.avatar_id
        self.events = []
        self.states = {}
        self.moving = self.work / "moving-avatar.mp4"

        def motion(*args):
            self.events.append("movement")
            self.moving.write_bytes(b"moving frames")
            return self.moving

        def prepare(avatar_id, source):
            self.events.append("prepare")
            self.assertEqual(source, self.moving)
            self.assertEqual(avatar_id, self.avatar_id)
            self.assertTrue(source.is_file())
            return Mock(inference=inference)

        def inference(**kwargs):
            self.events.append("lipsync")
            target = self.base / "vid_output" / f"{self.job_id}.mp4"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"moving frames and voice")

        self.ns = dict(
            Path=Path, shutil=shutil, time=time, JOBS_DIR=self.root,
            SOURCES_DIR=self.root, FPS=25, avatar_cache={},
            set_job=lambda job, **kw: self.states.update(kw),
            release_musetalk_engine=lambda: self.events.append("release"),
            run_motion=Mock(side_effect=motion), prepare_avatar_from_local=Mock(side_effect=prepare),
            synthesize_speech=lambda *args: self.events.append("speech"),
            avatar_base=lambda avatar: self.base, cleanup_outputs=Mock(),
        )
        # Import only the worker function so these checks require no installed
        # GPU frameworks or external weights. Its actual source is executed.
        module = ast.parse((ROOT / "server/main.py").read_text())
        fn = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "run_motion_job")
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "main.py", "exec"), self.ns)

    def run_job(self):
        self.ns["run_motion_job"](self.job_id, {"work_dir": str(self.work), "text": "Hola"})

    def test_lipsync_receives_generated_motion_after_gpu_release(self):
        self.run_job()
        self.assertEqual(self.events, ["release", "movement", "prepare", "speech", "lipsync"])
        self.assertEqual(self.states["status"], "done")
        self.assertEqual((self.root / f"{self.job_id}.mp4").read_bytes(), b"moving frames and voice")
        self.assertFalse(self.work.exists())
        self.assertFalse(self.base.exists())

    def test_failed_motion_never_falls_back_to_still_photo(self):
        self.ns["run_motion"].side_effect = RuntimeError("GPU out of memory")
        self.run_job()
        self.assertEqual(self.states["status"], "error")
        self.ns["prepare_avatar_from_local"].assert_not_called()
        self.assertFalse((self.root / f"{self.job_id}.mp4").exists())
        self.assertFalse(self.work.exists())

    def test_missing_lipsync_output_is_an_error(self):
        self.ns["prepare_avatar_from_local"].side_effect = None
        self.ns["prepare_avatar_from_local"].return_value = Mock()
        self.run_job()
        self.assertEqual(self.states["status"], "error")
        self.assertNotIn("video_endpoint", self.states)


if __name__ == "__main__":
    unittest.main()
