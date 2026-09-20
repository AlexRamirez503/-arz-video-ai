"""CPU-only checks for the visual-story compositor."""
import ast
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "server" / "main.py"


class CinematicMixTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "avatar.mp4"
        self.output = self.root / "cinematic.mp4"
        self.source.write_bytes(b"source")
        self.scenes = []
        for index in range(3):
            scene = self.root / f"scene-{index}.jpg"
            scene.write_bytes(b"scene")
            self.scenes.append(scene)
        self.run = Mock()
        self.ns = {
            "Path": Path,
            "shutil": shutil,
            "subprocess": Mock(run=self.run),
            "CINEMATIC_SCENE_PATHS": tuple(self.scenes),
            "_video_duration": lambda _: 8.0,
        }
        module = ast.parse(MAIN_PATH.read_text())
        fn = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "build_cinematic_mix"
        )
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "main.py", "exec"), self.ns)

    def test_visual_story_intercuts_scenes_and_keeps_audio_source(self):
        self.ns["build_cinematic_mix"](self.source, self.output)
        command = self.run.call_args.args[0]
        self.assertIn("-filter_complex", command)
        filters = command[command.index("-filter_complex") + 1]
        self.assertIn("concat=n=6:v=1:a=0[video]", filters)
        self.assertIn("[avatar0][scene0][avatar1][scene1][avatar2][scene2]", filters)
        self.assertEqual(command.count("-loop"), 3)
        self.assertIn("0:a?", command)

    def test_missing_scene_assets_produce_clear_error(self):
        self.scenes[1].unlink()
        with self.assertRaisesRegex(RuntimeError, "Faltan las escenas visuales"):
            self.ns["build_cinematic_mix"](self.source, self.output)


if __name__ == "__main__":
    unittest.main()
