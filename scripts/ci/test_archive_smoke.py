# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Offline assertions for the archive probe; synthetic vectors are not inference proof."""

import copy
import runpy
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

PROBE = runpy.run_path(str(Path(__file__).with_name("archive_smoke.py")))


def fixture():
    return {
        "model": PROBE["MODEL"],
        "data": [{"index": i, "embedding": [1.0] + [0.0] * 383} for i in range(2)],
    }


class VectorContract(unittest.TestCase):
    def test_valid_ordered_unit_vectors(self):
        self.assertEqual(len(PROBE["vectors"](fixture())), 2)

    def test_rejects_identity_count_and_order(self):
        original = fixture()
        cases = [copy.deepcopy(original) for _ in range(3)]
        cases[0]["model"] = "wrong"
        cases[1]["data"].pop()
        cases[2]["data"].reverse()
        for candidate in cases:
            with self.subTest(candidate=candidate), self.assertRaises(AssertionError):
                PROBE["vectors"](candidate)

    def test_rejects_dimensions_nonfinite_and_nonunit(self):
        for vector in ([1.0], [float("nan")] * 384, [float("inf")] * 384, [0.0] * 384):
            candidate = fixture()
            candidate["data"][0]["embedding"] = vector
            with self.subTest(vector=vector[0]), self.assertRaises(AssertionError):
                PROBE["vectors"](candidate)


class ReleaseRecipeContract(unittest.TestCase):
    def test_annotated_tag_and_scoped_push(self):
        source = (Path(__file__).resolve().parents[2] / "justfile").read_text()
        recipe = source.split("release-tag VERSION:\n", 1)[1].split("\n\n", 1)[0]
        commands = [shlex.split(line.strip().replace("{{VERSION}}", "0.4.0"))
                    for line in recipe.splitlines() if line.strip()]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "repo"
            repo.mkdir()
            hooks = root / "hooks"
            hooks.mkdir()
            def git(*args):
                return subprocess.check_output(["git", *args], cwd=repo, text=True,
                                               stderr=subprocess.STDOUT).strip()
            git("init", "--quiet")
            git("init", "--bare", "--quiet", "../remote.git")
            for key, value in [("user.name", "Release Contract"),
                               ("user.email", "release-contract@example.invalid"),
                               ("commit.gpgsign", "false"), ("tag.gpgsign", "false"),
                               ("core.hooksPath", str(hooks))]:
                git("config", key, value)
            (repo / "fixture").write_text("release contract\n")
            git("add", "fixture")
            git("commit", "--quiet", "-m", "fixture")
            git("remote", "add", "origin", "../remote.git")
            git("tag", "unrelated-fixture-tag")
            for command in commands:
                subprocess.check_output(command, cwd=repo, text=True, stderr=subprocess.STDOUT)
            tag_type = git("cat-file", "-t", "v0.4.0")
            remote_tags = git("--git-dir=../remote.git", "tag", "--list").splitlines()
            self.assertEqual((tag_type, remote_tags), ("tag", ["v0.4.0"]))


if __name__ == "__main__":
    unittest.main()
