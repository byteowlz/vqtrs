"""Watchdog cleanup regression: a sampled child may exit before termination."""

import json
import os
import runpy
import signal
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

GUARD = Path(__file__).with_name("guard.py")


class CleanupTests(unittest.TestCase):
    def run_race(self, polls):
        process = MagicMock(pid=123, returncode=0)
        process.poll.side_effect = polls
        process.wait.return_value = 0
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"EG2_BENCH_WORK_DIR": directory}),
            patch("sys.argv", [str(GUARD), "race", "unused-command"]),
            patch("subprocess.Popen", return_value=process),
            patch("subprocess.check_output", return_value="123 1\n"),
            patch("subprocess.run", return_value=MagicMock(stdout="")),
            patch(
                "os.killpg", side_effect=PermissionError("group disappeared")
            ) as kill,
        ):
            with self.assertRaises(SystemExit) as error:
                runpy.run_path(str(GUARD), run_name="__main__")
            self.assertEqual(error.exception.code, 1)
            report = json.loads((Path(directory) / "race-memory.json").read_text())
            self.assertEqual(report["exit"], 1)
            self.assertEqual(report["stopped"], "cannot monitor owned task footprint")
            return kill.call_args_list

    def test_completed_child_does_not_receive_a_group_signal(self):
        self.assertEqual(self.run_race([None, None, 0]), [])

    def test_exit_during_group_signal_still_writes_failed_monitor_report(self):
        calls = self.run_race([None, None, None, 0])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].args, (123, signal.SIGTERM))

    def test_live_task_permission_failure_is_not_hidden(self):
        with self.assertRaises(PermissionError):
            self.run_race([None, None, None, None])


if __name__ == "__main__":
    unittest.main()
