#!/usr/bin/env python3

import contextlib
import io
import sys
import unittest
from unittest import mock

import monitor_fleet
from monitor_fleet import cancellation_targets, choose_winner, parse_queue


PREFIX = "u6gb-16-nodes-18-jluy-"


class MonitorFleetTest(unittest.TestCase):
    def test_parse_and_choose_earliest_running_candidate(self) -> None:
        rows = parse_queue(
            "20|u6gb-16-nodes-18-jluy-002|RUNNING|2026-07-16T15:01:00|16\n"
            "10|u6gb-16-nodes-18-jluy-001|RUNNING|2026-07-16T15:00:00|16\n"
            "30|u6gb-16-nodes-18-jluy-003|PENDING|N/A|16\n"
        )
        self.assertEqual(choose_winner(rows, PREFIX), "10")

    def test_cancel_only_explicit_active_siblings(self) -> None:
        rows = parse_queue(
            "10|u6gb-16-nodes-18-jluy-001|RUNNING|2026-07-16T15:00:00|16\n"
            "20|u6gb-16-nodes-18-jluy-002|PENDING|N/A|16\n"
            "30|unrelated|RUNNING|2026-07-16T14:00:00|16\n"
            "40|u6gb-16-nodes-18-jluy-004|COMPLETED|2026-07-16T13:00:00|16\n"
        )
        self.assertEqual(
            cancellation_targets(rows, {"10", "20", "30", "40"}, "10", PREFIX),
            ["20"],
        )

    def test_no_running_candidate_has_no_winner(self) -> None:
        rows = parse_queue("10|u6gb-16-nodes-18-jluy-001|PENDING|N/A|16\n")
        self.assertIsNone(choose_winner(rows, PREFIX))

    def test_monitor_rejects_sub_minute_interval(self) -> None:
        with mock.patch.object(sys, "argv", ["monitor_fleet.py", "--interval", "59", "10"]):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    monitor_fleet.parse_args()


if __name__ == "__main__":
    unittest.main()
