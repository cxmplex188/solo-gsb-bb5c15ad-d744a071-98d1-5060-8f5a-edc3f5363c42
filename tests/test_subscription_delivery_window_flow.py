"""Regression tests: a freshly created subscription is immediately configurable.

The id returned by ``Database.create_subscription`` (the same id
``POST /subscriptions`` returns) must be usable straight away for
``set_delivery_window`` (the backing call of
``PUT /subscriptions/{id}/delivery-window``): it must persist inside the
install transaction, and the following ``get_delivery_window``
(``GET`` on the same path) must answer ``configured=True`` with the
canonicalized window. A genuinely unknown id must still be rejected
(``None`` -> the API maps it to 404), and the installed window must survive a
process restart (a new ``Database`` over the same file). These tests pin the
single identifier/transaction-visible row shared by subscription creation,
lookup and delivery-window read/write. The DST math itself is covered in
``test_delivery_window_dst.py`` and is intentionally left untouched.

Pure stdlib: run with either of

    python3 -m unittest discover -s tests -v
    python3 -m pytest tests/
"""
from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timezone

from app.db import Database

# A never-created, well-formed 12-hex id (the same shape create returns).
UNKNOWN_ID = "0" * 12

NY = "America/New_York"


class DeliveryWindowCreateFlowTests(unittest.TestCase):
    """create -> configure -> query on the id create returned."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(os.path.join(self.tmp.name, "flow.db"))

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def _create(self) -> str:
        sid = self.db.create_subscription("http://127.0.0.1:1/hook", "s")["id"]
        self.assertEqual(len(sid), 12)
        return sid

    def test_set_window_immediately_after_create_persists(self):
        sid = self._create()
        # Unsorted weekdays / HH:MM inputs are canonicalized on persistence.
        policy = self.db.set_delivery_window(sid, NY, [5, 1, 3, 2, 4], 9 * 3600, 18 * 3600)
        self.assertIsNotNone(policy)
        self.assertTrue(policy["configured"])
        self.assertEqual(policy["timezone"], NY)
        self.assertEqual(policy["weekdays"], [1, 2, 3, 4, 5])
        self.assertEqual(policy["local_start"], 9 * 3600)
        self.assertEqual(policy["local_end"], 18 * 3600)
        self.assertIsNotNone(policy["updated_at"])

    def test_get_after_set_returns_enabled_and_same_canonical_window(self):
        sid = self._create()
        installed = self.db.set_delivery_window(sid, NY, [5, 1, 3, 2, 4], 9 * 3600, 18 * 3600)
        read = self.db.get_delivery_window(sid)
        self.assertIsNotNone(read)
        self.assertTrue(read["configured"])
        # The write's read-back and the subsequent GET are one persisted view.
        self.assertEqual(read, installed)
        self.assertEqual(read["weekdays"], [1, 2, 3, 4, 5])

    def test_fresh_subscription_has_no_window_but_exists(self):
        sid = self._create()
        policy = self.db.get_delivery_window(sid)
        self.assertIsNotNone(policy)          # subscription exists -> not a 404
        self.assertFalse(policy["configured"])  # enabled=false until configured
        self.assertTrue(self.db.subscription_exists(sid))
        self.assertIsNotNone(self.db.get_subscription(sid))

    def test_created_identity_round_trips_through_every_lookup(self):
        """The id create hands out is the exact key every lookup reads by."""
        sid = self._create()
        self.assertEqual(self.db.get_subscription(sid)["id"], sid)
        self.assertTrue(self.db.subscription_exists(sid))
        self.assertIsNotNone(self.db.set_delivery_window(sid, "UTC", [1, 2], 32400, 64800))
        self.assertTrue(self.db.get_delivery_window(sid)["configured"])

    def test_installed_window_survives_process_restart(self):
        sid = self._create()
        self.db.set_delivery_window(sid, NY, [1, 2, 3, 4, 5], 9 * 3600, 18 * 3600)
        # A second "process": a new connection/Database over the same file.
        reopened = Database(os.path.join(self.tmp.name, "flow.db"))
        try:
            self.assertIsNotNone(reopened.get_subscription(sid))
            policy = reopened.get_delivery_window(sid)
            self.assertIsNotNone(policy)
            self.assertTrue(policy["configured"])
            self.assertEqual(policy["timezone"], NY)
            self.assertEqual(policy["weekdays"], [1, 2, 3, 4, 5])
            self.assertEqual(policy["local_start"], 9 * 3600)
            self.assertEqual(policy["local_end"], 18 * 3600)
        finally:
            reopened.close()

    def test_just_persisted_window_is_what_the_claim_gate_reads(self):
        """End-to-end chain: create -> event -> install closed window -> claim.

        Monday-Friday 09:00-18:00 New York (September is EDT, UTC-4, so open
        13:00-22:00 UTC). A due task on Saturday 2026-09-19 23:00 UTC must not
        be claimed and is pushed to the Monday opening, proving the
        just-persisted configuration is immediately the worker's gate.
        """
        sid = self._create()
        self.db.set_delivery_window(sid, NY, [1, 2, 3, 4, 5], 9 * 3600, 18 * 3600)
        self.db.create_event("evt-flow-closed", '{"n": 1}')
        task, created = self.db.create_task("evt-flow-closed", sid, 4)
        self.assertTrue(created)
        saturday_night = datetime(2026, 9, 19, 23, 0, tzinfo=timezone.utc).timestamp()
        self.assertIsNone(self.db.claim_due_task("flow-worker", saturday_night))
        pushed = self.db.get_delivery_task(task["id"])["task"]["next_attempt_at"]
        monday_open = datetime(2026, 9, 21, 13, 0, tzinfo=timezone.utc).timestamp()
        self.assertEqual(pushed, monday_open)


class UnknownSubscriptionWindowTests(unittest.TestCase):
    """Real non-existent ids stay non-existent at every layer (API -> 404)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(os.path.join(self.tmp.name, "missing.db"))

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_get_window_unknown_is_none(self):
        self.assertIsNone(self.db.get_delivery_window(UNKNOWN_ID))

    def test_set_window_unknown_is_none_and_writes_nothing(self):
        self.assertIsNone(
            self.db.set_delivery_window(UNKNOWN_ID, NY, [1], 9 * 3600, 18 * 3600)
        )
        self.assertFalse(self.db.subscription_exists(UNKNOWN_ID))
        self.assertIsNone(self.db.get_subscription(UNKNOWN_ID))

    def test_lookup_helpers_unknown(self):
        self.assertFalse(self.db.subscription_exists(UNKNOWN_ID))
        self.assertIsNone(self.db.get_subscription(UNKNOWN_ID))
        # The created id and the unknown id never collide.
        sid = self.db.create_subscription("http://x", "s")["id"]
        self.assertTrue(self.db.subscription_exists(sid))
        self.assertFalse(self.db.subscription_exists(UNKNOWN_ID))


class HttpMappingTests(unittest.TestCase):
    """The API layer maps the DB outcomes to 200 vs 404 (uses no HTTP client)."""

    def setUp(self):
        try:
            import fastapi  # noqa: F401
        except ImportError:
            self.skipTest("fastapi not installed; DB-layer coverage still runs")
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "http.db")
        self.db = Database(self.db_path)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_endpoint_chain_via_app_functions(self):
        # Exercise the exact decision the route handlers make, including the
        # serialization the response model validates against.
        from app.main import _delivery_window_out
        import time

        sid = self.db.create_subscription("http://x", "s")["id"]
        policy = self.db.set_delivery_window(sid, NY, [1, 2, 3, 4, 5], 9 * 3600, 18 * 3600)
        self.assertIsNotNone(policy)  # route returns 200 with this payload
        out = _delivery_window_out(time.time(), policy)
        self.assertTrue(out["enabled"])
        self.assertEqual(out["weekdays"], [1, 2, 3, 4, 5])
        self.assertEqual(out["local_start"], "09:00:00")
        self.assertEqual(out["local_end"], "18:00:00")

        # Same for the GET path.
        read = self.db.get_delivery_window(sid)
        self.assertTrue(_delivery_window_out(time.time(), read)["enabled"])

        # Unknown id -> None -> the routes raise 404.
        self.assertIsNone(self.db.get_delivery_window(UNKNOWN_ID))
        self.assertIsNone(
            self.db.set_delivery_window(UNKNOWN_ID, NY, [1], 9 * 3600, 18 * 3600)
        )


if __name__ == "__main__":
    unittest.main()
