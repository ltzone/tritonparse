#  Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for Ascend bishengir NPUIR pass-level IR capture.

These tests are fully offline: the raw ``bishengir-compile`` dump and the
TTAdapter input are checked in as fixtures under ``tests/cpu/fixtures/npuir``.
"""

import gzip
import json
import os
import unittest
from pathlib import Path

from tritonparse.backend import (
    AscendTritonAdapter,
    humanize_bishengir_pass,
    make_npuir_pass_stage_descriptor,
    NPUIR_PASS_DISPLAY_ORDER_BASE,
    NPUIR_PASS_STAGE_PREFIX,
    sanitize_npuir_pass_name,
)
from tritonparse.parse.trace_processor import (
    _inject_npuir_pass_irs,
    _resolve_source_mappable_stage_keys,
    _split_bishengir_dump_log,
    parse_single_trace_content,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "npuir"
DUMP_PATH = FIXTURE_DIR / "kernel.npuir_passes.log.gz"
TTADAPTER_PATH = FIXTURE_DIR / "kernel.ttadapter"

RAW_LOG_FILENAME = "kernel.npuir_passes.log"
KERNEL_HASH = "0" * 64
SOURCE_FILE = (
    "/home/user1/litao/Triton-distributed-ascend-release/python/"
    "triton_dist/test/ascend/api/test_barrier_ops.py"
)


def _load_dump() -> str:
    with gzip.open(DUMP_PATH, "rt", encoding="utf-8", errors="replace") as f:
        return f.read()


def _build_entry(*, include_python_source: bool = True, backend: str = "cann") -> dict:
    payload = {
        "metadata": {
            "backend_name": backend,
            "hash": KERNEL_HASH,
            "num_warps": 4,
            "name": "kernel",
        },
        "file_content": {
            "kernel.ttadapter": TTADAPTER_PATH.read_text(),
            RAW_LOG_FILENAME: _load_dump(),
        },
        "file_path": {RAW_LOG_FILENAME: str(DUMP_PATH)},
    }
    if include_python_source:
        payload["python_source"] = {
            "file_path": SOURCE_FILE,
            "code": "def kernel():\n    pass\n",
            "start_line": 1,
            "end_line": 2,
        }
    return {"event_type": "compilation", "payload": payload}


class TestSplitBishengirDumpLog(unittest.TestCase):
    """Unit tests for the header-driven log splitter."""

    LOG = (
        "[WARNING] some compiler noise\n"
        "// -----// IR Dump After Canonicalizer (canonicalize) //----- //\n"
        "module {\n  // canonicalized\n}\n"
        "// -----// IR Dump After CSE (cse) //----- //\n"
        "module {\n  // cse\n}\n"
        "// -----// IR Dump After CSE (cse) //----- //\n"
        "module {\n  // cse again\n}\n"
        "[ERROR] trailing noise\n"
    )

    def test_splits_each_pass_and_keeps_order(self):
        records = _split_bishengir_dump_log(self.LOG)
        self.assertEqual([r[0] for r in records], ["Canonicalizer", "CSE", "CSE"])
        self.assertEqual([r[1] for r in records], ["canonicalize", "cse", "cse"])

    def test_ignores_content_outside_dumps(self):
        records = _split_bishengir_dump_log(self.LOG)
        for _, _, ir in records:
            self.assertNotIn("compiler noise", ir)
            self.assertNotIn("trailing noise", ir)
        self.assertEqual(records[0][2], "module {\n  // canonicalized\n}\n")

    def test_empty_and_headerless_inputs(self):
        self.assertEqual(_split_bishengir_dump_log(""), [])
        self.assertEqual(_split_bishengir_dump_log("no dumps here\n"), [])

    def test_header_without_trailing_newline(self):
        records = _split_bishengir_dump_log(
            "// -----// IR Dump After Foo (foo) //----- //\nmodule {}\n"
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0][:2], ("Foo", "foo"))
        self.assertEqual(records[0][2], "module {}\n")

    def test_real_fixture_dump(self):
        records = _split_bishengir_dump_log(_load_dump())
        # The fixture is a real bishengir run; assert on shape rather than an
        # exact pass count so a toolchain bump does not break the test.
        self.assertGreater(len(records), 100)
        args = [r[1] for r in records]
        self.assertIn("convert-linalg-to-hfusion", args)
        self.assertIn("hivm-graph-sync-solver", args)
        self.assertIn("convert-hivm-to-std", args)
        for _, pass_arg, ir in records:
            self.assertTrue(pass_arg)
            self.assertTrue(ir.strip())
            # Compiler diagnostics are never attributed to a pass's IR.
            self.assertNotRegex(ir, r"(?m)^\[(?:ERROR|WARNING|INFO|DEBUG)\]")


class TestPassLabelHelpers(unittest.TestCase):
    """Label/sanitisation helpers used to name stages."""

    def test_curated_label_wins(self):
        self.assertEqual(
            humanize_bishengir_pass(
                "convert-linalg-to-hfusion", "ConvertLinalgToHFusion"
            ),
            "Convert Linalg to HFusion",
        )

    def test_camel_case_fallback(self):
        self.assertEqual(
            humanize_bishengir_pass("some-pass", "SomeUnknownPass"), "Some Unknown Pass"
        )

    def test_title_case_fallback_when_no_mlir_name(self):
        self.assertEqual(humanize_bishengir_pass("weird_pass"), "Weird Pass")

    def test_sanitize(self):
        self.assertEqual(
            sanitize_npuir_pass_name("hivm-graph-sync-solver"), "hivm_graph_sync_solver"
        )
        self.assertEqual(sanitize_npuir_pass_name("scf.for-loop"), "scf_for_loop")

    def test_stage_descriptor_shape(self):
        stage = make_npuir_pass_stage_descriptor(
            "npuir_pass_cse", "CSE", NPUIR_PASS_DISPLAY_ORDER_BASE + 1
        )
        self.assertEqual(stage.name, "npuir_pass_cse")
        # No trailing ".mlir": the frontend recovers the stage name from the
        # filename with split(".").pop().
        self.assertEqual(stage.extension, ".npuir_pass_cse")
        self.assertEqual(stage.display_name, "NPUIR Pass: CSE")
        self.assertTrue(stage.is_text)
        self.assertTrue(stage.supports_source_mapping)
        self.assertEqual(stage.syntax_id, "mlir")
        # Flagged so the frontend keeps grouping anchors on the main stages.
        self.assertTrue(stage.is_pass_level)

    def test_static_stages_are_not_pass_level(self):
        adapter = AscendTritonAdapter()
        for stage in adapter.list_ir_stages():
            self.assertFalse(stage.is_pass_level, stage.name)


class TestInjectNpuirPassIrs(unittest.TestCase):
    """Injection of split pass artifacts into a trace payload."""

    def test_injects_one_artifact_per_pass(self):
        payload = _build_entry()["payload"]
        stages = _inject_npuir_pass_irs(payload)

        pass_files = [
            k
            for k in payload["file_content"]
            if k.startswith(f"kernel.{NPUIR_PASS_STAGE_PREFIX}")
        ]
        self.assertEqual(len(pass_files), len(stages))
        self.assertGreater(len(stages), 100)

        # The raw combined log is consumed, never surfaced as a stage.
        self.assertNotIn(RAW_LOG_FILENAME, payload["file_content"])

        # Summary metadata mirrors the descriptors.
        self.assertEqual(len(payload["npuir_passes"]), len(stages))
        self.assertEqual(payload["npuir_passes"][0]["stage"], stages[0].name)

    def test_stage_order_is_strictly_increasing_and_after_ttadapter(self):
        payload = _build_entry()["payload"]
        stages = _inject_npuir_pass_irs(payload)
        orders = [s.display_order for s in stages]
        self.assertEqual(orders, sorted(orders))
        self.assertEqual(len(set(orders)), len(orders))
        self.assertGreater(orders[0], NPUIR_PASS_DISPLAY_ORDER_BASE)

    def test_repeated_passes_get_suffixed_stage_names(self):
        payload = {
            "metadata": {"backend_name": "cann"},
            "file_content": {
                RAW_LOG_FILENAME: (
                    "// -----// IR Dump After CSE (cse) //----- //\nmodule {}\n"
                    "// -----// IR Dump After CSE (cse) //----- //\nmodule {}\n"
                )
            },
        }
        stages = _inject_npuir_pass_irs(payload)
        self.assertEqual(
            [s.name for s in stages], ["npuir_pass_cse", "npuir_pass_cse_2"]
        )
        self.assertIn("kernel.npuir_pass_cse", payload["file_content"])
        self.assertIn("kernel.npuir_pass_cse_2", payload["file_content"])

    def test_filter_env_var_restricts_passes(self):
        payload = _build_entry()["payload"]
        original = os.environ.get("TRITONPARSE_NPUIR_PASS_FILTER")
        os.environ["TRITONPARSE_NPUIR_PASS_FILTER"] = "cse,hivm-graph-sync-solver"
        try:
            stages = _inject_npuir_pass_irs(payload)
        finally:
            if original is None:
                os.environ.pop("TRITONPARSE_NPUIR_PASS_FILTER", None)
            else:
                os.environ["TRITONPARSE_NPUIR_PASS_FILTER"] = original

        self.assertTrue(stages)
        self.assertTrue(
            all(
                s.name.startswith(
                    ("npuir_pass_cse", "npuir_pass_hivm_graph_sync_solver")
                )
                for s in stages
            ),
            [s.name for s in stages],
        )

    def test_no_dump_is_a_noop(self):
        payload = {"metadata": {"backend_name": "cann"}, "file_content": {}}
        self.assertEqual(_inject_npuir_pass_irs(payload), [])
        self.assertEqual(payload["file_content"], {})

    def test_falls_back_to_file_path_when_content_absent(self):
        payload = {
            "metadata": {"backend_name": "cann"},
            "file_content": {},
            "file_path": {RAW_LOG_FILENAME: str(DUMP_PATH)},
        }
        # The fixture on disk is gzipped; point file_path at the plain file we
        # materialise for this test only if it exists, else skip.
        import tempfile

        with tempfile.NamedTemporaryFile(
            "w", suffix=".log", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(_load_dump())
            tmp_path = tmp.name
        try:
            payload["file_path"] = {RAW_LOG_FILENAME: tmp_path}
            stages = _inject_npuir_pass_irs(payload)
            self.assertGreater(len(stages), 100)
        finally:
            os.unlink(tmp_path)


class TestStageResolutionWithNpuir(unittest.TestCase):
    """NPUIR stages participate in source-mappable stage resolution."""

    def test_extra_stages_are_resolved(self):
        payload = _build_entry()["payload"]
        stages = _inject_npuir_pass_irs(payload)
        entry = {"payload": payload}

        stage_keys = _resolve_source_mappable_stage_keys(entry, stages)
        self.assertIn("ttadapter", stage_keys)
        self.assertEqual(stage_keys["ttadapter"], "kernel.ttadapter")
        npuir_keys = [k for k in stage_keys if k.startswith(NPUIR_PASS_STAGE_PREFIX)]
        self.assertEqual(len(npuir_keys), len(stages))

    def test_resolution_without_extra_stages_ignores_npuir(self):
        payload = _build_entry()["payload"]
        _inject_npuir_pass_irs(payload)
        stage_keys = _resolve_source_mappable_stage_keys({"payload": payload})
        self.assertEqual(
            [k for k in stage_keys if k.startswith(NPUIR_PASS_STAGE_PREFIX)], []
        )


class TestParseSingleTraceContentNpuir(unittest.TestCase):
    """End-to-end parse of a trace carrying an NPUIR dump."""

    @classmethod
    def setUpClass(cls):
        cls.entry = _build_entry()
        cls.parsed = json.loads(parse_single_trace_content(json.dumps(cls.entry)))

    def test_ir_stages_include_npuir_and_are_ordered(self):
        stages = self.parsed["payload"]["ir_stages"]
        by_name = {s["name"]: s for s in stages}

        self.assertIn("ttir", by_name)
        self.assertIn("ttadapter", by_name)
        self.assertIn("npubin", by_name)
        self.assertIn("json", by_name)

        npuir = [s for s in stages if s["name"].startswith(NPUIR_PASS_STAGE_PREFIX)]
        self.assertGreater(len(npuir), 100)
        for stage in npuir:
            self.assertTrue(stage["is_pass_level"])
        for stage in stages:
            if not stage["name"].startswith(NPUIR_PASS_STAGE_PREFIX):
                self.assertFalse(stage["is_pass_level"])

        orders = {s["name"]: s["display_order"] for s in stages}
        # Every NPUIR pass sits between the Python-side stages and NPUBIN.
        for stage in npuir:
            self.assertGreater(orders[stage["name"]], orders["ttadapter"])
            self.assertLess(orders[stage["name"]], orders["npubin"])

        # Sorted by display_order, the first two viewable stages are TTIR and
        # TTAdapter -- i.e. the NPUIR passes do not steal the default panels.
        viewable = sorted(
            (s for s in stages if s["is_text"] and s["supports_source_mapping"]),
            key=lambda s: s["display_order"],
        )
        self.assertEqual(viewable[0]["name"], "ttir")
        self.assertEqual(viewable[1]["name"], "ttadapter")

    def test_file_content_has_per_pass_artifacts_and_no_raw_log(self):
        file_content = self.parsed["payload"]["file_content"]
        self.assertNotIn(RAW_LOG_FILENAME, file_content)
        self.assertNotIn("npuir_passes.log", file_content)

        pass_files = [
            k for k in file_content if k.startswith(f"kernel.{NPUIR_PASS_STAGE_PREFIX}")
        ]
        self.assertGreater(len(pass_files), 100)
        for key in pass_files:
            # The stage name is the final dot-separated token.
            self.assertTrue(key.split(".")[-1].startswith(NPUIR_PASS_STAGE_PREFIX))

    def test_source_mapping_generated_for_passes_that_carry_locs(self):
        mappings = self.parsed["payload"]["source_mappings"]
        file_content = self.parsed["payload"]["file_content"]

        npuir_artifacts = {
            k.split(".")[-1]
            for k in file_content
            if k.startswith(f"kernel.{NPUIR_PASS_STAGE_PREFIX}")
        }
        # Every generated mapping must belong to a real artifact; nothing is
        # invented for passes whose IR carries no locations.
        mapped_stages = set(mappings) - {"python"}
        self.assertTrue(mapped_stages)
        self.assertTrue(mapped_stages.issubset(npuir_artifacts | {"ttadapter"}))

        # Most bishengir passes drop the original `loc(...)` annotations, but
        # the debug-info pass preserves them, so at least that one is mapped.
        debug_stage = mappings.get("npuir_pass_ensure_debug_info_scope_on_llvm_func")
        self.assertIsNotNone(debug_stage)
        entry = next(iter(debug_stage.values()))
        self.assertIn(SOURCE_FILE, entry["file"])
        self.assertIn("npuir_pass_ensure_debug_info_scope_on_llvm_func_line", entry)

    def test_ttadapter_source_mapping_still_present(self):
        mappings = self.parsed["payload"]["source_mappings"]
        self.assertIn("ttadapter", mappings)
        self.assertTrue(mappings["ttadapter"])

    def test_summary_metadata_is_recorded(self):
        summary = self.parsed["payload"]["npuir_passes"]
        self.assertGreater(len(summary), 100)
        first = summary[0]
        self.assertEqual(
            set(first),
            {
                "stage",
                "pass_arg",
                "mlir_pass",
                "display_name",
                "display_order",
                "occurrence",
            },
        )

    def test_parsed_entry_validates_against_the_compilation_schema(self):
        """`is_pass_level` and `npuir_passes` must be schema-legal.

        The IRStage definition is `additionalProperties: false`, so a new field
        on a stage descriptor is rejected unless the schema knows about it.
        """
        from datetime import datetime, timezone

        from tritonparse.validation.json_validator import validate_record

        entry = _build_entry()
        entry["pid"] = 1
        entry["timestamp"] = datetime.now(timezone.utc).isoformat()
        parsed = json.loads(parse_single_trace_content(json.dumps(entry)))

        ok, errors = validate_record(parsed)
        self.assertTrue(ok, errors)
        # The parsed entry must actually exercise both new fields.
        self.assertTrue(parsed["payload"]["npuir_passes"])
        self.assertTrue(
            any(stage.get("is_pass_level") for stage in parsed["payload"]["ir_stages"])
        )

    def test_trace_without_dump_is_unaffected(self):
        entry = _build_entry()
        entry["payload"]["file_content"].pop(RAW_LOG_FILENAME)
        entry["payload"]["file_path"] = {}
        parsed = json.loads(parse_single_trace_content(json.dumps(entry)))

        stages = parsed["payload"]["ir_stages"]
        self.assertFalse(
            [s for s in stages if s["name"].startswith(NPUIR_PASS_STAGE_PREFIX)]
        )
        self.assertNotIn("npuir_passes", parsed["payload"])
        self.assertIn("ttadapter", parsed["payload"]["source_mappings"])


class TestAttachNpuirPassLog(unittest.TestCase):
    """Embedding the raw bishengir dump into the compilation trace."""

    def setUp(self):
        self.trace_data = {"file_content": {}, "file_path": {}}
        self._orig_env = os.environ.pop("TRITONPARSE_ENABLE_NPUIR_PASS_DUMP", None)

    def tearDown(self):
        if self._orig_env is None:
            os.environ.pop("TRITONPARSE_ENABLE_NPUIR_PASS_DUMP", None)
        else:
            os.environ["TRITONPARSE_ENABLE_NPUIR_PASS_DUMP"] = self._orig_env

    @staticmethod
    def _write_dump():
        import tempfile

        with tempfile.NamedTemporaryFile(
            "w", suffix=".log", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(_load_dump())
            return tmp.name

    def test_recorded_path_is_embedded(self):
        from tritonparse.structured_logging import _attach_npuir_pass_log

        path = self._write_dump()
        try:
            _attach_npuir_pass_log(self.trace_data, {"npuir_passes_log_path": path})
        finally:
            os.unlink(path)

        self.assertIn(RAW_LOG_FILENAME, self.trace_data["file_content"])
        self.assertIn(
            "IR Dump After", self.trace_data["file_content"][RAW_LOG_FILENAME]
        )
        self.assertEqual(self.trace_data["file_path"][RAW_LOG_FILENAME], path)

    def test_recorded_path_that_no_longer_exists_is_ignored(self):
        from tritonparse.structured_logging import _attach_npuir_pass_log

        _attach_npuir_pass_log(
            self.trace_data, {"npuir_passes_log_path": "/nonexistent/npuir_passes.log"}
        )
        self.assertEqual(self.trace_data["file_content"], {})

    def test_feature_off_and_no_recorded_path_is_a_noop(self):
        from tritonparse.structured_logging import _attach_npuir_pass_log

        _attach_npuir_pass_log(self.trace_data, {"hash": KERNEL_HASH})
        self.assertEqual(self.trace_data["file_content"], {})

    def test_feature_on_falls_back_to_dump_dir_lookup(self):
        from unittest.mock import patch

        from tritonparse.structured_logging import _attach_npuir_pass_log

        path = self._write_dump()
        os.environ["TRITONPARSE_ENABLE_NPUIR_PASS_DUMP"] = "1"
        try:
            with patch(
                "tritonparse.structured_logging._resolve_npuir_pass_log_path",
                return_value=path,
            ):
                _attach_npuir_pass_log(self.trace_data, {"hash": KERNEL_HASH})
        finally:
            os.unlink(path)

        self.assertIn(RAW_LOG_FILENAME, self.trace_data["file_content"])

    def test_feature_on_but_nothing_found_is_a_noop(self):
        from unittest.mock import patch

        from tritonparse.structured_logging import _attach_npuir_pass_log

        os.environ["TRITONPARSE_ENABLE_NPUIR_PASS_DUMP"] = "1"
        with patch(
            "tritonparse.structured_logging._resolve_npuir_pass_log_path",
            return_value=None,
        ):
            _attach_npuir_pass_log(self.trace_data, {"hash": KERNEL_HASH})
        self.assertEqual(self.trace_data["file_content"], {})

    def test_resolver_without_hash_returns_none(self):
        from tritonparse.structured_logging import _resolve_npuir_pass_log_path

        self.assertIsNone(_resolve_npuir_pass_log_path({}))

    def test_env_flag_parsing(self):
        from tritonparse.structured_logging import _npuir_pass_dump_enabled

        for value in ("1", "true", "TRUE", "yes", "on"):
            os.environ["TRITONPARSE_ENABLE_NPUIR_PASS_DUMP"] = value
            self.assertTrue(_npuir_pass_dump_enabled(), value)
        for value in ("0", "false", "no", ""):
            os.environ["TRITONPARSE_ENABLE_NPUIR_PASS_DUMP"] = value
            self.assertFalse(_npuir_pass_dump_enabled(), value)
        os.environ.pop("TRITONPARSE_ENABLE_NPUIR_PASS_DUMP", None)
        self.assertFalse(_npuir_pass_dump_enabled())


class TestAscendAdapterStageOrdering(unittest.TestCase):
    """The static Ascend adapter keeps NPUBIN/JSON last."""

    def test_npubin_and_json_are_ordered_after_npuir_range(self):
        adapter = AscendTritonAdapter()
        by_name = {s.name: s for s in adapter.list_ir_stages()}
        self.assertGreater(by_name["npubin"].display_order, 1000)
        self.assertGreater(
            by_name["json"].display_order, by_name["npubin"].display_order
        )
        self.assertGreater(
            by_name["npubin"].display_order, by_name["bcmlir"].display_order
        )


if __name__ == "__main__":
    unittest.main()
