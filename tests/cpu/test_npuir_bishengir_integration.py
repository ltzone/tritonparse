#  Copyright (c) Meta Platforms, Inc. and affiliates.
"""Integration test: real ``bishengir-compile`` emits per-pass IR dumps.

The NPUIR pass capture relies on bishengir honouring the *standard* MLIR flag
``--mlir-print-ir-after-all`` (rather than bishengir's own whitelist-restricted
``--bishengir-print-ir-after=``). This test runs the real toolchain against a
checked-in TTAdapter module and feeds the output through TritonParse's splitter,
so a bishengir upgrade that changed the flag or the dump header format fails
here instead of silently producing traces with no passes.

No NPU device is required: this is a compile-only path, and the assertions hold
whether or not the toolchain can finish the whole pipeline. Both the standalone
AscendNPU-IR build and the bishengir bundled with CANN are exercised in practice;
the CANN one can crash late in the pipeline (after ``cv-pipelining``) on some
modules, which still yields a valid, complete prefix of the pass dumps -- exactly
the prefix a user wants when debugging a failure. The test therefore asserts on
the beginning of the pipeline rather than on a specific total pass count. The
test is skipped when ``bishengir-compile`` is not on ``PATH``.
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from tritonparse._json_compat import dumps
from tritonparse.parse.trace_processor import (
    _inject_npuir_pass_irs,
    _split_bishengir_dump_log,
    parse_single_trace_content,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "npuir"
TTADAPTER_PATH = FIXTURE_DIR / "kernel.ttadapter"

BISHENGIR = shutil.which("bishengir-compile")

# Passes from the first half of the pipeline: present in every toolchain tried,
# and reached before any late-pipeline failure.
EARLY_PASSES = (
    "canonicalize-module",
    "convert-linalg-to-hfusion",
    "convert-hfusion-to-hivm",
)

# Text MLIR prints when it does not recognise a command line flag.
UNKNOWN_FLAG_MARKERS = (
    "Unknown command line argument",
    "Unknown option",
)


@unittest.skipUnless(BISHENGIR, "bishengir-compile not found on PATH")
@unittest.skipUnless(TTADAPTER_PATH.is_file(), "TTAdapter fixture missing")
class TestBishengirDumpFlag(unittest.TestCase):
    """``--mlir-print-ir-after-all`` must produce one dump per pass."""

    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "kernel.o")
            # Mirrors the flags triton-ascend passes for the 910_95 path. The
            # dump flags are the part under test.
            cmd = [
                BISHENGIR,
                str(TTADAPTER_PATH),
                "--target=Ascend910B1",
                "--enable-hfusion-compile=true",
                "--enable-triton-kernel-compile=true",
                "--mlir-print-ir-after-all",
                # Without this MLIR prints no loc(...) at all and no pass can be
                # mapped to a source line; see the splitter's module docstring.
                "--mlir-print-debuginfo",
                "-o",
                out,
            ]
            proc = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
            )
            cls.stdout = proc.stdout.decode("utf-8", errors="replace")
            cls.stderr = proc.stderr.decode("utf-8", errors="replace")
            cls.returncode = proc.returncode
            cls.records = _split_bishengir_dump_log(cls.stdout + cls.stderr)

    def test_flag_is_accepted(self):
        for marker in UNKNOWN_FLAG_MARKERS:
            self.assertNotIn(
                marker,
                self.stderr,
                "--mlir-print-ir-after-all was rejected by bishengir-compile",
            )

    def test_dumps_appear_on_stderr(self):
        self.assertGreater(
            len(self.records),
            20,
            "expected many per-pass dumps from --mlir-print-ir-after-all",
        )
        # Dumps go to stderr; stdout is reserved for tool output.
        self.assertEqual(
            len(_split_bishengir_dump_log(self.stdout)),
            0,
            "unexpected IR dumps on stdout",
        )

    def test_expected_early_passes_are_present(self):
        args = [r[1] for r in self.records]
        self.assertEqual(args[0], "canonicalize-module")
        for expected in EARLY_PASSES:
            self.assertIn(expected, args)

    def test_debuginfo_flag_makes_locations_visible(self):
        """The real toolchain must honour --mlir-print-debuginfo.

        Locations are stored on ops but only printed when this flag is set. If a
        bishengir build ignored it, the trace would still look fine while every
        pass silently lost its source mapping.
        """
        with_loc = [r for r in self.records if re.search(r"loc\([\"#]", r[2])]
        self.assertGreater(
            len(with_loc),
            len(self.records) * 0.9,
            f"only {len(with_loc)}/{len(self.records)} dumps carry loc(...); "
            "--mlir-print-debuginfo did not take effect",
        )

    def test_every_record_has_a_usable_name_and_body(self):
        for mlir_name, pass_arg, ir in self.records:
            self.assertTrue(mlir_name)
            self.assertTrue(pass_arg)
            self.assertRegex(pass_arg, r"^[a-z0-9][a-z0-9-]*$")
            self.assertTrue(ir.strip())

    def test_captured_dump_parses_into_trace_stages(self):
        """End-to-end: toolchain output -> trace artifacts and stages."""
        raw_log = self.stdout + self.stderr

        # First, the splitter contract on its own.
        payload = {
            "metadata": {"backend_name": "cann", "hash": "integration"},
            "file_content": {
                "kernel.ttadapter": TTADAPTER_PATH.read_text(),
                "kernel.npuir_passes.log": raw_log,
            },
        }
        stages = _inject_npuir_pass_irs(payload)
        self.assertEqual(len(stages), len(self.records))
        # Injection consumes the raw log; it is never a selectable stage.
        self.assertNotIn("kernel.npuir_passes.log", payload["file_content"])

        # Then the full parse path, from a fresh payload.
        parsed = json.loads(
            parse_single_trace_content(
                dumps(
                    {
                        "event_type": "compilation",
                        "payload": {
                            "metadata": {
                                "backend_name": "cann",
                                "hash": "integration",
                            },
                            "file_content": {
                                "kernel.ttadapter": TTADAPTER_PATH.read_text(),
                                "kernel.npuir_passes.log": raw_log,
                            },
                        },
                    }
                )
            )
        )
        names = {s["name"] for s in parsed["payload"]["ir_stages"]}
        self.assertIn("npuir_pass_convert_linalg_to_hfusion", names)
        self.assertIn("ttadapter", names)
        # The captured passes are real artifacts in the trace, not just stages.
        self.assertIn(
            "kernel.npuir_pass_convert_linalg_to_hfusion",
            parsed["payload"]["file_content"],
        )


if __name__ == "__main__":
    unittest.main()
