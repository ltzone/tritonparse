#  Copyright (c) Meta Platforms, Inc. and affiliates.
"""Integration tests for NPUIR pass capture against the real Ascend backend.

Drives the actual ``linalg_to_bin_enable_npu_compile_*`` entry point with the
bishengir subprocess faked, and asserts the contract TritonParse depends on:

* ``bishengir-compile`` is invoked with ``--mlir-print-ir-after-all`` and
  *without* ``--mlir-print-ir-after-failure`` (which would silence every dump),
* the captured output is persisted as ``npuir_passes.log`` and its absolute path
  is recorded in the compilation metadata under ``npuir_passes_log_path``,
* the recorded dump splits cleanly into per-pass records.

The capture is expected to work against a **stock** triton-ascend: the hook in
``tritonparse.ascend_npuir`` installs it at runtime. If the optional
triton-ascend patch is also present its own helpers are exercised as well, but
its absence is not a failure. Everything here is skipped when the Ascend backend
cannot be imported (NVIDIA/AMD-only environments).
"""

import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tritonparse.ascend_npuir import _reset_for_tests, enable_npuir_pass_dump

try:
    from triton.backends.ascend import compiler as ascend_compiler

    ASCEND_AVAILABLE = True
    ASCEND_IMPORT_ERROR = ""
except Exception as e:  # pragma: no cover - depends on the environment
    ascend_compiler = None
    ASCEND_AVAILABLE = False
    ASCEND_IMPORT_ERROR = f"{type(e).__name__}: {e}"

# The optional triton-ascend patch. Its presence changes nothing for users, but
# when it is installed its helpers get their own coverage below.
TA_PATCHED = ASCEND_AVAILABLE and hasattr(ascend_compiler, "_npuir_pass_dump_enabled")

_ENV_KEYS = (
    "TRITONPARSE_ENABLE_NPUIR_PASS_DUMP",
    "TRITONPARSE_NPUIR_PASS_FILTER",
)

TTADAPTER = (
    "module attributes {} {\n"
    "  func.func @k(%arg0: memref<?xf32> {tt.tensor_kind = 0 : i32}) "
    'attributes {mix_mode = "aiv", parallel_mode = "simd"} {\n'
    "    return\n"
    "  }\n"
    "}\n"
)

DUMP = (
    "[WARNING] noise\n"
    "// -----// IR Dump After Canonicalizer (canonicalize) //----- //\n"
    "module {\n}\n"
    "// -----// IR Dump After GraphSyncSolver (hivm-graph-sync-solver) //----- //\n"
    "module {\n}\n"
)


class _Metadata(dict):
    """Metadata stand-in: unknown keys read as ``None``.

    The Ascend compile functions read a large and evolving set of option keys;
    ``None`` is the documented "unset" value for all of them, so this keeps the
    test focused on the NPUIR contract instead of on the option list.
    """

    def __missing__(self, key):
        return None


class _Target:
    arch = "Ascend910B1"


class _Options:
    debug = False
    mix_mode = "aiv"


def _run_compile(stderr: bytes = DUMP.encode()):
    """Run the 910_95 compile entry point with the bishengir subprocess faked.

    Returns ``(cmd, metadata, dump_path, dump_content)``. The 910_95 variant is
    used because A2_A3 resolves its target via ``NPUUtils().get_arch()``, which
    needs a live NPU runtime. The temp trees holding the dump are removed before
    returning, so the content is read while they still exist.
    """
    captured = {}
    metadata = _Metadata(
        {
            "hash": "abc123",
            "target": _Target(),
            "kernel_name": "k",
            "enable_auto_bind_sub_block": None,
            "auto_tile_and_bind_sub_block": True,
            "required_ub_bits": 0,
            "mix_mode": "aiv",
            "parallel_mode": "simd",
            "bitcodes": None,
        }
    )

    def fake_run(cmd_list, **kwargs):
        cmd_list = list(cmd_list)
        if "--help" in cmd_list:
            # Only the API-change probe passes --help. Advertising the marker
            # option selects the newer "kernel.o" output name.
            return subprocess.CompletedProcess(
                cmd_list, 0, stdout="--limit-auto-multi-buffer-buffer\n", stderr=""
            )
        captured["cmd"] = cmd_list
        out_path = cmd_list[cmd_list.index("-o") + 1]
        for candidate in (out_path + ".o", out_path + "_reloc.o"):
            with open(candidate, "w") as f:
                f.write("object")
        return subprocess.CompletedProcess(cmd_list, 0, stdout=b"", stderr=stderr)

    with (
        tempfile.TemporaryDirectory() as fake_tool_dir,
        tempfile.TemporaryDirectory() as dump_root,
    ):
        tool = os.path.join(fake_tool_dir, "bishengir-compile")
        with open(tool, "w") as f:
            f.write("#!/bin/sh\n")
        os.chmod(tool, 0o755)

        original_path = os.environ.get("PATH", "")
        original_dump_dir = os.environ.get("TRITON_DUMP_DIR")
        os.environ["PATH"] = fake_tool_dir + os.pathsep + original_path
        # Keep the dump inside the temp tree instead of the user's cache.
        os.environ["TRITON_DUMP_DIR"] = dump_root
        try:
            with patch.object(subprocess, "run", fake_run):
                ascend_compiler.linalg_to_bin_enable_npu_compile_910_95(
                    TTADAPTER, metadata, _Options()
                )
            dump_path = metadata["npuir_passes_log_path"]
            if dump_path and os.path.exists(dump_path):
                with open(dump_path, encoding="utf-8") as f:
                    captured["dump_content"] = f.read()
        finally:
            os.environ["PATH"] = original_path
            if original_dump_dir is None:
                os.environ.pop("TRITON_DUMP_DIR", None)
            else:
                os.environ["TRITON_DUMP_DIR"] = original_dump_dir

    if not captured.get("cmd"):
        raise AssertionError("the compile subprocess was never invoked")
    return captured["cmd"], metadata, dump_path, captured.get("dump_content")


@unittest.skipUnless(
    ASCEND_AVAILABLE, f"Ascend backend unavailable: {ASCEND_IMPORT_ERROR}"
)
class TestNpuirFlagsReachSubprocess(unittest.TestCase):
    """The bishengir command line and the dump it produces, hook installed."""

    def setUp(self):
        self._env_backup = {key: os.environ.pop(key, None) for key in _ENV_KEYS}
        os.environ["TRITONPARSE_ENABLE_NPUIR_PASS_DUMP"] = "1"
        if not enable_npuir_pass_dump():
            self.skipTest("Ascend NPUIR hook could not be installed")

    def tearDown(self):
        _reset_for_tests()
        for key, value in self._env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_flag_is_added_and_dump_is_recorded(self):
        cmd, _, dump_path, dump_content = _run_compile()
        self.assertIn("--mlir-print-ir-after-all", cmd)
        # MLIR suppresses after-pass printing entirely when this is present, so
        # the dump mode must replace it rather than add to it.
        self.assertNotIn("--mlir-print-ir-after-failure", cmd)
        self.assertIsNotNone(dump_path)
        self.assertTrue(dump_path.endswith("npuir_passes.log"), dump_path)
        self.assertIn("IR Dump After GraphSyncSolver", dump_content)

    def test_dump_is_split_into_tritonparse_artifacts(self):
        """The captured dump feeds TritonParse's splitter unchanged."""
        from tritonparse.parse.trace_processor import _split_bishengir_dump_log

        _, _, _, dump_content = _run_compile()
        records = _split_bishengir_dump_log(dump_content)
        self.assertEqual(
            [r[1] for r in records], ["canonicalize", "hivm-graph-sync-solver"]
        )

    def test_filter_env_var_narrows_the_injected_flags(self):
        os.environ["TRITONPARSE_NPUIR_PASS_FILTER"] = "cse,hivm-graph-sync-solver"
        cmd, _, _, _ = _run_compile()
        self.assertNotIn("--mlir-print-ir-after-all", cmd)
        self.assertIn("--mlir-print-ir-after=cse", cmd)
        self.assertIn("--mlir-print-ir-after=hivm-graph-sync-solver", cmd)
        self.assertNotIn("--mlir-print-ir-after-failure", cmd)


@unittest.skipUnless(
    ASCEND_AVAILABLE, f"Ascend backend unavailable: {ASCEND_IMPORT_ERROR}"
)
class TestNpuirDisabled(unittest.TestCase):
    """With the hook uninstalled the command line is left exactly as before."""

    def setUp(self):
        self._env_backup = {key: os.environ.pop(key, None) for key in _ENV_KEYS}
        _reset_for_tests()

    def tearDown(self):
        _reset_for_tests()
        for key, value in self._env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_no_dump_flags_and_no_dump_recorded(self):
        cmd, metadata, dump_path, _ = _run_compile(stderr=b"")
        self.assertNotIn("--mlir-print-ir-after-all", cmd)
        self.assertFalse(
            any(arg.startswith("--mlir-print-ir-after=") for arg in cmd), cmd
        )
        # The stock backend's own failure-IR flag is left untouched.
        self.assertIn("--mlir-print-ir-after-failure", cmd)
        self.assertIsNone(dump_path)


@unittest.skipUnless(
    TA_PATCHED,
    "optional triton-ascend patch not installed (the runtime hook covers it)",
)
class TestOptionalTaPatchHelpers(unittest.TestCase):
    """Coverage for the optional in-tree triton-ascend patch, when present."""

    def setUp(self):
        self._env_backup = {key: os.environ.pop(key, None) for key in _ENV_KEYS}

    def tearDown(self):
        for key, value in self._env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_helper_functions_exist(self):
        for name in (
            "_npuir_pass_dump_enabled",
            "_append_npuir_pass_dump_options",
            "_save_npuir_pass_dump",
        ):
            self.assertTrue(hasattr(ascend_compiler, name), name)

    def test_enable_flag_parsing(self):
        for value in ("1", "true", "TRUE", "yes", "on"):
            os.environ["TRITONPARSE_ENABLE_NPUIR_PASS_DUMP"] = value
            self.assertTrue(ascend_compiler._npuir_pass_dump_enabled(), value)
        for value in ("0", "false", "no", ""):
            os.environ["TRITONPARSE_ENABLE_NPUIR_PASS_DUMP"] = value
            self.assertFalse(ascend_compiler._npuir_pass_dump_enabled(), value)

    def test_dump_all_flag(self):
        opts = ascend_compiler._append_npuir_pass_dump_options([])
        self.assertEqual(opts, ["--mlir-print-ir-after-all", "--mlir-print-debuginfo"])

    def test_dump_filter_flag(self):
        os.environ["TRITONPARSE_NPUIR_PASS_FILTER"] = " hivm-graph-sync-solver , cse "
        opts = ascend_compiler._append_npuir_pass_dump_options([])
        self.assertEqual(
            opts,
            [
                "--mlir-print-ir-after=hivm-graph-sync-solver",
                "--mlir-print-ir-after=cse",
                "--mlir-print-debuginfo",
            ],
        )
        self.assertNotIn("--mlir-print-ir-after-all", opts)

    def test_dump_flags_never_include_after_failure(self):
        """--mlir-print-ir-after-failure would silence every after-pass dump."""
        for filter_value in (None, "cse"):
            if filter_value is None:
                os.environ.pop("TRITONPARSE_NPUIR_PASS_FILTER", None)
            else:
                os.environ["TRITONPARSE_NPUIR_PASS_FILTER"] = filter_value
            opts = ascend_compiler._append_npuir_pass_dump_options([])
            self.assertNotIn("--mlir-print-ir-after-failure", opts)
            self.assertTrue(
                any(
                    arg == "--mlir-print-ir-after-all"
                    or arg.startswith("--mlir-print-ir-after=")
                    for arg in opts
                ),
                opts,
            )

    def test_empty_filter_falls_back_to_dump_all(self):
        os.environ["TRITONPARSE_NPUIR_PASS_FILTER"] = "  ,  "
        opts = ascend_compiler._append_npuir_pass_dump_options([])
        self.assertEqual(opts, ["--mlir-print-ir-after-all", "--mlir-print-debuginfo"])

    def test_debuginfo_flag_is_present(self):
        """Without it no dump carries loc(...), so no pass maps to a line."""
        for filter_value in (None, "cse"):
            if filter_value is None:
                os.environ.pop("TRITONPARSE_NPUIR_PASS_FILTER", None)
            else:
                os.environ["TRITONPARSE_NPUIR_PASS_FILTER"] = filter_value
            opts = ascend_compiler._append_npuir_pass_dump_options([])
            self.assertIn("--mlir-print-debuginfo", opts, filter_value)


if __name__ == "__main__":
    unittest.main()
