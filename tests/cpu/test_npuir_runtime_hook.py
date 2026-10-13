#  Copyright (c) Meta Platforms, Inc. and affiliates.
"""Unit tests for the runtime NPUIR pass-dump hook.

``tritonparse.ascend_npuir`` installs the NPUIR capture into a *stock*
triton-ascend by rewriting the ``bishengir-compile`` command line and capturing
its output. These tests cover that logic with fakes, so they run on any machine
(no triton, no bishengir, no NPU). The end-to-end check against a real toolchain
lives in ``test_npuir_ascend_contract.py`` and
``test_npuir_bishengir_integration.py``.
"""

import os
import subprocess
import unittest
from unittest.mock import patch

from tritonparse.ascend_npuir import (
    _COMPILER_BASENAMES,
    _DEBUGINFO_FLAG,
    _DUMP_ALL_FLAG,
    _FAILURE_FLAG,
    _is_bishengir_compile,
    _rewrite_cmd,
    _SubprocessProxy,
    enable_npuir_pass_dump,
    npuir_pass_dump_enabled,
)

_ENV_KEYS = (
    "TRITONPARSE_ENABLE_NPUIR_PASS_DUMP",
    "TRITONPARSE_NPUIR_PASS_FILTER",
)


class _EnvSandbox(unittest.TestCase):
    def setUp(self):
        self._backup = {key: os.environ.pop(key, None) for key in _ENV_KEYS}

    def tearDown(self):
        for key, value in self._backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class TestIsBishengirCompile(unittest.TestCase):
    def test_matches_both_compiler_names(self):
        self.assertTrue(_is_bishengir_compile(["/opt/cann/bin/bishengir-compile"]))
        self.assertTrue(_is_bishengir_compile(["/root/npuc", "in.mlir"]))
        self.assertTrue(_is_bishengir_compile(["bishengir-compile"]))

    def test_rejects_other_commands(self):
        self.assertFalse(_is_bishengir_compile(["/usr/bin/nvcc", "a.cu"]))
        self.assertFalse(_is_bishengir_compile(["/opt/bin/bishengir-opt"]))
        self.assertFalse(_is_bishengir_compile([]))
        self.assertFalse(_is_bishengir_compile(None))
        self.assertFalse(_is_bishengir_compile("bishengir-compile"))
        # A tuple is a valid argv for subprocess.run, so it is accepted.
        self.assertTrue(_is_bishengir_compile(("bishengir-compile",)))

    def test_expected_basenames_are_covered(self):
        for name in _COMPILER_BASENAMES:
            self.assertTrue(_is_bishengir_compile([f"/x/{name}"]))


class TestRewriteCmd(_EnvSandbox):
    BASE = [
        "/bin/bishengir-compile",
        "/tmp/kernel.ttadapter.mlir",
        "--target=Ascend910B1",
        _FAILURE_FLAG,
        "--mlir-print-stacktrace-on-diagnostic",
        "-o",
        "/tmp/kernel",
    ]

    def test_replaces_failure_flag_with_dump_all(self):
        out = _rewrite_cmd(list(self.BASE))
        self.assertNotIn(_FAILURE_FLAG, out)
        self.assertIn(_DUMP_ALL_FLAG, out)
        # Everything else is preserved, in order.
        self.assertEqual(
            [
                a
                for a in out
                if a not in (_FAILURE_FLAG, _DUMP_ALL_FLAG, _DEBUGINFO_FLAG)
            ],
            [a for a in self.BASE if a != _FAILURE_FLAG],
        )

    def test_debuginfo_flag_is_always_added(self):
        """Without it MLIR prints no `loc(...)`, so no pass maps to a line.

        MLIR stores locations on every op but only prints them when
        OpPrintingFlags::printDebugInfo is set. Omitting this flag makes every
        dump loc-free, which silently guts source mapping for all passes -- the
        bug this test exists to prevent.
        """
        for env_filter in (None, "cse", "cse,other"):
            if env_filter is None:
                os.environ.pop("TRITONPARSE_NPUIR_PASS_FILTER", None)
            else:
                os.environ["TRITONPARSE_NPUIR_PASS_FILTER"] = env_filter
            out = _rewrite_cmd(list(self.BASE))
            self.assertIn(_DEBUGINFO_FLAG, out, env_filter)

    def test_debuginfo_flag_is_not_duplicated(self):
        cmd = list(self.BASE) + [_DEBUGINFO_FLAG]
        out = _rewrite_cmd(cmd)
        self.assertEqual(out.count(_DEBUGINFO_FLAG), 1)

    def test_debuginfo_is_added_even_when_dump_flags_already_present(self):
        """A patched triton-ascend adds the dump flag but not necessarily this."""
        cmd = list(self.BASE) + [_DUMP_ALL_FLAG]
        out = _rewrite_cmd(cmd)
        self.assertEqual(out.count(_DUMP_ALL_FLAG), 1)
        self.assertIn(_DEBUGINFO_FLAG, out)

    def test_dump_flag_is_inserted_before_output(self):
        out = _rewrite_cmd(list(self.BASE))
        self.assertLess(out.index(_DUMP_ALL_FLAG), out.index("-o"))

    def test_filter_mode_uses_per_pass_flags(self):
        os.environ["TRITONPARSE_NPUIR_PASS_FILTER"] = " cse , hivm-inject-sync "
        out = _rewrite_cmd(list(self.BASE))
        self.assertNotIn(_DUMP_ALL_FLAG, out)
        self.assertIn("--mlir-print-ir-after=cse", out)
        self.assertIn("--mlir-print-ir-after=hivm-inject-sync", out)

    def test_blank_filter_falls_back_to_dump_all(self):
        for value in ("", "   ", " , "):
            os.environ["TRITONPARSE_NPUIR_PASS_FILTER"] = value
            self.assertIn(_DUMP_ALL_FLAG, _rewrite_cmd(list(self.BASE)), value)

    def test_existing_dump_flag_is_not_duplicated(self):
        """A patched triton-ascend already adds the flag; do not add it twice."""
        cmd = list(self.BASE) + [_DUMP_ALL_FLAG]
        out = _rewrite_cmd(cmd)
        self.assertEqual(out.count(_DUMP_ALL_FLAG), 1)
        self.assertNotIn(_FAILURE_FLAG, out)

    def test_existing_per_pass_flag_suppresses_dump_all(self):
        cmd = list(self.BASE) + ["--mlir-print-ir-after=cse"]
        out = _rewrite_cmd(cmd)
        self.assertNotIn(_DUMP_ALL_FLAG, out)
        self.assertEqual(out.count("--mlir-print-ir-after=cse"), 1)

    def test_appends_when_there_is_no_output_flag(self):
        out = _rewrite_cmd(["/bin/bishengir-compile", "in.mlir"])
        # No "-o", so the flags are appended after the input path.
        self.assertEqual(out[-2:], [_DUMP_ALL_FLAG, _DEBUGINFO_FLAG])


class _FakeSubprocess:
    """Stand-in for the ``subprocess`` module inside the Ascend compiler."""

    CalledProcessError = subprocess.CalledProcessError

    def __init__(self, result=None, raise_error=False):
        self.calls = []
        self._result = result
        self._raise = raise_error
        self.sentinel = object()

    def run(self, cmd, *args, **kwargs):
        self.calls.append((list(cmd), kwargs))
        if self._raise:
            raise self.CalledProcessError(1, cmd, b"out", b"err")
        return self._result

    def check_output(self, *args, **kwargs):  # forwarded, never intercepted
        return "check_output"


class TestSubprocessProxy(_EnvSandbox):
    def _proxy(self, **kwargs):
        real = _FakeSubprocess(**kwargs)
        return _SubprocessProxy(real), real

    def test_forwards_other_attributes(self):
        proxy, real = self._proxy()
        self.assertIs(proxy.sentinel, real.sentinel)
        self.assertEqual(proxy.check_output("x"), "check_output")

    @staticmethod
    def _completed(cmd=None, stdout=b"", stderr=b""):
        return subprocess.CompletedProcess(cmd or [], 0, stdout, stderr)

    def test_non_bishengir_calls_pass_through_untouched(self):
        result = self._completed()
        proxy, real = self._proxy(result=result)
        self.assertIs(proxy.run(["/usr/bin/true", "--flag"]), result)
        self.assertEqual(real.calls[0][0], ["/usr/bin/true", "--flag"])

    def test_bishengir_call_is_rewritten(self):
        proxy, real = self._proxy(result=self._completed())
        proxy.run(["/bin/bishengir-compile", "in.mlir", _FAILURE_FLAG, "-o", "out"])
        sent = real.calls[0][0]
        self.assertNotIn(_FAILURE_FLAG, sent)
        self.assertIn(_DUMP_ALL_FLAG, sent)

    def test_kwargs_are_forwarded(self):
        proxy, real = self._proxy(result=self._completed())
        proxy.run(["/bin/bishengir-compile", "-o", "x"], env={"A": "1"}, check=True)
        self.assertEqual(real.calls[0][1], {"env": {"A": "1"}, "check": True})

    def test_missing_stdout_attributes_do_not_break_the_proxy(self):
        """A custom runner may return something that is not a CompletedProcess."""
        proxy, _ = self._proxy(result="plain string")
        self.assertEqual(
            proxy.run(["/bin/bishengir-compile", "-o", "x"]), "plain string"
        )

    def test_capture_is_recorded_on_success(self):
        from tritonparse import ascend_npuir

        proxy, _ = self._proxy(result=subprocess.CompletedProcess([], 0, b"O", b"E"))
        proxy.run(["/bin/bishengir-compile", "-o", "x"])
        self.assertEqual(ascend_npuir._local.capture, (b"O", b"E"))

    def test_capture_is_recorded_on_failure(self):
        """The dump matters most when compilation fails."""
        from tritonparse import ascend_npuir

        proxy, _ = self._proxy(raise_error=True)
        with self.assertRaises(subprocess.CalledProcessError):
            proxy.run(["/bin/bishengir-compile", "-o", "x"])
        self.assertEqual(ascend_npuir._local.capture, (b"out", b"err"))

    def test_failure_of_a_non_bishengir_call_is_not_captured(self):
        from tritonparse import ascend_npuir

        ascend_npuir._local.capture = None
        proxy, _ = self._proxy(raise_error=True)
        with self.assertRaises(subprocess.CalledProcessError):
            proxy.run(["/usr/bin/false"])
        self.assertIsNone(ascend_npuir._local.capture)


class TestEnvFlags(_EnvSandbox):
    def test_enable_flag_parsing(self):
        for value in ("1", "true", "TRUE", "yes", "on", " On "):
            os.environ["TRITONPARSE_ENABLE_NPUIR_PASS_DUMP"] = value
            self.assertTrue(npuir_pass_dump_enabled(), value)
        for value in ("0", "false", "no", "off", ""):
            os.environ["TRITONPARSE_ENABLE_NPUIR_PASS_DUMP"] = value
            self.assertFalse(npuir_pass_dump_enabled(), value)
        os.environ.pop("TRITONPARSE_ENABLE_NPUIR_PASS_DUMP", None)
        self.assertFalse(npuir_pass_dump_enabled())


class TestEnableInstallation(_EnvSandbox):
    """`enable_npuir_pass_dump` installs into whatever backend is importable."""

    def tearDown(self):
        super().tearDown()
        try:
            from tritonparse.ascend_npuir import _reset_for_tests

            _reset_for_tests()
        except Exception:
            pass

    def test_missing_ascend_backend_is_not_an_error(self):
        """On a CUDA-only box the feature must degrade, not raise."""
        with patch.dict(
            "sys.modules", {"triton.backends.ascend": None, "triton": None}
        ):
            self.assertFalse(enable_npuir_pass_dump())

    def test_install_is_idempotent_and_restorable(self):
        from tritonparse.ascend_npuir import _reset_for_tests

        if not enable_npuir_pass_dump():
            self.skipTest("Ascend backend not importable in this environment")

        from triton.backends.ascend import compiler as ascend_compiler

        proxy = ascend_compiler.subprocess
        self.assertIsInstance(proxy, _SubprocessProxy)

        # Second call is a no-op, not a second layer of wrapping.
        self.assertTrue(enable_npuir_pass_dump())
        self.assertIs(ascend_compiler.subprocess, proxy)

        _reset_for_tests()
        self.assertNotIsInstance(ascend_compiler.subprocess, _SubprocessProxy)

    def test_refactored_backend_is_reported_not_silently_ignored(self):
        """A renamed entry point must warn, not produce a pass-less trace."""
        import types

        from tritonparse.ascend_npuir import _reset_for_tests

        _reset_for_tests()
        import triton.backends.ascend as ascend_pkg

        fake = types.ModuleType("triton.backends.ascend.compiler")
        fake.subprocess = subprocess
        # Deliberately no linalg_to_bin_enable_npu_compile_* functions.
        # Patch the attribute on the parent package too: `from X import Y` prefers
        # getattr(X, "Y") over sys.modules["X.Y"].
        with (
            patch.object(ascend_pkg, "compiler", fake),
            self.assertLogs("tritonparse", level="WARNING") as captured,
        ):
            self.assertFalse(enable_npuir_pass_dump())
        self.assertTrue(
            any("could not be installed" in line for line in captured.output),
            captured.output,
        )

    def test_install_wraps_both_compile_entry_points(self):
        if not enable_npuir_pass_dump():
            self.skipTest("Ascend backend not importable in this environment")

        from triton.backends.ascend import compiler as ascend_compiler

        for name in (
            "linalg_to_bin_enable_npu_compile_910_95",
            "linalg_to_bin_enable_npu_compile_A2_A3",
        ):
            fn = getattr(ascend_compiler, name)
            self.assertTrue(
                getattr(fn, "_tritonparse_npuir_wrapped", False),
                f"{name} was not wrapped",
            )
            self.assertTrue(callable(getattr(fn, "__wrapped__", None)))


if __name__ == "__main__":
    unittest.main()
