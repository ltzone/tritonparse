#  Copyright (c) Meta Platforms, Inc. and affiliates.
"""Ascend NPUIR pass-level IR capture, applied from TritonParse at runtime.

Capturing bishengir's per-pass IR needs one thing from the compile path: the
``bishengir-compile`` command line must carry ``--mlir-print-ir-after-all``
instead of ``--mlir-print-ir-after-failure``, and the resulting stderr must be
saved somewhere TritonParse can read it.

Rather than requiring a patched triton-ascend, this module installs that
behaviour into the *stock* Ascend backend at runtime:

* a wrapper around the two ``linalg_to_bin_enable_npu_compile_*`` entry points
  gives access to the ``metadata`` dict, so the dump path can be recorded
  exactly as a patched compiler would,
* a proxy over the module-level ``subprocess`` name in
  ``triton.backends.ascend.compiler`` rewrites the bishengir invocation and
  captures its output.

Both are installed once, only when the feature is enabled, and only touch
``triton.backends.ascend.compiler``'s own name binding -- the global
``subprocess`` module is never modified, so no other subprocess call in the
process is affected.

The dump is written to ``<triton dump dir>/<kernel hash>/npuir_passes.log`` and
its path recorded in the compilation metadata under ``npuir_passes_log_path``,
which is the same contract the patched compiler provides; the rest of the
pipeline (``structured_logging`` and the trace parser) is unchanged.
"""

from __future__ import annotations

import functools
import os
import subprocess
import threading
from typing import Any, Dict, List

from tritonparse.tp_logger import logger

# bishengir writes its IR dumps to stderr. Anything else on the command line is
# passed through untouched.
_DUMP_ALL_FLAG = "--mlir-print-ir-after-all"
_FAILURE_FLAG = "--mlir-print-ir-after-failure"

# triton-ascend resolves the compiler to either `bishengir-compile` (modern) or
# `<root>/npuc` (legacy TRITON_NPU_COMPILER_PATH layout).
_COMPILER_BASENAMES = ("bishengir-compile", "npuc")

_ENV_ENABLE = "TRITONPARSE_ENABLE_NPUIR_PASS_DUMP"
_ENV_FILTER = "TRITONPARSE_NPUIR_PASS_FILTER"

# Per-thread capture slot. Compilation may run on several threads (Triton uses a
# thread pool), so the last captured dump must not be shared between them.
_local = threading.local()

# Guarded by _INSTALL_LOCK: installation happens at most once per process.
_install_lock = threading.Lock()
_installed = False


def npuir_pass_dump_enabled() -> bool:
    """Whether the Ascend NPUIR pass dump feature is switched on."""
    return os.getenv(_ENV_ENABLE, "").strip().lower() in ("1", "true", "yes", "on")


def _dump_flags() -> List[str]:
    """The IR-dump flags to pass to bishengir, honouring the optional filter.

    ``--mlir-print-ir-after-change`` is deliberately not used: several sync
    passes (e.g. hivm-inject-sync) are observable precisely when they change the
    IR, and bishengir also mutates the module in place for some passes, so
    "after-all" is the only lossless choice.
    """
    raw = os.getenv(_ENV_FILTER, "")
    names = [name.strip() for name in raw.split(",") if name.strip()]
    if names:
        return [f"--mlir-print-ir-after={name}" for name in names]
    # No filter, or a filter with no usable names (e.g. ","): dump everything
    # rather than silently capturing nothing.
    return [_DUMP_ALL_FLAG]


def _is_bishengir_compile(cmd: Any) -> bool:
    if not cmd or not isinstance(cmd, (list, tuple)) or not cmd:
        return False
    try:
        argv0 = os.path.basename(str(cmd[0]))
    except (TypeError, ValueError):
        return False
    return argv0 in _COMPILER_BASENAMES


def _rewrite_cmd(cmd: List[str]) -> List[str]:
    """Return the bishengir command line with the NPUIR dump flags applied.

    ``--mlir-print-ir-after-failure`` is *removed*, not augmented: MLIR gates
    every after-pass print on "did the pass fail?" once that flag is present,
    which silences ``--mlir-print-ir-after-all`` and
    ``--mlir-print-ir-after=<pass>`` entirely (verified against a real
    toolchain). Nothing is lost -- after-all prints after every pass, including
    the failing one.
    """
    rewritten = [arg for arg in cmd if arg != _FAILURE_FLAG]
    dump_flags = _dump_flags()
    already = any(
        arg == _DUMP_ALL_FLAG or arg.startswith("--mlir-print-ir-after=")
        for arg in rewritten
    )
    if not already:
        # Insert before "-o <file>" so the flags stay grouped with the other
        # compile options rather than trailing the output path.
        try:
            insert_at = rewritten.index("-o")
        except ValueError:
            insert_at = len(rewritten)
        rewritten[insert_at:insert_at] = dump_flags
    return rewritten


class _SubprocessProxy:
    """Stands in for the ``subprocess`` module inside the Ascend compiler.

    Every attribute is forwarded to the real module; only ``run`` is wrapped, and
    only for bishengir invocations.
    """

    def __init__(self, real: Any) -> None:
        self._real = real

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    def run(self, cmd: Any, *args: Any, **kwargs: Any) -> Any:
        if not _is_bishengir_compile(cmd):
            return self._real.run(cmd, *args, **kwargs)

        rewritten = _rewrite_cmd(list(cmd))
        try:
            proc = self._real.run(rewritten, *args, **kwargs)
        except subprocess.CalledProcessError as e:
            # The dump is most valuable when compilation failed: the last pass
            # printed is the IR the failing pass ran on.
            _local.capture = (e.stdout, e.stderr)
            raise
        # getattr rather than attribute access: a caller may have swapped in a
        # custom runner whose result is not a CompletedProcess.
        _local.capture = (
            getattr(proc, "stdout", None),
            getattr(proc, "stderr", None),
        )
        return proc


def _save_dump(metadata: Dict[str, Any]) -> None:
    """Persist the captured bishengir output and record its path in metadata."""
    capture = getattr(_local, "capture", None)
    _local.capture = None
    if not capture:
        return
    stdout_bytes, stderr_bytes = capture
    stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
    stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
    combined = stdout + stderr
    if not combined.strip():
        return

    kernel_hash = metadata.get("hash")
    if not kernel_hash:
        logger.warning("NPUIR pass dump captured but no kernel hash to store it under")
        return

    try:
        from triton.runtime.cache import get_dump_manager

        dump_path = get_dump_manager(kernel_hash).put(
            combined, "npuir_passes.log", binary=False
        )
    except Exception as e:
        # The dump directory is where the trace listener looks for the dump, so
        # there is nothing useful to record if this fails. Leave the metadata key
        # unset and degrade to "no NPUIR passes captured".
        logger.warning(f"Could not persist the NPUIR pass dump: {e}")
        return

    metadata["npuir_passes_log_path"] = dump_path


def _wrap_compile_fn(module: Any, name: str) -> None:
    """Wrap ``module.<name>`` so the dump is saved against its ``metadata`` dict."""
    original = getattr(module, name, None)
    if original is None or getattr(original, "_tritonparse_npuir_wrapped", False):
        return

    @functools.wraps(original)
    def wrapper(linalg: str, metadata: Dict[str, Any], opt: Any) -> Any:
        _local.capture = None
        try:
            return original(linalg, metadata, opt)
        finally:
            _save_dump(metadata)

    wrapper._tritonparse_npuir_wrapped = True  # type: ignore[attr-defined]
    setattr(module, name, wrapper)


def enable_npuir_pass_dump() -> bool:
    """Install the NPUIR pass-dump hooks into the Ascend backend.

    Idempotent. Returns True when the hooks are in place, False when the Ascend
    backend could not be imported (e.g. on a CUDA-only environment), in which
    case the caller should carry on unaffected.
    """
    global _installed
    with _install_lock:
        if _installed:
            return True
        try:
            from triton.backends.ascend import compiler as ascend_compiler
        except Exception as e:
            logger.debug(
                f"Ascend backend not importable; NPUIR pass dump disabled: {e}"
            )
            return False

        # Only bishengir calls are touched, and only via the module-level name in
        # the Ascend compiler -- the global subprocess module is left alone.
        if not isinstance(
            getattr(ascend_compiler, "subprocess", None), _SubprocessProxy
        ):
            ascend_compiler.subprocess = _SubprocessProxy(ascend_compiler.subprocess)

        for fn_name in (
            "linalg_to_bin_enable_npu_compile_910_95",
            "linalg_to_bin_enable_npu_compile_A2_A3",
        ):
            _wrap_compile_fn(ascend_compiler, fn_name)

        _installed = True
        logger.info(
            "Ascend NPUIR pass dump enabled"
            + (
                f" (filter: {os.getenv(_ENV_FILTER)})"
                if os.getenv(_ENV_FILTER, "").strip()
                else " (all passes)"
            )
        )
        return True


def maybe_enable_from_env() -> bool:
    """Install the hooks when ``TRITONPARSE_ENABLE_NPUIR_PASS_DUMP`` is set.

    Called from ``tritonparse.structured_logging.init`` so that enabling the
    feature is a single environment variable, with no code change in the kernel
    script.
    """
    if not npuir_pass_dump_enabled():
        return False
    return enable_npuir_pass_dump()


def _reset_for_tests() -> None:
    """Undo :func:`enable_npuir_pass_dump` (test helper)."""
    global _installed
    with _install_lock:
        try:
            from triton.backends.ascend import compiler as ascend_compiler
        except Exception:
            _installed = False
            return

        if isinstance(getattr(ascend_compiler, "subprocess", None), _SubprocessProxy):
            ascend_compiler.subprocess = ascend_compiler.subprocess._real
        for fn_name in (
            "linalg_to_bin_enable_npu_compile_910_95",
            "linalg_to_bin_enable_npu_compile_A2_A3",
        ):
            fn = getattr(ascend_compiler, fn_name, None)
            if fn is not None and getattr(fn, "_tritonparse_npuir_wrapped", False):
                setattr(ascend_compiler, fn_name, fn.__wrapped__)
        _installed = False
        _local.capture = None
