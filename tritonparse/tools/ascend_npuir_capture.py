#!/usr/bin/env python3
#  Copyright (c) Meta Platforms, Inc. and affiliates.
"""Run a Triton kernel on an Ascend NPU and capture bishengir NPUIR passes.

This is the on-device counterpart to
``tritonparse.tools.demo_npuir_pass_capture`` (which fakes the NPU so it can run
anywhere). Everything here is real: a real kernel on a real device, a real
``bishengir-compile`` subprocess, and the real TritonParse parse step.

This ships inside the installed package, so no repo checkout is needed::

    # On the Ascend server, in the environment that runs your kernels:
    python -m tritonparse.tools.ascend_npuir_capture --out ./npuir_demo

    # Only the sync-related passes (much smaller trace):
    python -m tritonparse.tools.ascend_npuir_capture --filter hivm-graph-sync-solver,hivm-inject-sync

Then open the printed URL, or copy ``<out>/parsed/*.ndjson.gz`` to a machine with
a browser and load it at https://meta-pytorch.org/tritonparse/.

Requires: an Ascend NPU, torch_npu, a triton-ascend with ``triton.knobs``
(daily/3.6+; see ``--check``), and ``bishengir-compile`` on ``PATH``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

# The capture must be opted into before TritonParse is initialised, because
# init() is what installs the hook into the Ascend backend.
os.environ.setdefault("TRITONPARSE_ENABLE_NPUIR_PASS_DUMP", "1")


def _fail(message: str) -> None:
    # Flush stdout first: the environment report goes there, and without this the
    # error can appear above the lines it refers to when output is piped.
    sys.stdout.flush()
    print(f"\n[ERROR] {message}\n", file=sys.stderr)
    sys.exit(1)


def check_environment() -> None:
    """Report the pieces the capture depends on, before doing any work."""
    print("== environment ==")

    try:
        import triton
    except ImportError as e:
        _fail(f"triton is not importable: {e}")

    print(f"  triton          : {triton.__version__}  ({triton.__file__})")

    # TritonParse hooks compilation through knobs.compilation.listener. Older
    # triton-ascend builds (the 3.2 lineage) have no knobs module, so nothing
    # would be traced at all -- not just NPUIR passes.
    try:
        from triton import knobs

        knobs.compilation  # noqa: B018
        print("  triton.knobs    : present")
    except Exception as e:
        _fail(
            "this triton has no usable `knobs.compilation` "
            f"({type(e).__name__}: {e}).\n"
            "        TritonParse traces compilations through it, so no trace "
            "would be written.\n"
            "        Use a triton-ascend build that ships triton/knobs.py "
            "(daily/3.6+), or put\n"
            "        the built source tree on PYTHONPATH:\n"
            "          PYTHONPATH=/path/to/triton-ascend/python python "
            "tritonparse.tools.ascend_npuir_capture ..."
        )

    try:
        import triton.backends.ascend.compiler as ascend_compiler

        print(f"  ascend backend  : {ascend_compiler.__file__}")
    except Exception as e:
        _fail(f"the Ascend backend is not importable: {e}")

    if shutil.which("bishengir-compile") is None:
        _fail(
            "bishengir-compile is not on PATH; source the CANN/BiSheng "
            "environment first."
        )
    print(f"  bishengir       : {shutil.which('bishengir-compile')}")

    try:
        import torch  # noqa: F401
        import torch_npu  # noqa: F401
    except ImportError as e:
        _fail(f"torch/torch_npu is not importable: {e}")

    import torch

    if not torch.npu.is_available():
        _fail(
            "no Ascend NPU is visible to torch_npu (torch.npu.is_available() is False)."
        )
    print(f"  npu             : {torch.npu.get_device_name(0)}")
    print()


def build_kernel():
    """A vector add, with a second reduction kernel to give the passes more to do."""
    import triton
    import triton.language as tl

    @triton.jit
    def add_kernel(
        x_ptr,
        y_ptr,
        out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x + y, mask=mask)

    return add_kernel


def run_workload(size: int, block_size: int):
    """Run the kernel once and verify it against a torch reference."""
    import torch
    import triton

    add_kernel = build_kernel()

    x = torch.randn(size, device="npu", dtype=torch.float32)
    y = torch.randn(size, device="npu", dtype=torch.float32)
    out = torch.empty_like(x)

    grid = (triton.cdiv(size, block_size),)
    add_kernel[grid](x, y, out, size, BLOCK_SIZE=block_size)
    torch.npu.synchronize()

    reference = x + y
    ok = torch.allclose(out, reference, atol=1e-5, rtol=1e-5)
    max_err = (out - reference).abs().max().item()
    print(f"  kernel result   : {'PASS' if ok else 'FAIL'} (max abs err {max_err:.3e})")
    if not ok:
        _fail("the kernel produced a wrong result; fix that before reading the IR.")
    return add_kernel


def report_stages(parsed_dir: str, filter_value: str | None) -> None:
    """Print what the trace actually contains."""
    import glob
    import gzip
    import json

    files = [
        p
        for p in glob.glob(os.path.join(parsed_dir, "**", "*.ndjson*"), recursive=True)
        if not p.endswith("log_file_list.json")
    ]
    if not files:
        _fail(f"no parsed trace found under {parsed_dir}")

    print("== parsed trace ==")
    for path in files:
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                event = json.loads(line)
                if event.get("event_type") != "compilation":
                    continue
                payload = event["payload"]
                stages = payload.get("ir_stages", [])
                passes = [s for s in stages if s["name"].startswith("npuir_pass_")]
                mappings = payload.get("source_mappings", {})

                mapped = [s for s in passes if mappings.get(s["name"])]
                print(f"  trace           : {os.path.basename(path)}")
                print(f"  total stages    : {len(stages)}")
                print(f"  NPUIR passes    : {len(passes)}")
                print(f"  mapped to source: {len(mapped)}/{len(passes)} passes")
                if passes and len(mapped) < len(passes) * 0.9:
                    print(
                        "    ^ low: the dumps carry few loc(...) annotations, so "
                        "most passes\n"
                        "      cannot be tied back to a source line. Check that "
                        "--mlir-print-debuginfo\n"
                        "      reaches bishengir."
                    )

                if not passes:
                    print(
                        "\n  No NPUIR pass stages. Check the "
                        "'Ascend NPUIR pass dump enabled' line above; if it is\n"
                        "  missing, the hook did not install. Re-run with "
                        "--check to diagnose."
                    )
                    return

                print("\n  first passes (in execution order):")
                for stage in passes[:6]:
                    print(
                        f"    {stage['display_order']:>5}  "
                        f"{stage['display_name']:<46} {stage['name']}"
                    )

                # The sync passes are usually why someone turns this on.
                interesting = [
                    s
                    for s in passes
                    if any(
                        k in s["name"]
                        for k in ("sync", "split_mix", "graph_sync", "inject")
                    )
                ]
                if interesting:
                    print("\n  sync/split-related passes (compare these side by side):")
                    for stage in interesting:
                        print(
                            f"    {stage['display_order']:>5}  "
                            f"{stage['display_name']:<46} {stage['name']}"
                        )

                if filter_value:
                    print(f"\n  (TRITONPARSE_NPUIR_PASS_FILTER={filter_value})")
                return


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a Triton kernel on Ascend NPU and capture NPUIR passes."
    )
    parser.add_argument("--out", default="./npuir_demo", help="output directory")
    parser.add_argument(
        "--filter",
        default=None,
        help="comma-separated pass arguments to capture (default: all)",
    )
    parser.add_argument("--size", type=int, default=1 << 20, help="vector length")
    parser.add_argument("--block-size", type=int, default=1024, help="tile size")
    parser.add_argument(
        "--check",
        action="store_true",
        help="only check the environment, then exit",
    )
    args = parser.parse_args()

    if args.filter:
        os.environ["TRITONPARSE_NPUIR_PASS_FILTER"] = args.filter

    check_environment()
    if args.check:
        print("Environment looks usable. Re-run without --check to capture.")
        return

    out_root = os.path.abspath(args.out)
    logs_dir = os.path.join(out_root, "logs")
    parsed_dir = os.path.join(out_root, "parsed")
    if os.path.exists(out_root):
        shutil.rmtree(out_root)
    os.makedirs(logs_dir)

    # Initialising TritonParse installs both the compilation listener and (when
    # TRITONPARSE_ENABLE_NPUIR_PASS_DUMP is set) the NPUIR pass-dump hook.
    import tritonparse.structured_logging

    tritonparse.structured_logging.init(logs_dir)
    print(f"  trace folder    : {logs_dir}\n")

    print("== run ==")
    run_workload(args.size, args.block_size)

    print("\n== parse ==")
    import tritonparse.parse.utils

    tritonparse.parse.utils.unified_parse(logs_dir, out=parsed_dir)

    print("\n== result ==")
    report_stages(parsed_dir, args.filter)

    print(
        "\nView it:\n"
        f"  cd {parsed_dir} && python -m http.server 8000\n"
        "  # then, from your laptop: ssh -N -L 8000:localhost:8000 <user>@<server>\n"
        "  # open: http://localhost:8000/standalone.html?json_url="
        "http://localhost:8000/<trace>.ndjson.gz\n"
        "  # (or just scp the .ndjson.gz out and load it in the web UI)\n"
    )


if __name__ == "__main__":
    main()
