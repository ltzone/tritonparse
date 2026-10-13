#!/usr/bin/env python3
"""End-to-end demo of Ascend NPUIR pass capture (no NPU device required).

Runs against a STOCK triton-ascend (no patch): the capture is installed by
``tritonparse.ascend_npuir`` at runtime. It drives the *real* Ascend compile path
with a *real* ``bishengir-compile`` subprocess, then runs the *real* TritonParse
trace/parse pipeline over the result and reports what the user would see.

Usage:
    export PATH=/path/to/bishengir/bin:$PATH
    python -m tritonparse.tools.demo_npuir_pass_capture [--out /tmp/npuir_demo]

The only thing faked is the NPU itself: bishengir's late-pipeline failure (no
hivmc/npuc in a compile-only container) is swallowed and a placeholder object
file is written so the rest of the pipeline proceeds.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TRITON_ADAPTER = Path(
    Path(__file__).resolve().parent.parent.parent
    / "tests/cpu/fixtures/npuir/kernel.ttadapter"
)


class _Metadata(dict):
    def __missing__(self, key):
        return None


class _Target:
    arch = "Ascend910B1"


class _Options:
    debug = False
    mix_mode = "aiv"


def compile_with_npuir_dump(ttadapter: str, metadata: dict):
    """Run the Ascend compile path with bishengir really invoked.

    The NPUIR hook is already installed by the time this runs (see ``main``); we
    only wrap the global ``subprocess.run`` so bishengir's late-pipeline failure
    on a compile-only container does not abort the demo. The hook's own proxy
    rewrites the command line and captures stderr *before* delegating here, so
    the printout below is the command bishengir actually received.
    """
    from triton.backends.ascend import compiler as ascend_compiler

    seen = {}

    def run_and_tolerate(cmd_list, **kwargs):
        cmd_list = list(cmd_list)
        if "--help" in cmd_list:
            # The capability probe; let it through untouched.
            return _real_run(cmd_list, **kwargs)
        seen["cmd"] = cmd_list
        print(f"[demo] bishengir-compile {' '.join(cmd_list[1:])}")
        kwargs["check"] = False
        proc = _real_run(cmd_list, **kwargs)
        out_path = cmd_list[cmd_list.index("-o") + 1]
        for candidate in (out_path + ".o", out_path + "_reloc.o"):
            Path(candidate).write_text("placeholder object")
        return proc

    # The hook's proxy resolves ``self._real.run`` at call time, so patching the
    # module-level ``run`` is enough and the rewrite still happens first.
    _real_run = subprocess.run
    subprocess.run = run_and_tolerate
    try:
        ascend_compiler.linalg_to_bin_enable_npu_compile_910_95(
            ttadapter, metadata, _Options()
        )
    finally:
        subprocess.run = _real_run
    return seen.get("cmd")


def build_trace_event(metadata: dict, metadata_group: dict):
    """Reproduce exactly what the compilation listener records."""
    from collections import defaultdict

    from tritonparse.structured_logging import (
        _attach_npuir_pass_log,
        extract_file_content,
    )

    trace_data = defaultdict(dict)
    trace_data["metadata"].update(metadata)
    extract_file_content(trace_data, metadata_group, metadata.get("backend_name", ""))
    _attach_npuir_pass_log(trace_data, metadata)
    # The real listener adds these via trace_structured_triton; include them so
    # the demo trace validates against the compilation schema.
    from datetime import datetime, timezone

    return {
        "event_type": "compilation",
        "pid": os.getpid(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": trace_data,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/npuir_demo")
    parser.add_argument("--ttadapter", default=str(TRITON_ADAPTER))
    args = parser.parse_args()

    if shutil.which("bishengir-compile") is None:
        sys.exit("bishengir-compile not on PATH; source the CANN/BiSheng env first")

    # Opt in exactly as a real user would, then install the runtime capture into
    # the (stock) Ascend backend -- the same thing
    # ``tritonparse.structured_logging.init()`` does.
    os.environ["TRITONPARSE_ENABLE_NPUIR_PASS_DUMP"] = "1"
    from tritonparse.ascend_npuir import enable_npuir_pass_dump

    if not enable_npuir_pass_dump():
        sys.exit("Ascend backend not importable; cannot run this demo")

    out_root = Path(args.out)
    if out_root.exists():
        shutil.rmtree(out_root)
    logs_dir = out_root / "logs"
    logs_dir.mkdir(parents=True)

    with tempfile.TemporaryDirectory() as workdir:
        dump_dir = os.path.join(workdir, "dump")
        os.environ["TRITON_DUMP_DIR"] = dump_dir

        ttadapter = Path(args.ttadapter).read_text()
        metadata = _Metadata(
            {
                # A real sha256 hex digest: triton derives the dump
                # directory name from it via base64.
                "hash": "0f1e2d3c4b5a69788796a5b4c3d2e1f0" * 2,
                "backend_name": "cann",
                "target": _Target(),
                "name": "kernel",
                "kernel_name": "kernel",
                "enable_auto_bind_sub_block": None,
                "auto_tile_and_bind_sub_block": True,
                "required_ub_bits": 0,
                "mix_mode": "aiv",
                "parallel_mode": "simd",
                "bitcodes": None,
                "num_warps": 4,
            }
        )

        print("== step 1: compile with NPUIR pass dump enabled ==")
        compile_with_npuir_dump(ttadapter, metadata)

        dump_path = metadata.get("npuir_passes_log_path")
        print(f"[demo] dump recorded at: {dump_path}")
        assert dump_path and os.path.exists(dump_path), "no NPUIR dump was produced"
        print(f"[demo] dump size: {os.path.getsize(dump_path)} bytes")

        # Simulate the triton dump directory contents that the listener reads.
        hash_dir = os.path.join(dump_dir, metadata["hash"])
        os.makedirs(hash_dir, exist_ok=True)
        (Path(hash_dir) / "kernel.ttir").write_text(
            "module {\n  tt.func public @kernel() {\n    tt.return\n  }\n}\n"
        )
        (Path(hash_dir) / "kernel.ttadapter").write_text(ttadapter)
        (Path(hash_dir) / "kernel.npubin").write_bytes(b"\x00binary")
        metadata_group = {
            "kernel.ttir": str(Path(hash_dir) / "kernel.ttir"),
            "kernel.ttadapter": str(Path(hash_dir) / "kernel.ttadapter"),
            "kernel.npubin": str(Path(hash_dir) / "kernel.npubin"),
        }

        print("== step 2: build the compilation trace event ==")
        event = build_trace_event(metadata, metadata_group)
        file_content = event["payload"]["file_content"]
        passes = [k for k in file_content if k.startswith("kernel.npuir_pass_")]
        print(f"[demo] file_content entries: {len(file_content)}")
        print(f"[demo] npuir pass artifacts: {len(passes)}")
        print(f"[demo] raw dump embedded: {'kernel.npuir_passes.log' in file_content}")

        trace_file = logs_dir / "dedicated_log_triton_trace_demo_.ndjson"
        with open(trace_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")
        print(f"[demo] wrote raw trace: {trace_file}")

    print("== step 3: parse with the tritonparse CLI ==")
    out_dir = out_root / "parsed"
    cmd = [
        sys.executable,
        "-m",
        "tritonparse",
        "parse",
        str(logs_dir),
        "--out",
        str(out_dir),
        "--overwrite",
    ]
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    proc = subprocess.run(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    print(proc.stdout.decode("utf-8", errors="replace")[-3000:])
    if proc.returncode != 0:
        sys.exit(f"parse failed with {proc.returncode}")

    print("== step 4: inspect the parsed trace ==")
    parsed_files = sorted(out_dir.rglob("*.ndjson*"))
    print(f"[demo] parsed files: {[p.name for p in parsed_files]}")
    for parsed in parsed_files:
        if parsed.name == "log_file_list.json":
            continue
        import gzip

        opener = gzip.open if parsed.name.endswith(".gz") else open
        with opener(parsed, "rt", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("event_type") != "compilation":
                    continue
                payload = entry["payload"]
                stages = payload.get("ir_stages", [])
                npuir_stages = [
                    s for s in stages if s["name"].startswith("npuir_pass_")
                ]
                mapped_passes = [
                    s
                    for s in npuir_stages
                    if payload.get("source_mappings", {}).get(s["name"])
                ]
                print(f"[demo] total stages       : {len(stages)}")
                print(f"[demo] npuir pass stages  : {len(npuir_stages)}")
                print(
                    f"[demo] mapped to source   : {len(mapped_passes)}/"
                    f"{len(npuir_stages)} passes"
                )
                if npuir_stages and len(mapped_passes) < len(npuir_stages) * 0.9:
                    print(
                        "         ^ low: the dumps carry few loc(...) annotations.\n"
                        "           Check that --mlir-print-debuginfo was passed "
                        "to bishengir."
                    )
                print("[demo] first 5 npuir stages:")
                for s in npuir_stages[:5]:
                    print(
                        f"         {s['display_order']:>5}  {s['name']:<45} "
                        f"{s['display_name']!r}"
                    )
                print("[demo] last 3 stages (should be npubin/json):")
                for s in sorted(stages, key=lambda s: s["display_order"])[-3:]:
                    print(f"         {s['display_order']:>5}  {s['name']}")
                break


if __name__ == "__main__":
    main()
