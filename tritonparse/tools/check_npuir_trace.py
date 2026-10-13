#!/usr/bin/env python3
#  Copyright (c) Meta Platforms, Inc. and affiliates.
"""Diagnose why NPUIR pass stages show no source locations in the web UI.

Feeding a trace to the frontend has several failure modes that all look the
same ("no loc"), and they need different fixes:

* the dump was captured without ``--mlir-print-debuginfo``, so the IR text has
  no ``loc(...)`` at all -- regenerate the trace with a current TritonParse;
* the IR has locations but the trace has no ``source_mappings`` entry for the
  stage -- a parser/naming regression, report it;
* both are present and the frontend still shows nothing -- the mapping's file
  path does not match the recorded Python source path, which is a display
  problem, not a capture one.

This script tells those apart from a single trace file.

Usage::

    python -m tritonparse.tools.check_npuir_trace <trace.ndjson[.gz]> [...]
    python -m tritonparse.tools.check_npuir_trace <raw_log_dir>   # parses first
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
import sys
from typing import Any, Dict, Iterator, List

# Real MLIR locations are `loc("...")` or `loc(#locN)`; `memref.alloc(` must not
# match, which a naive `loc\(` does.
_LOC_RE = re.compile(r'loc\(["#]')
_PASS_FILE_RE = re.compile(r"kernel\.(npuir_pass_[^.]*)$")

GREEN, YELLOW, RED, DIM, RESET = (
    "\033[32m",
    "\033[33m",
    "\033[31m",
    "\033[2m",
    "\033[0m",
)


def _iter_events(path: str) -> Iterator[Dict[str, Any]]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _looks_parsed(path: str) -> bool:
    """A parsed trace has been through trace_processor; a raw one has not."""
    for event in _iter_events(path):
        if event.get("event_type") == "compilation":
            payload = event.get("payload", {})
            return "ir_stages" in payload
        break
    return False


def _sources(args: List[str]) -> List[str]:
    """Expand arguments to concrete trace files, parsing a raw dir if needed."""
    out: List[str] = []
    for arg in args:
        if os.path.isdir(arg):
            raw = sorted(
                glob.glob(os.path.join(arg, "*.ndjson"))
                + glob.glob(os.path.join(arg, "*.ndjson.gz"))
            )
            if not raw:
                print(f"{RED}no .ndjson files under {arg}{RESET}")
                continue
            print(f"{DIM}parsing {arg} first...{RESET}")
            import tempfile

            import tritonparse.parse.utils

            tmp = tempfile.mkdtemp(prefix="npuir_check_")
            tritonparse.parse.utils.unified_parse(arg, out=tmp, skip_logger=True)
            out.extend(
                sorted(glob.glob(os.path.join(tmp, "**", "*.ndjson*"), recursive=True))
            )
        else:
            out.extend(sorted(glob.glob(arg)))
    return [p for p in out if os.path.isfile(p)]


def _report(path: str) -> int:
    """Return 0 when source mapping is available, 1 otherwise."""
    print(f"\n== {path} ==")
    if not _looks_parsed(path):
        print(
            f"  {YELLOW}this looks like a RAW log, not a parsed trace.{RESET}\n"
            f"  Run `tritonparse parse` on it first (or pass the log directory)."
        )
        return 1

    checked = 0
    worst = 0
    for event in _iter_events(path):
        if event.get("event_type") != "compilation":
            continue
        payload = event.get("payload", {})
        file_content = payload.get("file_content", {})
        mappings = payload.get("source_mappings", {})
        stages = {s.get("name") for s in payload.get("ir_stages", [])}

        pass_files = {
            name: content
            for name, content in file_content.items()
            if _PASS_FILE_RE.search(name)
        }
        if not pass_files:
            continue
        checked += 1

        with_loc = [n for n, c in pass_files.items() if _LOC_RE.search(c)]
        stage_name = lambda n: _PASS_FILE_RE.search(n).group(1)  # noqa: E731
        with_map = [n for n in pass_files if mappings.get(stage_name(n))]
        unresolvable = [n for n in pass_files if stage_name(n) not in stages]

        n = len(pass_files)
        print(f"  NPUIR pass artifacts      : {n}")
        print(
            f"    IR contains loc(...)    : {len(with_loc)}"
            f"{GREEN if len(with_loc) >= n * 0.9 else RED}  "
            f"(expect ~all){RESET}"
        )
        print(
            f"    has source_mappings     : {len(with_map)}"
            f"{GREEN if len(with_map) >= n * 0.9 else RED}  "
            f"(expect ~all){RESET}"
        )
        if unresolvable:
            print(
                f"    {RED}{len(unresolvable)} artifacts match no ir_stage "
                f"(UI will not label them){RESET}"
            )
        print(f"  source_mappings top keys  : {len(mappings)}  (expect ~{n + 3})")

        # The upstream artifacts matter as much as the passes: bishengir can only
        # print locations that its input still has. ttir usually keeps them, so
        # ttir-without-ttadapter is the signature of a lowering bug that no
        # bishengir flag can work around.
        upstream = {
            name: content
            for name, content in file_content.items()
            if isinstance(content, str)
            and not _PASS_FILE_RE.search(name)
            and name.rsplit(".", 1)[-1] in ("ttir", "ttadapter", "bcmlir", "mlirbc")
        }
        if upstream:
            print("  upstream IR (bishengir's input is ttadapter/bcmlir):")
            for name, content in sorted(upstream.items()):
                mark = GREEN if _LOC_RE.search(content) else RED
                print(
                    f"    {name:<28} loc(...) = {len(_LOC_RE.findall(content))}"
                    f"{mark}{RESET}"
                )

        # A sample, so the file/line can be eyeballed.
        sample = next(iter(pass_files))
        entry = mappings.get(stage_name(sample)) or {}
        if entry:
            first = next(iter(entry.values()))
            print(
                f"  sample: {stage_name(sample)}"
                f"\n    IR line -> {first.get('file')}:{first.get('line')}:"
                f"{first.get('column')}"
            )

        if len(with_loc) < n * 0.9:
            ttir = next((c for k, c in upstream.items() if k.endswith(".ttir")), "")
            lowered = next(
                (
                    c
                    for k, c in upstream.items()
                    if k.endswith((".ttadapter", ".bcmlir"))
                ),
                "",
            )
            print(f"\n  {RED}VERDICT: no locations in the IR.{RESET}")
            if _LOC_RE.search(ttir) and not _LOC_RE.search(lowered):
                # bishengir can only print locations its input still carries, so
                # this is not fixable from TritonParse.
                print(
                    "  ttir has locations but ttadapter/bcmlir do not. bishengir can\n"
                    "  only print what its input still has, so no dump flag can\n"
                    "  recover them -- the triton-ascend lowering for this kernel\n"
                    "  dropped them. Upgrading TritonParse is still needed (it adds\n"
                    "  --mlir-print-debuginfo) but will not help here; report the\n"
                    "  lowering to triton-ascend. Meanwhile ttir IS mappable, so use\n"
                    "  it for source-level correlation and the NPUIR passes for\n"
                    "  structural diffing."
                )
            else:
                print(
                    "  The dump was produced without --mlir-print-debuginfo, so no\n"
                    "  pass can map to a source line. TritonParse adds that flag\n"
                    "  itself; a trace with this shape was made by an older install\n"
                    "  or a patched triton-ascend that supplies its own flags.\n"
                    "  Fix: upgrade TritonParse and regenerate the trace.\n"
                    "    python -m tritonparse.tools.check_npuir_trace <trace>   # "
                    "after reinstalling"
                )
            worst = 1
        elif len(with_map) < n * 0.9:
            # Locations that never reach a .py file are the parser's own
            # positions in the temporary kernel.mlir, i.e. the real source
            # locations were stripped upstream of bishengir.
            py_re = re.compile(r'"([^"]*\.py)"')
            py_locs = sum(len(py_re.findall(c)) for c in pass_files.values())
            if py_locs == 0:
                print(
                    f"\n  {RED}VERDICT: only parse positions, no source "
                    f"locations.{RESET}\n"
                    "  The dumps carry loc(...), but every one points at the\n"
                    "  temporary input file (e.g. kernel.mlir), not at a .py\n"
                    "  source. The triton-ascend lowering for this kernel\n"
                    "  dropped the real locations before bishengir ran, so no\n"
                    "  NPUIR pass can be mapped. Check whether the ttadapter\n"
                    "  artifact in this trace has loc(...); if it does not while\n"
                    "  ttir does, that is the regression to report upstream.\n"
                    "  TritonParse deliberately drops these mappings rather than\n"
                    "  pointing you at a deleted temp file."
                )
            else:
                print(
                    f"\n  {RED}VERDICT: locations are present but not mapped."
                    f"{RESET}\n"
                    "  This is a TritonParse parsing bug, not a capture problem --\n"
                    "  please report it with this trace."
                )
            worst = 1
        else:
            py = mappings.get("python", {})
            print(
                f"\n  {GREEN}VERDICT: locations captured and mapped.{RESET}\n"
                f"  python_source lines referenced: {len(py)}"
            )
            if not py:
                print(
                    f"  {YELLOW}but the Python mapping is empty; the trace may "
                    f"lack python_source.{RESET}"
                )

    if checked == 0:
        print(
            f"  {YELLOW}no compilation event with NPUIR pass artifacts found.{RESET}\n"
            "  Either the trace has no NPUIR capture (was "
            "TRITONPARSE_ENABLE_NPUIR_PASS_DUMP set when the kernel ran?),"
            "\n  or it is not an Ascend NPU trace."
        )
        return 1
    return worst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", help="trace file(s) or a log directory")
    args = parser.parse_args()

    files = _sources(args.traces)
    if not files:
        sys.exit("no trace files found")

    status = 0
    for path in files:
        if path.endswith("log_file_list.json"):
            continue
        status |= _report(path)
    sys.exit(status)


if __name__ == "__main__":
    main()
