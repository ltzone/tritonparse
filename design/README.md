# Design Documents Summary

This directory contains the design specification for **Bishengir NPUIR Pass-Level IR Capture** for TritonParse on Ascend NPU.

> **Status: implemented, and no longer needs a triton-ascend patch.** The
> capture is installed into a *stock* triton-ascend at runtime by
> `tritonparse/ascend_npuir.py`, so daily/stable triton-ascend packages work
> as-is. The design below is kept as the original proposal; where the
> implementation differs, see [Implementation Notes](#-implementation-notes).
> User-facing documentation lives in
> [`docs/11.-Ascend-NPUIR-Pass-Capture.md`](../docs/11.-Ascend-NPUIR-Pass-Capture.md).
> The equivalent triton-ascend patch is kept in
> [`patches/`](patches) for upstreaming only — it is optional.

## Quick Navigation

| Document | Purpose | Target Audience |
|----------|---------|----------------|
| **[Main Design Doc](0001-npuir-pass-level-ir-capture.md)** | Complete technical specification | All reviewers |
| **[README](README.md)** | Quick overview + navigation (this file) | Everyone |

---

## Design at a Glance

### What is Being Added?

Fine-grained IR capture of **bishengir NPUIR transformation passes** that run inside the `bishengir-compile` subprocess during TTAdapter→NPUBIN lowering.

**Before**:
```
TTIR (100) → TTAdapter (200) → [bishengir black box] → NPUBIN (500)
```

**After (with `TRITONPARSE_ENABLE_NPUIR_PASS_DUMP=1`)**:
```
TTIR (100) → TTAdapter (200)
                     ↓ bishengir-compile subprocess
                     ├─ 401 After convert-linalg-to-hfusion
                     ├─ 402 After convert-hfusion-to-hivm
                     ├─ 403 After hivm-graph-sync-solver
                     ├─ 404 After hivm-split-mix-kernel
                     ├─ 405 After hivm-inject-sync
                     ├─ 406 After convert-hivm-to-standard
                     ├─ ... (~20-30 bishengir passes)
                     ↓
                  NPUBIN (500)
```

**Scope**: Captures bishengir NPUIR passes (401-499) that run in the `bishengir-compile` subprocess after TTAdapter.

---

## 🎉 Key Discovery: No Upstream Dependency!

Bishengir **already supports** dumping all passes via standard MLIR flag:

```bash
bishengir-compile input.ttadapter --mlir-print-ir-after-all --target=...
```

**Verified working**:
```bash
$ bishengir-opt test.mlir --canonicalize --mlir-print-ir-after-all
// -----// IR Dump After Canonicalizer (canonicalize) //----- //
module { ... }
```

**Whitelist limitation does NOT apply** — the whitelist only restricts `--bishengir-print-ir-*`, not standard MLIR's `--mlir-print-ir-*` flags.

**Result**: Feature is **implementable immediately** with ~160 lines of code, no waiting on upstream!

---

## Use Cases

1. **Synchronization Debugging**: Compare `hivm-graph-sync-solver` vs. `hivm-inject-sync` outputs
2. **Kernel Splitting**: Inspect `hivm-split-mix-kernel` transformations
3. **Lowering Issues**: Track `convert-hivm-to-standard` step by step
4. **Memory Problems**: Debug buffer allocation passes
5. **Performance Analysis**: Correlate pass changes with codegen quality

---

## Implementation Roadmap

| Phase | Work | Duration | Status |
|-------|------|----------|--------|
| **Phase 1** | Capture flag + subprocess output | 1-2 weeks | ✅ Done (as a runtime hook in TritonParse; the triton-ascend patch is optional) |
| Phase 2 | TritonParse: log splitting + injection | 1 week | ✅ Done |
| Phase 3 | Documentation + examples | 1 week | ✅ Done |

**Blocker**: None! ✅

---

## Design Process

This directory follows a lightweight RFC process:

1. ✅ **Proposal** — Draft design doc with motivation, architecture, plan
2. ✅ **Validation** — Test bishengir-compile to verify mechanism works
3. 🔜 **Discussion** — Open GitHub issue, gather feedback
4. 🔜 **Implementation** — Land focused PRs
5. ⏸️ **Completion** — Mark implemented with release version

### Current Status

- ✅ **Proposal complete**: [0001-npuir-pass-level-ir-capture.md](0001-npuir-pass-level-ir-capture.md)
- ✅ **Mechanism validated**: `--mlir-print-ir-after-all` tested and working
- ✅ **Implemented** in TritonParse and triton-ascend; verified end-to-end against
  a real `bishengir-compile` run (398 pass dumps from one kernel)

---

## Key Design Decisions

### 1. Use Standard MLIR Flag (Not Bishengir-Specific)

**Rationale**: `--mlir-print-ir-after-all` works for all passes, no whitelist restriction.

**Alternative rejected**: `--bishengir-print-ir-after=<pass>` only works for 3 whitelisted passes.

### 2. Capture at Subprocess Level (Not In-Process)

**Rationale**: Bishengir passes run in a separate subprocess (`bishengir-compile`), not Python.

**Scope**: This design captures **bishengir subprocess** passes (TTAdapter→NPUBIN). Python-side MLIR passes (TTIR→TTAdapter, accessible via `MLIR_ENABLE_DUMP=1`) are a separate, independent feature if needed.

### 3. Opt-In via Environment Variable

**Rationale**: 
- Large trace size impact (4-5× uncompressed)
- Compilation overhead (~10-20%)
- Not needed for most users

**API**: `TRITONPARSE_ENABLE_NPUIR_PASS_DUMP=1`

### 4. Subprocess Output Capture

**Rationale**: `--mlir-print-ir-after-all` prints to stderr. Must capture subprocess stderr, parse it, and inject into trace.

**Implementation**: Modify `compiler.py` to save `subprocess.run()` stderr to `npuir_passes.log`.

---

## Technical Summary

### What Changes

| File | Changes | Lines |
|------|---------|-------|
| `triton/backends/ascend/compiler.py` | Add `--mlir-print-ir-after-all`, capture stderr | ~30 |
| `tritonparse/parse/trace_processor.py` | Parse log, inject pass IRs | ~80 |
| `tritonparse/backend.py` | Register NPUIR pass stages | ~50 |
| **Total** | | **~160** |

### What Users See

New environment variables:
```bash
export TRITONPARSE_ENABLE_NPUIR_PASS_DUMP="1"  # Enable feature
export TRITONPARSE_NPUIR_PASS_FILTER="hivm-graph-sync-solver,hivm-inject-sync"  # Optional filter
```

New stages in UI:
- `NPUIR Pass: Convert Linalg to HFusion` (401)
- `NPUIR Pass: HIVM Graph Sync Solver` (403)
- `NPUIR Pass: HIVM Inject Sync` (405)
- ... (~20-30 total)

---

## References

- **Main design doc**: [0001-npuir-pass-level-ir-capture.md](0001-npuir-pass-level-ir-capture.md)
- **Triton-ascend source**: `/path/to/triton-ascend/python/triton/backends/ascend/compiler.py`
- **Bishengir source**: `/path/to/triton-ascend/third_party/ascend/AscendNPU-IR/bishengir`
- **Test evidence**: `bishengir-opt --mlir-print-ir-after-all` verified working

---

**Status**: ✅ **Implemented**  
**See also**: [`docs/11.-Ascend-NPUIR-Pass-Capture.md`](../docs/11.-Ascend-NPUIR-Pass-Capture.md)  
**Questions?** See [main design doc](0001-npuir-pass-level-ir-capture.md) or open an issue.

---

## 📐 Implementation Notes

The following corrections and additions were discovered while implementing the
proposal above.

### 0. No triton-ascend patch is required

The proposal assumed Phase 1 had to land in triton-ascend's `compiler.py`. It does
not. `tritonparse/ascend_npuir.py` installs the whole capture at runtime:

* wraps `linalg_to_bin_enable_npu_compile_910_95` and `..._A2_A3` to get at the
  `metadata` dict,
* proxies the module-level `subprocess` name in
  `triton/backends/ascend/compiler.py` to rewrite the bishengir command line and
  capture its output.

Consequences:

* **Stock triton-ascend works**, including daily/stable packages — no rebuild, no
  patch, no fork.
* The hook is installed only when `TRITONPARSE_ENABLE_NPUIR_PASS_DUMP` is set,
  only once per process, and only touches the Ascend compiler's own `subprocess`
  binding (never the global module). With the feature off, the compile path is
  unchanged.
* The triton-ascend patch in `patches/` is retained for upstreaming only. Both
  mechanisms are safe together: the runtime hook detects flags already present
  and does not duplicate them.

### 1. `--mlir-print-ir-after-failure` must be *replaced*, not combined

MLIR gates every after-pass print on "did the pass fail?" once
`--mlir-print-ir-after-failure` is present, which silences
`--mlir-print-ir-after-all` **and** `--mlir-print-ir-after=<pass>` completely.
Verified against a real toolchain:

| Flags | Pass dumps |
|-------|-----------|
| `--mlir-print-ir-after-all` | 397 |
| `--mlir-print-ir-after-all --mlir-print-ir-after-failure` | 0 |
| `--mlir-print-ir-after=hivm-graph-sync-solver` | 3 |
| `--mlir-print-ir-after=hivm-graph-sync-solver --mlir-print-ir-after-failure` | 0 |

triton-ascend passes `--mlir-print-ir-after-failure` unconditionally, so the
implementation drops it while the dump is enabled. Nothing is lost: after-all
prints after every pass, including the failing one.

### 1b. `--mlir-print-debuginfo` is required for source mapping

MLIR stores a location on every operation but does not *print* it unless
`OpPrintingFlags::printDebugInfo` is set, and it defaults to off. So
`--mlir-print-ir-after-all` alone yields dumps that contain **zero** `loc(...)`
annotations, and no pass can be mapped back to a source line.

This was missed initially and produced a subtly green test: the only `loc(...)`
in a loc-free dump comes from bishengir's own *error messages*, and the parser
happily matched those. The fix is one flag; the fixture now carries ~6.8k
locations and ~390 of ~400 passes map to source. It costs ~1.7x raw dump size
(1.5 MB -> 2.6 MB) that gzip almost entirely removes.

The lesson worth keeping: assert the *rate* of mapping, not that one example
mapped, or a false positive passes as success.

### 2. Stage naming has no `.mlir` suffix

Artifacts are `kernel.<stage_name>` with `stage_name = npuir_pass_<sanitized>`.
The web UI recovers a stage name from a filename with `split(".").pop()`, so a
`.mlir` suffix would collapse every pass to the name `mlir` and silently drop
source mappings and MLIR highlighting. This matches `.ttadapter` / `.bcmlir`.

### 3. Display order is int, and the range is much wider than 401-499

A real run emitted 398 dumps covering 134 distinct passes; NPUBIN/JSON were moved
to 10000/20000 so they stay last. Ordering stays monotonic in execution order
rather than being bucketed into the notional 401-499 window.

### 4. `is_pass_level` keeps the frontend heuristics sane

With 100+ extra viewable stages, the frontend's "middle of the pipeline"
grouping anchor and default-panel selection would be swamped. Pass stages carry
`is_pass_level: true` and are excluded from those two heuristics (but still fully
selectable).

### 5. Pairwise cross-stage mappings are skipped for pass stages

`parse_single_trace_content` builds bidirectional mappings for every pair of
populated stages. With ~400 pass stages that is ~80k pairs over a multi-hundred
line module — billions of comparisons. Pass stages are excluded from the pairwise
step and keep only their own source mapping (which is what the Python-source
highlight needs).

### 6. The raw dump is embedded in the trace, not left on disk

The proposal captured the subprocess output to a temp file. The implementation
also writes it into Triton's dump directory and records the absolute path in
`metadata["npuir_passes_log_path"]`, so `structured_logging` can embed the dump
into the trace event. That makes the trace self-contained: parsing works on a
different machine, and cache hits still carry the dump.

### 7. Compiler diagnostics are stripped from the IR

bishengir interleaves `[WARNING]` / `[ERROR]` lines with the dumps, and those
lines land in the buffer of whichever pass was printed last. The splitter drops
lines matching `^\[(ERROR|WARNING|INFO|DEBUG|FATAL)\]`, which MLIR text can never
start with.

### 8. Actual change footprint

| File | Change |
|------|--------|
| `tritonparse/ascend_npuir.py` | ~250 lines (runtime hook; the primary mechanism) |
| `triton-ascend/.../ascend/backend/compiler.py` | ~70 lines (**optional**, for upstreaming only) |
| `tritonparse/parse/trace_processor.py` | ~150 lines (splitter, injection, stage resolution) |
| `tritonparse/backend.py` | ~120 lines (label table, descriptor factory, ordering) |
| `tritonparse/structured_logging.py` | ~60 lines (embed the dump) |
| `website/src/utils/dataLoader.ts` | ~20 lines (`is_pass_level` anchor exclusion) |
| Tests | 3 new modules, 1 checked-in fixture |

### 9. Verification

- `tests/cpu/test_npuir_pass_capture.py` — offline, fixture-driven
- `tests/cpu/test_npuir_runtime_hook.py` — the runtime hook, fully mocked, runs
  everywhere
- `tests/cpu/test_npuir_ascend_contract.py` — drives the real Ascend compile path
  with a faked bishengir, auto-skipped without the Ascend backend; asserts the
  capture works on a **stock** compiler
- `tests/cpu/test_npuir_bishengir_integration.py` — runs the real
  `bishengir-compile`, auto-skipped when it is not on `PATH`
- `tritonparse/tools/demo_npuir_pass_capture.py` — end-to-end demo through the real
  `tritonparse parse` CLI (`python -m tritonparse.tools.demo_npuir_pass_capture`)


