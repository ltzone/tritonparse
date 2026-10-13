# Bishengir NPUIR Pass-Level IR Capture for Ascend NPU

**Proposal**: 0001 - NPUIR Pass-Level IR Capture  
**Status**: ✅ Implemented (see [Implementation Notes](README.md#-implementation-notes) for
where the implementation differs from this proposal)  
**Target**: TritonParse Enhancement

> This document is the original design proposal, preserved for context. For the
> as-built behaviour, read the implementation notes in [`README.md`](README.md)
> and the user guide at [`docs/11.-Ascend-NPUIR-Pass-Capture.md`](../docs/11.-Ascend-NPUIR-Pass-Capture.md).
> Key deltas from the proposal:
>
> 1. **No triton-ascend patch is required.** The capture is installed at runtime
>    by `tritonparse/ascend_npuir.py`, so stock/daily triton-ascend packages work
>    as-is. The patch in `patches/` is optional and only for upstreaming.
> 2. `--mlir-print-ir-after-failure` must be *replaced* rather than combined (it
>    silences all after-pass printing, including `--mlir-print-ir-after-all`).
> 3. Stage names carry no `.mlir` suffix.
> 4. Real runs emit ~400 dumps covering ~134 distinct passes.
> 5. The raw dump is embedded into the trace rather than left on disk.

---

## 📋 Executive Summary

Add **bishengir pass-level IR capture** for Ascend NPU, exposing the NPUIR transformation passes that run inside the `bishengir-compile` subprocess during TTAdapter→NPUBIN lowering.

### Key Discovery

Bishengir **already supports** dumping all passes via the standard MLIR flag:
```bash
--mlir-print-ir-after-all
```

**No upstream changes needed!** The feature is implementable entirely within TritonParse + triton-ascend's `compiler.py`.

### What Gets Captured

Currently:
```
TTIR → TTAdapter → [bishengir black box] → NPUBIN
```

After this feature:
```
TTAdapter (Linalg IR)
  ↓ bishengir-compile subprocess
  ├─ 401 After convert-linalg-to-hfusion
  ├─ 402 After convert-hfusion-to-hivm
  ├─ 403 After hivm-graph-sync-solver
  ├─ 404 After hivm-split-mix-kernel
  ├─ 405 After hivm-inject-sync
  ├─ 406 After convert-hivm-to-standard
  ├─ ... (~20-30 more passes)
  ↓
NPUBIN (kernel.o)
```

---

## 🎯 Motivation

### Current State

The Ascend compilation has two stages:

1. ✅ **Python-side MLIR** (TTIR → TTAdapter): 8 passes, in-process — *already accessible via `MLIR_ENABLE_DUMP=1` if needed*
2. ❌ **Bishengir subprocess** (TTAdapter → NPUBIN): ~20-30 passes, only 1 visible

Today, `bishengir-compile` runs ~20-30 HIVM/HFusion passes but only outputs:
```python
# compiler.py line 689
_compile_option_list += ["--bishengir-print-ir-after=hivm-graph-sync-solver"]
```

This captures only one pass (`hivm-graph-sync-solver`) as `kernel.npuir.mlir`. All other passes are invisible.

### Why This Matters

**Real debugging scenarios**:

1. **Synchronization bugs**: Compare `hivm-graph-sync-solver` vs. `hivm-inject-sync` outputs
2. **Kernel splitting**: Inspect `hivm-split-mix-kernel` transformations
3. **Lowering issues**: Track `convert-hivm-to-standard` step by step
4. **Memory problems**: See buffer allocation pass results
5. **Performance**: Correlate pass changes with codegen quality

### User Pain Point (From Migration Guide)

Current workflow:
```bash
# Step 1: Get ttadapter file
export TRITON_DEBUG=1
python kernel.py  # produces ~/.triton/cache/xxx.ttadapter

# Step 2: Manual bishengir-compile for each pass
bishengir-compile xxx.ttadapter --target=Ascend910B3 \
  --bishengir-print-ir-after=hivm-inject-sync

bishengir-compile xxx.ttadapter --target=Ascend910B3 \
  --bishengir-print-ir-after=hivm-split-mix-kernel
...
```

**Problem**: Manual, tedious, not integrated into traces.

---

## 🏗️ Architecture Design

### 1. Mechanism

**Key discovery**: Bishengir supports **standard MLIR printing flags**:

```bash
$ /path/to/bishengir-compile --help | grep mlir-print-ir-after
  --mlir-print-ir-after=<pass-arg>       - Print IR after specified passes
  --mlir-print-ir-after-all              - Print IR after each pass  ← THIS!
  --mlir-print-ir-after-change           - Only print if IR changed
  --mlir-print-ir-after-failure          - Only print on pass failure
```

**Verified working**:
```bash
$ bishengir-opt test.mlir --canonicalize --mlir-print-ir-after-all
// -----// IR Dump After Canonicalizer (canonicalize) //----- //
module { ... }
```

**No whitelist restrictions** — the whitelist only applies to `--bishengir-print-ir-*`, not `--mlir-print-ir-*`.

### 2. Integration Flow

```
┌────────────────────────────────────────────────────────────┐
│ TRITONPARSE_ENABLE_NPUIR_PASS_DUMP=1                       │
└───────────────────────┬────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────────┐
│ triton-ascend compiler.py: linalg_to_bin()                │
│  • Add --mlir-print-ir-after-all to bishengir-compile cmd │
│  • Capture subprocess stdout/stderr to npuir_passes.log   │
│  • Save to dump_manager                                    │
└───────────────────────┬────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────────┐
│ bishengir-compile subprocess                               │
│  • Runs ~20-30 passes                                      │
│  • Prints IR to stdout after each pass                     │
│  • Format: "// -----// IR Dump After <PassName> //----- //"│
└───────────────────────┬────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────────┐
│ TritonParse: _split_bishengir_dump_log()                  │
│  • Read npuir_passes.log                                   │
│  • Split on "// -----// IR Dump After" delimiter          │
│  • Extract pass name from each section                     │
│  • Inject as file_content["kernel.npuir_pass_<name>.mlir"]│
└───────────────────────┬────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────────┐
│ Trace: ir_stages with NPUIR pass stages (401-499)         │
└───────────────────────┬────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────────┐
│ Frontend: data-driven UI (no changes needed)               │
└────────────────────────────────────────────────────────────┘
```

### 3. Display Order Allocation

```
TTIR (100)
TTAdapter (200)
MLIRBC (300)
BCMLIR (400)
  ├─ 401 convert-linalg-to-hfusion
  ├─ 402 convert-hfusion-to-hivm
  ├─ 403 hivm-graph-sync-solver
  ├─ 404 hivm-split-mix-kernel
  ├─ 405 hivm-inject-sync
  ├─ 406 convert-hivm-to-standard
  ├─ ... (allocate 401-499, 99 slots for bishengir passes)
NPUBIN (500)
JSON (1000)
```

---

## 🔧 Implementation Details

### 1. Environment Variables

```bash
# TritonParse-owned
export TRITONPARSE_ENABLE_NPUIR_PASS_DUMP="1"  # opt-in
export TRITONPARSE_NPUIR_PASS_FILTER="hivm-graph-sync-solver,hivm-inject-sync"  # optional

# Existing (for .ttadapter file)
export TRITON_DEBUG="1"
```

### 2. Triton-Ascend Changes (compiler.py)

**Location**: `linalg_to_bin_enable_npu_compile_*()` functions (lines ~690, ~915)

**Current code**:
```python
if opt.debug or os.getenv("TRITON_PRINT_AUTOTUNING", None) == "1":
    _compile_option_list += ["--bishengir-print-ir-after=hivm-graph-sync-solver"]
```

**Proposed code**:
```python
if os.getenv("TRITONPARSE_ENABLE_NPUIR_PASS_DUMP") == "1":
    # Use standard MLIR flag to dump all passes
    _compile_option_list += ["--mlir-print-ir-after-all"]
    
    # Optional: filter specific passes
    pass_filter = os.getenv("TRITONPARSE_NPUIR_PASS_FILTER")
    if pass_filter:
        # Override: only dump specified passes
        _compile_option_list.remove("--mlir-print-ir-after-all")
        for pass_name in pass_filter.split(","):
            _compile_option_list += [f"--mlir-print-ir-after={pass_name}"]
elif opt.debug or os.getenv("TRITON_PRINT_AUTOTUNING", None) == "1":
    # Existing behavior: only hivm-graph-sync-solver
    _compile_option_list += ["--bishengir-print-ir-after=hivm-graph-sync-solver"]
```

**Capture subprocess output**:
```python
# After subprocess.run(...), before current _save_npuir_debug_output:
if os.getenv("TRITONPARSE_ENABLE_NPUIR_PASS_DUMP") == "1":
    # Save full stdout+stderr with all pass dumps
    npuir_log_path = os.path.join(tmpdir, "npuir_passes.log")
    with open(npuir_log_path, 'wb') as f:
        f.write(ret.stdout if ret.stdout else b'')
        f.write(ret.stderr if ret.stderr else b'')
    
    # Copy to dump manager
    dump_manager = get_dump_manager(metadata["hash"])
    with open(npuir_log_path, 'r', encoding='utf-8', errors='replace') as f:
        dump_manager.put(f.read(), "npuir_passes.log", binary=False)
```

**Estimated changes**: ~30 lines in `compiler.py`

### 3. TritonParse: Log Splitting

```python
import re
from typing import List, Tuple

_BISHENGIR_DUMP_HEADER_RE = re.compile(
    r"^// -----// IR Dump After (.+?) \((.+?)\) //----- //\s*$"
)

def _split_bishengir_dump_log(log_path: str) -> List[Tuple[str, str, str]]:
    """Split bishengir IR dump log into (display_name, pass_arg, ir) tuples.
    
    Args:
        log_path: Path to npuir_passes.log
    
    Returns:
        List of (display_name, pass_arg, ir_content) tuples
        display_name: e.g., "Canonicalizer"
        pass_arg: e.g., "canonicalize"
        ir_content: Full MLIR module text
    
    Example log format:
        // -----// IR Dump After Canonicalizer (canonicalize) //----- //
        module { ... }
        // -----// IR Dump After CSE (cse) //----- //
        module { ... }
    """
    if not log_path or not os.path.exists(log_path):
        return []
    
    records, display_name, pass_arg, buf = [], None, None, []
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _BISHENGIR_DUMP_HEADER_RE.match(line)
            if m:
                # Save previous section
                if display_name is not None and buf:
                    records.append((display_name, pass_arg, "".join(buf)))
                display_name, pass_arg, buf = m.group(1), m.group(2), []
            elif display_name is not None:
                buf.append(line)
    
    # Save final section
    if display_name is not None and buf:
        records.append((display_name, pass_arg, "".join(buf)))
    
    return records


def _inject_npuir_pass_irs(payload: dict, adapter, cache_dir: str) -> None:
    """Inject bishengir pass IRs into compilation payload."""
    if os.getenv("TRITONPARSE_ENABLE_NPUIR_PASS_DUMP") != "1":
        return
    
    log_path = os.path.join(cache_dir, "npuir_passes.log")
    records = _split_bishengir_dump_log(log_path)
    
    enabled_names = None
    if filter_str := os.getenv("TRITONPARSE_NPUIR_PASS_FILTER"):
        enabled_names = set(filter_str.split(","))
    
    seq_by_arg = {}
    for display_name, pass_arg, ir in records:
        if enabled_names and pass_arg not in enabled_names:
            continue
        
        # Handle repeated passes (e.g., canonicalize runs multiple times)
        seq_by_arg[pass_arg] = seq_by_arg.get(pass_arg, 0) + 1
        suffix = f"_{seq_by_arg[pass_arg]}" if seq_by_arg[pass_arg] > 1 else ""
        
        stage_name = f"npuir_pass_{pass_arg.replace('-', '_')}{suffix}"
        key = f"kernel.{stage_name}.mlir"
        payload["file_content"][key] = ir
```

**Estimated changes**: ~80 lines in `trace_processor.py`

### 4. Backend: Stage Registration

```python
# In backend.py: AscendTritonAdapter

# TODO: Extract full list from bishengir/include/*/Passes.td
BISHENGIR_PASS_STAGES = [
    # Conversion passes (pass_arg, display_label, order)
    ("convert-linalg-to-hfusion", "Convert Linalg to HFusion", 401),
    ("convert-hfusion-to-hivm", "Convert HFusion to HIVM", 402),
    
    # HIVM optimization passes
    ("hivm-graph-sync-solver", "HIVM Graph Sync Solver", 403),
    ("hivm-split-mix-kernel", "HIVM Split Mix Kernel", 404),
    ("hivm-inject-sync", "HIVM Inject Sync", 405),
    ("hivm-inject-block-sync", "HIVM Inject Block Sync", 406),
    
    # Lowering
    ("convert-hivm-to-std", "Convert HIVM to Standard", 407),
    
    # TODO: Add remaining ~20 passes from Passes.td analysis
]

def _register_npuir_pass_stages(self):
    """Dynamically register NPUIR pass stages when enabled."""
    if os.getenv("TRITONPARSE_ENABLE_NPUIR_PASS_DUMP") != "1":
        return
    
    for pass_arg, display_label, order in BISHENGIR_PASS_STAGES:
        stage_name = f"npuir_pass_{pass_arg.replace('-', '_')}"
        if not self.get_stage_by_name(stage_name):
            self._stages.append(IRStageDescriptor(
                name=stage_name,
                extension=f".{stage_name}.mlir",
                display_name=f"NPUIR Pass: {display_label}",
                display_order=order,
                is_text=True,
                supports_source_mapping=True,  # HIVM has #loc annotations
                parser_id="generic_loc",
                syntax_id="mlir",
            ))
    
    self._stages.sort(key=lambda s: s.display_order)
```

**Estimated changes**: ~50 lines in `backend.py`

---

## 📊 Impact Analysis

### No Upstream Dependency

✅ **Zero blocking dependencies** — uses existing `--mlir-print-ir-after-all` flag

### Changes Required

| Component | Owner | Lines | Risk |
|-----------|-------|-------|------|
| `compiler.py` | Triton-Ascend | ~30 | Low (add flag + capture output) |
| `trace_processor.py` | TritonParse | ~80 | Low (log parsing) |
| `backend.py` | TritonParse | ~50 | Low (stage registration) |
| **Total** | | **~160** | **Low** |

### Trace File Size Impact

| Configuration | Files | Size | Multiplier |
|--------------|-------|------|------------|
| Baseline | 5 | 120 KB | 1.0× |
| + 20 NPUIR passes | 25 | ~500 KB | 4.2× |
| With .gz | 25 | 100-200 KB | 0.8-1.7× |

### Performance Impact

| Stage | Impact |
|-------|--------|
| Compilation | +10-20% (IR printing in bishengir) |
| Trace parsing | +100-150% (more IR) |
| UI load | +20-30% |

**Mitigation**: Opt-in via `TRITONPARSE_ENABLE_NPUIR_PASS_DUMP=1`

---

## 🚀 Implementation Roadmap

### Phase 1: Core Implementation (Week 1-2)

- [ ] Modify `compiler.py` to add `--mlir-print-ir-after-all` and capture output
- [ ] Implement `_split_bishengir_dump_log()` in `trace_processor.py`
- [ ] Implement `_inject_npuir_pass_irs()`
- [ ] Unit tests for log parsing

### Phase 2: Stage Registration (Week 2)

- [ ] Extract full pass list from `Passes.td` files
- [ ] Build `BISHENGIR_PASS_STAGES` table
- [ ] Implement `_register_npuir_pass_stages()` in `backend.py`
- [ ] Integration tests

### Phase 3: Documentation (Week 3)

- [ ] User guide: debugging NPUIR passes
- [ ] Pass catalog (what each pass does)
- [ ] Example workflows
- [ ] Environment variable reference

**Total: 3-4 weeks from start to merge**

---

## 📋 Open Questions

### Q1: Which Passes to Register?

**Options**:
- A: All ~50 passes from Passes.td (complete but verbose)
- B: Only commonly-used ~15 passes (cleaner UI)
- C: Dynamic discovery from log (flexible but complex)

**Recommendation**: Start with B (common passes), expand to A incrementally

### Q2: Filtering Strategy

**User wants**: Only see specific passes (e.g., sync-related)

**Options**:
- A: `TRITONPARSE_NPUIR_PASS_FILTER="pass1,pass2"` (proposed)
- B: UI-side filtering (post-capture)
- C: Both

**Recommendation**: A (reduces trace size at source)

### Q3: Repeated Passes

Some passes run multiple times (e.g., `canonicalize` ×3). How to name?
- Current: `npuir_pass_canonicalize_2`, `npuir_pass_canonicalize_3`
- Alternative: `npuir_pass_canonicalize[2]`

**Recommendation**: Current (simpler for stage name parsing)

---

## 🎓 References

- **Bishengir source**: `third_party/ascend/AscendNPU-IR/bishengir`
- **Pass definitions**: `bishengir/include/bishengir/*/Passes.td`
- **Triton-ascend compiler**: `python/triton/backends/ascend/compiler.py`
- **Test evidence**: `/home/litao/AscendNPU-IR/build/bin/bishengir-opt` with `--mlir-print-ir-after-all` verified working

---

## 📝 Revision History

| Date | Version | Changes |
|------|---------|---------|
| — | 0.1 | Initial (WRONG: assumed whitelist blocked all passes) |
| — | 0.2 | **Corrected**: Discovered `--mlir-print-ir-after-all` works, no upstream dependency |

---

**Status**: ✅ Implemented  
**Next Step**: None — see [`docs/11.-Ascend-NPUIR-Pass-Capture.md`](../docs/11.-Ascend-NPUIR-Pass-Capture.md)
