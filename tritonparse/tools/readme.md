The tool scripts in this folder are used separately. They are not part of the main tritonparse functionality.

## Available Tools

### extract_irs.py
Extract IR (Intermediate Representation) files from NDJSON trace logs.

**Usage:**
```bash
python extract_irs.py -i <input.ndjson> --line <line_number> -o <output_folder>
```

**Arguments:**
- `-i, --input`: Path to the input NDJSON file
- `--line`: Line number to extract (0-based indexing, where 0 = first line)
- `-o, --output`: Output directory to save extracted IR files
- `--kernel-name`: (Optional) Custom kernel name for output files

**Examples:**
```bash
# Extract IRs from the first line (line 0)
python extract_irs.py -i logs.ndjson --line 0 -o output_folder

# Extract from line 5
python extract_irs.py -i logs.ndjson --line 5 -o ./irs

# Specify custom kernel name
python extract_irs.py -i logs.ndjson --line 0 -o ./irs --kernel-name my_kernel
```

**Extracted Files:**
- `*.ttir` - Triton IR
- `*.ttgir` - Triton GPU IR
- `*.llir` - LLVM IR
- `*.ptx` - PTX assembly
- `*.json` - Kernel metadata
- `*_source.py` - Python source code (if available)

### ascend_npuir_capture.py
Run a Triton kernel on a real Ascend NPU and capture every bishengir NPUIR
compiler pass into a TritonParse trace. Self-diagnosing: `--check` reports which
of the required pieces (NPU, `torch_npu`, `bishengir-compile`, a triton with
`knobs`) is missing.

**Usage:**
```bash
python -m tritonparse.tools.ascend_npuir_capture --check
python -m tritonparse.tools.ascend_npuir_capture --out ./npuir_demo
python -m tritonparse.tools.ascend_npuir_capture --filter hivm-graph-sync-solver,hivm-inject-sync
```

**Arguments:**
- `--out`: Output directory (default `./npuir_demo`)
- `--filter`: Comma-separated pass arguments to capture (default: all)
- `--size`, `--block-size`: Workload shape
- `--check`: Only verify the environment, then exit

See [docs/11.-Ascend-NPUIR-Pass-Capture.md](../docs/11.-Ascend-NPUIR-Pass-Capture.md).

### demo_npuir_pass_capture.py
Same feature, but with the NPU faked so it runs on any machine (it drives a real
`bishengir-compile` and the real parse pipeline).

**Usage:**
```bash
export PATH=/path/to/bishengir/bin:$PATH
python -m tritonparse.tools.demo_npuir_pass_capture --out /tmp/npuir_demo
```

### check_npuir_trace.py
Diagnose why NPUIR pass stages show no source locations in the web UI. Several
distinct failures all look the same from the UI ("no loc"), and they need
different fixes; this tells them apart from the trace alone.

**Usage:**
```bash
python -m tritonparse.tools.check_npuir_trace ./parsed/*.ndjson.gz
python -m tritonparse.tools.check_npuir_trace ./logs        # parses first
```

It reports how many pass artifacts contain `loc(...)`, how many have a
`source_mappings` entry, and a verdict:
- *locations captured and mapped* — nothing to do
- *no locations in the IR* — the dump was made without `--mlir-print-debuginfo`
  (regenerate with a current TritonParse)
- *locations present but not mapped* — a TritonParse parsing bug; report it
