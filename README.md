# nvidia-uma-fault-probe

Ground-truth measurement of NVIDIA Unified Memory behavior.

Low-level probes for fault latency, memory bandwidth, and atomic
coherence cost using PTX instrumentation inside the kernel.

All metrics are captured with `%clock64` and PTX cache/atomic scope
operators — not CUPTI callbacks or NVML polling.

Validated on discrete PCIe (Pascal SM 6.1). Designed for architectures
through Blackwell GB10 (SM 12.1), covering both discrete PCIe and
hardware-coherent UMA platforms.

Written in C and PTX. No dependencies beyond CUDA.
Engineers share JSON output for remote analysis.

---

## Tools

### uma_probe — UMA Fault Latency Probe

Measures cycle-accurate page fault latency using PTX `%clock64`
before and after `ld.global.cv` on unified memory pages.

Three passes expose the full UMA behavior profile:

| Pass     | Setup                      | Measures                        |
|----------|----------------------------|---------------------------------|
| COLD     | CPU touches all pages      | Hardware fault + migration cost |
| WARM     | GPU prefetch before launch | Resident access latency         |
| PRESSURE | Mixed CPU/GPU residency    | Thrash latency                  |

The COLD/WARM ratio is the key signal:

| Platform                      | Ratio   | Meaning                                        |
|-------------------------------|---------|------------------------------------------------|
| Discrete PCIe (Pascal to Ada) | ~1.0x   | No hardware fault visible at instruction level |
| Hardware-coherent UMA (GB10)  | 20-100x | True hardware migration cost                   |

---

### uma_bw — UMA Bandwidth Test

Measures achieved memory bandwidth using PTX cache operators:

- `ld.global.cg` — cache at L2, bypass L1 (read)
- `st.global.cs` — bypass L2, true DRAM write bandwidth

Tests GPU read, GPU write, GPU copy, CPU read, CPU write,
and concurrent CPU+GPU access to the same memory pool.

Peak bandwidth is derived from hardware attributes at runtime.
On GB10, memory clock is not exposed by the driver; peak is
reported as 0 rather than fabricated.

On GB10 the concurrent test measures Grace CPU and GB10 GPU
accessing the same LPDDR5X pool simultaneously.

---

### uma_atomic — UMA Atomic Coherence Probe

Measures cycle-accurate latency of atomic operations at GPU scope
vs system scope on unified managed memory.

Three passes:

| Pass       | PTX Operation             | Measures                              |
|------------|---------------------------|---------------------------------------|
| GPU-scope  | `atom.global.gpu.add.u32` | Atomic latency within GPU memory      |
| SYS-scope  | `atom.global.sys.add.u32` | Atomic latency through coherence path |
| CONTENTION | `atom.global.sys` + CPU   | True concurrent access cost           |

The SYS/GPU ratio is the key signal:

| Platform                      | Ratio  | Meaning                          |
|-------------------------------|--------|----------------------------------|
| Discrete PCIe (Pascal to Ada) | ~1.0x  | No coherence protocol            |
| Hardware-coherent UMA (GB10)  | > 1.0x | NVLink-C2C coherence overhead    |

On discrete GPUs, gpu-scope and sys-scope atomics typically have similar cost,
as there is no system-level CPU–GPU coherence protocol to traverse.
On GB10, sys-scope atomics must coordinate with the Grace CPU through NVLink-C2C.
The latency delta reflects coherence protocol overhead — not publicly quantified.

---

## Why PTX

PTX (Parallel Thread Execution) is NVIDIA's virtual machine assembly.
Targeting `compute_60` allows driver JIT compilation into native SASS
for whatever GPU is present at runtime.

- One binary runs from Pascal through GB10
- `%clock64` measures cycles inside the kernel, no driver overhead
- Cache operators `.cg` and `.cs` control DRAM path behavior
- Atomic scope operators `.gpu` and `.sys` expose coherence behavior
- Ground truth from inside the kernel — not from callbacks

[NVIDIA PTX ISA Reference](https://docs.nvidia.com/cuda/parallel-thread-execution/)

---

## Build

Requirements: CUDA 12.x or 13.x, C++17, Linux (x86_64 or aarch64)

### uma_probe

```bash
# x86_64:
nvcc -O2 -std=c++17 probe_launcher.cu -o uma_probe -lcudart -lcuda

# aarch64 (GB10 DGX Spark):
nvcc -O2 -std=c++17 probe_launcher.cu -o uma_probe -lcudart -lcuda -lpthread
```

### uma_bw

```bash
# x86_64:
nvcc -O2 -std=c++17 uma_bandwidth_test.cu -o uma_bw -lcudart

# aarch64 (GB10 DGX Spark):
nvcc -O2 -std=c++17 uma_bandwidth_test.cu -o uma_bw -lcudart -lpthread
```

### uma_atomic

```bash
# x86_64:
nvcc -O2 -std=c++17 uma_atomic_test.cu -o uma_atomic -lcudart -lcuda

# aarch64 (GB10 DGX Spark):
nvcc -O2 -std=c++17 uma_atomic_test.cu -o uma_atomic -lcudart -lcuda -lpthread
```

---

## Run

```bash
./uma_probe              # human-readable output + JSON log
./uma_probe --json-only  # JSON only

./uma_bw                 # human-readable output + JSON log
./uma_bw --json-only     # JSON only

./uma_atomic             # human-readable output + JSON log
./uma_atomic --json-only # JSON only
```

`uma_fault_probe.ptx` must remain in the same directory as `uma_probe`.
`uma_atomic_probe.ptx` must remain in the same directory as `uma_atomic`.

---

## Example Output — Pascal GTX 1080 (SM 6.1, validated)

**uma_probe:**
```
GPU      : NVIDIA GeForce GTX 1080 (SM 6.1)
Platform : DISCRETE_PCIE
COLD  p50:     32.9 ns  (57 cycles)
WARM  p50:     32.9 ns  (57 cycles)
COLD/WARM ratio: 1.00x
Expected : ratio ~1.0x (no HW fault visible on discrete GPU)
```

**uma_bw:**
```
GPU      : NVIDIA GeForce GTX 1080 (SM 6.1)
Platform : DISCRETE_PCIE
GPU read  : 248.91 GB/s  stddev 0.07
GPU write : 254.16 GB/s  stddev 0.03  [PTX .cs]
GPU copy  :   7.69 GB/s
CPU read  :   5.23 GB/s  (PCIe bottleneck)
CPU write :  22.69 GB/s
Peak      : 320.32 GB/s  (derived from hardware: 256-bit bus, 5005 MHz)
```

Pascal note: PTX `.cs` (cache streaming, L2 bypass) is a hint, not a guarantee
on SM 6.x. On Volta+ (SM 7.0+) `.cs` produces true DRAM write bandwidth.
On GB10 (SM 12.1) write results will reflect true LPDDR5X bandwidth.

**uma_atomic:**
```
GPU      : NVIDIA GeForce GTX 1080 (SM 6.1)
Platform : DISCRETE_PCIE
GPU-scope p50 :    176.0 ns  ( 305 cycles) [atom.global.gpu]
SYS-scope p50 :    177.1 ns  ( 307 cycles) [atom.global.sys]
CONTENTION p50:    180.6 ns  ( 313 cycles) [sys + CPU concurrent]
SYS/GPU ratio : 1.01x
Coherence cost: 1.2 ns overhead
Expected : ratio ~1.0x (no coherence protocol on discrete GPU)
```

---

## GB10 / DGX Spark — Community Data Needed

Developed from shared logs, sosreports, and PTX documentation.
GB10 hardware access is not available to the author.

If you have a DGX Spark, run all three tools and share the JSON
output via the Issues page.

https://github.com/parallelArchitect/nvidia-uma-fault-probe/issues

Expected GB10 values based on architecture:

| Tool       | Metric          | Expected                                              |
|------------|-----------------|-------------------------------------------------------|
| uma_probe  | COLD/WARM ratio | 20-100x                                               |
| uma_bw     | GPU read        | community data needed (memory clock N/A from driver)  |
| uma_bw     | CPU read        | 50-100 GB/s (native Grace CPU access, not PCIe)       |
| uma_atomic | SYS/GPU ratio   | > 1.0x (NVLink-C2C coherence cost, unquantified)      |

---

## Relationship to cuda-unified-memory-analyzer

This repository complements cuda-unified-memory-analyzer:
https://github.com/parallelArchitect/cuda-unified-memory-analyzer

| Tool                         | Measures                                | Method                      |
|------------------------------|-----------------------------------------|-----------------------------|
| uma_probe                    | Fault latency (ns)                      | PTX %clock64 + ld.global.cv |
| uma_bw                       | Bandwidth (GB/s)                        | PTX .cg/.cs + CUDA events   |
| uma_atomic                   | Atomic coherence latency (ns)           | PTX %clock64 + atom.global  |

---

## Supported Architectures

| Architecture               | SM       | uma_probe | uma_bw    | uma_atomic |
|----------------------------|----------|-----------|-----------|------------|
| Pascal                     | 6.0, 6.1 | validated | validated | validated  |
| Volta                      | 7.0      | expected  | expected  | expected   |
| Turing                     | 7.5      | expected  | expected  | expected   |
| Ampere                     | 8.0, 8.6 | expected  | expected  | expected   |
| Ada Lovelace               | 8.9      | expected  | expected  | expected   |
| Hopper                     | 9.0      | expected  | expected  | expected   |
| Blackwell GB10 (DGX Spark) | 12.1     | pending   | pending   | pending    |
| Blackwell GB202 (RTX 5090) | 12.0     | pending   | pending   | pending    |

---

## Author

Joe McLaren (parallelArchitect)
Human-directed GPU engineering with AI assistance.

Contact: gpu.validation@gmail.com
https://github.com/parallelArchitect

## License

MIT
