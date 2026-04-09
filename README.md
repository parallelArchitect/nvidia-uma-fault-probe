# nvidia-uma-fault-probe

Two companion tools for measuring unified memory behavior on NVIDIA GPUs.
Supports NVIDIA architectures from Pascal (SM 6.0) through Blackwell GB10 (SM 12.1).
One binary. No recompile.

Written in C and PTX — no Python, no dependencies beyond CUDA.
Engineers share JSON output for remote analysis.

## Tools

### uma_probe — UMA Fault Latency Probe

Measures cycle-accurate page fault latency using PTX %clock64
before and after ld.global.cv on unified memory pages.

Three passes expose the full UMA behavior profile:

| Pass     | Setup                      | Measures                        |
|----------|----------------------------|---------------------------------|
| COLD     | CPU touches all pages      | Hardware fault + migration cost |
| WARM     | GPU prefetch before launch | Resident access latency         |
| PRESSURE | Mixed CPU/GPU residency    | Thrash latency                  |

The COLD/WARM ratio is the key signal:

| Platform                     | Ratio   | Meaning                                        |
|------------------------------|---------|------------------------------------------------|
| Discrete PCIe (Pascal to Ada)| ~1.0x   | No hardware fault visible at instruction level |
| Hardware-coherent UMA (GB10) | 20-100x | True hardware migration cost                   |

### uma_bw — UMA Bandwidth Test

Measures achieved memory bandwidth using PTX cache operators:

- ld.global.cg — cache at L2, bypass L1 (read)
- st.global.cs — bypass L2, true DRAM write bandwidth

Tests GPU read, GPU write, GPU copy, CPU read, CPU write,
and concurrent CPU+GPU access to the same memory pool.

On GB10 the concurrent test measures Grace CPU and GB10 GPU
accessing the same LPDDR5X pool simultaneously.
This measurement does not exist in any other tool.

## Why PTX

PTX (Parallel Thread Execution) is NVIDIA's virtual machine assembly.
Targeting compute_60 allows driver JIT compilation into native SASS
for whatever GPU is present at runtime.

- One binary runs from Pascal through GB10
- Cache operators .cg and .cs control DRAM path behavior
- %clock64 measures cycles inside the kernel, no driver overhead
- Ground truth from inside the hardware, not from callbacks

Most tools abstract the hardware. These stay close to it.

[NVIDIA PTX ISA Reference](https://docs.nvidia.com/cuda/parallel-thread-execution/)

## Build

Requirements: CUDA 12.x or 13.x, C++17, Linux (x86_64 or aarch64)

### uma_probe — UMA Fault Latency Probe

```bash
# x86_64:
nvcc -O2 -std=c++17 probe_launcher.cu -o uma_probe -lcudart -lcuda

# aarch64 (GB10 DGX Spark):
nvcc -O2 -std=c++17 probe_launcher.cu -o uma_probe -lcudart -lcuda -lpthread
```

### uma_bw — UMA Bandwidth Test

```bash
# x86_64:
nvcc -O2 -std=c++17 uma_bandwidth_test.cu -o uma_bw -lcudart

# aarch64 (GB10 DGX Spark):
nvcc -O2 -std=c++17 uma_bandwidth_test.cu -o uma_bw -lcudart -lpthread
```

## Run

```bash
./uma_probe              — human-readable output + JSON log
./uma_probe --json-only  — JSON only

./uma_bw                 — human-readable output + JSON log
./uma_bw --json-only     — JSON only
```

uma_fault_probe.ptx must remain in the same directory as uma_probe.

## Example Output — Pascal GTX 1080 (SM 6.1, validated)

uma_probe:
  GPU      : NVIDIA GeForce GTX 1080 (SM 6.1)
  Platform : DISCRETE_PCIE
  COLD  p50:     32.9 ns  (57 cycles)
  WARM  p50:     32.9 ns  (57 cycles)
  COLD/WARM ratio: 1.00x
  Expected : ratio ~1.0x (no HW fault visible on discrete GPU)

uma_bw:
  GPU      : NVIDIA GeForce GTX 1080 (SM 6.1)
  Platform : DISCRETE_PCIE
  GPU read  : 248.51 GB/s  stddev 0.06
  GPU write : 251.67 GB/s  (PTX .cs not fully honored on SM6.1 — Pascal behavior)
  GPU copy  :   8.76 GB/s
  CPU read  :   5.18 GB/s  (PCIe bottleneck)
  CPU write :  19.08 GB/s

Pascal note: PTX .cs (cache streaming, L2 bypass) is a hint not a guarantee
on SM6.x. On Volta+ (SM7.0 and above) .cs produces true DRAM write bandwidth.
On GB10 (SM12.1) write results will reflect true LPDDR5X bandwidth.

## GB10 / DGX Spark — Community Data Needed

GB10 validation is in progress. If you have a DGX Spark run both tools
and share the JSON output. It directly improves platform coverage.

Share uma_probe_results.json and uma_bw_results.json at:
https://github.com/parallelArchitect/nvidia-uma-fault-probe/issues

Expected GB10 values based on hardware specification:
- uma_probe COLD/WARM ratio: 20-100x
- uma_bw GPU read: 200-270 GB/s (LPDDR5X peak 273 GB/s)
- uma_bw CPU read: 50-100 GB/s (native Grace CPU access, not PCIe)

## Relationship to cuda-unified-memory-analyzer

This repository complements cuda-unified-memory-analyzer:
https://github.com/parallelArchitect/cuda-unified-memory-analyzer

| Tool                         | Measures                                | Method                        |
|------------------------------|-----------------------------------------|-------------------------------|
| uma_probe                    | Fault latency (ns)                      | PTX %clock64                  |
| uma_bw                       | Bandwidth (GB/s)                        | PTX .cg/.cs + CUDA events     |
| cuda-unified-memory-analyzer | Fault counts, migration bytes, pressure | CUPTI + NVML                  |

Together they provide a complete UMA diagnostic picture.

## Supported Architectures

| Architecture               | SM       | uma_probe | uma_bw    |
|----------------------------|----------|-----------|-----------|
| Pascal                     | 6.0, 6.1 | validated | validated |
| Volta                      | 7.0      | expected  | expected  |
| Turing                     | 7.5      | expected  | expected  |
| Ampere                     | 8.0, 8.6 | expected  | expected  |
| Ada Lovelace               | 8.9      | expected  | expected  |
| Hopper                     | 9.0      | expected  | expected  |
| Blackwell GB10 (DGX Spark) | 12.1     | pending   | pending   |
| Blackwell GB202 (RTX 5090) | 12.0     | pending   | pending   |

## Author

Joe McLaren (parallelArchitect) Human-directed GPU engineering with AI assistance.

Contact
gpu.validation@gmail.com
https://github.com/parallelArchitect

## License

MIT

