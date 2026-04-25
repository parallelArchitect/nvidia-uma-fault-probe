# Changelog

All notable changes to nvidia-uma-fault-probe are documented here.

---

## [1.1.0] — 2026-04-12

### Added
- `uma_atomic_test.cu` — UMA Atomic Coherence Latency Probe
- `uma_atomic_probe.ptx` — PTX kernels for gpu-scope and sys-scope atomics
- Three measurement passes: gpu-scope (`atom.global.gpu`), sys-scope
  (`atom.global.sys`), and sys-scope with concurrent CPU thread
- SYS/GPU ratio metric — coherence protocol cost on hardware-coherent UMA
- Validated on Pascal GTX 1080 (SM 6.1): SYS/GPU ratio 1.01x (expected ~1.0x)
- GB10 pending: SYS/GPU ratio expected > 1.0x (NVLink-C2C coherence cost)

### Fixed
- `uma_bandwidth_test.cu`: removed hardcoded peak BW estimates by SM generation
  (484/900/936/3350 GB/s). Peak now derived from hardware attributes at runtime
  via `cudaDevAttrGlobalMemoryBusWidth` and `cudaDevAttrMemoryClockRate`.
  Validated GTX 1080: 320.32 GB/s (matches GDDR5X spec)
- `uma_bandwidth_test.cu`: GB10 `peak_bw_gbs` set to 0.0 — memory clock
  returns N/A on driver 580.142 (confirmed from nvidia-smi -q sosreport).
  Peak percentage omitted rather than fabricated
- `uma_fault_probe.ptx`: corrected stale header comment `compute_80+` to
  `compute_60`. The `.target` directive was already correct; only the comment
  was wrong

### Changed
- README updated to document all three tools
- Example output updated to current validated run (GTX 1080)
- GB10 expected bandwidth section updated: 273 GB/s spec estimate removed,
  replaced with community data request

---

## [1.0.0] — 2026-04-10

### Added
- `probe_launcher.cu` — UMA Fault Latency Probe host launcher
- `uma_fault_probe.ptx` — PTX kernel using `%clock64` + `ld.global.cv`
- Three passes: COLD, WARM, PRESSURE
- COLD/WARM ratio signal for discrete vs coherent UMA platform discrimination
- `uma_bandwidth_test.cu` — UMA Bandwidth Test
- PTX cache operators: `ld.global.cg` (L1 bypass read), `st.global.cs`
  (L2 bypass write, true DRAM bandwidth)
- Concurrent CPU+GPU bandwidth measurement
- Platform detection: HARDWARE_COHERENT_UMA, DISCRETE_PCIE, SOFTWARE_UMA
- JSON output for all tools — shareable for remote analysis
- `--json-only` flag on all tools
- Validated on Pascal GTX 1080 (SM 6.1):
  - COLD/WARM ratio: 1.00x
  - GPU read: 248.91 GB/s
  - GPU write: 254.16 GB/s
- GB10 / DGX Spark community data collection via Issues

## v1.2.0 — 2026-04-24

### Changed
- uma_probe: CUDA C kernel with inline PTX ld.global.cv, no PTX files
- uma_atomic: inline PTX scoped atomics, no PTX files (v1.1.0)
- uma_bw: CUDA 13 compat verified, cudaCpuDeviceId guarded
- All PTX files removed — nvcc compiles natively for target SM
- CUDA 13 cudaMemPrefetchAsync API compat across all three tools

### Validated
- Pascal SM 6.1 (CUDA 12.0) — confirmed
- Blackwell GB10 SM 12.1 (CUDA 13.0, driver 580.142) — confirmed

## v1.2.0 — 2026-04-25

### Added
- run_all.sh — runs all three probes with thermal cooldown, sparkview detection
- collect_results.sh — packages JSON results + sparkview logs into timestamped zip

### Changed
- uma_probe: CUDA C kernel with inline PTX ld.global.cv, no PTX files
- uma_atomic: inline PTX scoped atomics, no PTX files (v1.1.0)
- uma_bw: CUDA 13 compat verified, cudaCpuDeviceId guarded
- All PTX files removed — nvcc compiles natively for target SM
- CUDA 13 cudaMemPrefetchAsync API compat across all three tools

### Validated
- Pascal SM 6.1 (CUDA 12.0) — confirmed
- Blackwell GB10 SM 12.1 (CUDA 13.0, driver 580.142) — first community measurement
