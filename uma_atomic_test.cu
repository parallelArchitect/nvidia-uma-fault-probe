/*
 * uma_atomic_test.cu
 * UMA Atomic Coherence Latency Probe v1.1.0
 *
 * Measures cycle-accurate latency of GPU-scope vs system-scope atomics
 * on unified managed memory. The delta between gpu-scope and sys-scope
 * is the NVLink-C2C coherence protocol cost on GB10.
 *
 * On discrete PCIe: gpu-scope typically ~= sys-scope
 * On hardware-coherent UMA (GB10):  sys-scope may differ from gpu-scope
 *
 * Inline PTX kernels — no PTX files, no runtime JIT, works on all SM versions.
 * nvcc compiles inline PTX natively for the target GPU.
 *
 * Build:
 *   nvcc -O2 -std=c++17 -arch=sm_60 uma_atomic_test.cu -o uma_atomic -lcudart -lcuda -lpthread
 *   (SM 6.0+ required for scoped atomics)
 *
 * Run:
 *   ./uma_atomic
 *   ./uma_atomic --json-only
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <pthread.h>
#include <algorithm>
#include <vector>

#define TOOL_VERSION    "1.1.0"
#define N_ELEMENTS      (1024 * 64)   /* 64K elements */
#define THREADS_PER_BLK 256
#define WARMUP_RUNS     3
#define MEASURE_RUNS    5
#define JSON_OUTPUT     "uma_atomic_results.json"

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

/* ------------------------------------------------------------------ */
/* Platform detection                                                   */
/* ------------------------------------------------------------------ */

typedef enum {
    PLAT_HW_COHERENT_UMA,
    PLAT_DISCRETE_PCIE,
    PLAT_SOFTWARE_UMA,
    PLAT_UNKNOWN
} PlatformType;

typedef struct {
    char         name[256];
    int          sm_major, sm_minor;
    PlatformType type;
    int          hw_coherent;
    int          clock_mhz;
} Platform;

static Platform detect_platform(int device) {
    Platform p = {};
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    strncpy(p.name, prop.name, 255);
    p.sm_major = prop.major;
    p.sm_minor = prop.minor;

    int hpt = 0;
    cudaDeviceGetAttribute(&hpt,
        cudaDevAttrPageableMemoryAccessUsesHostPageTables, device);
    p.hw_coherent = hpt;

    int clk = 0;
    cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, device);
    p.clock_mhz = clk / 1000;

    if (hpt && prop.concurrentManagedAccess)
        p.type = PLAT_HW_COHERENT_UMA;
    else if (prop.concurrentManagedAccess)
        p.type = PLAT_DISCRETE_PCIE;
    else if (prop.managedMemory)
        p.type = PLAT_SOFTWARE_UMA;
    else
        p.type = PLAT_UNKNOWN;

    return p;
}

static const char *plat_name(PlatformType t) {
    switch(t) {
    case PLAT_HW_COHERENT_UMA: return "HARDWARE_COHERENT_UMA";
    case PLAT_DISCRETE_PCIE:   return "DISCRETE_PCIE";
    case PLAT_SOFTWARE_UMA:    return "SOFTWARE_UMA";
    default:                   return "UNKNOWN";
    }
}

/* ------------------------------------------------------------------ */
/* Kernels — inline PTX, nvcc compiles natively for target SM          */
/* ------------------------------------------------------------------ */

/*
 * GPU-scope atomic: atom.global.gpu.add.u32
 * Stays within GPU memory system — no coherence protocol.
 * Baseline atomic cost.
 */
__global__ void uma_atomic_gpu_kernel(
    uint32_t * __restrict__ data,
    uint64_t * __restrict__ latency,
    uint64_t n)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    uint32_t *ptr = data + tid;
    uint32_t result;

    uint64_t t0 = clock64();
    /* atom.global.gpu: GPU-scope atomic add, no coherence protocol */
    asm volatile("atom.global.gpu.add.u32 %0, [%1], 1;"
                 : "=r"(result) : "l"(ptr) : "memory");
    uint64_t t1 = clock64();

    (void)result;
    latency[tid] = t1 - t0;
}

/*
 * System-scope atomic: atom.global.sys.add.u32
 * Traverses coherence protocol — on GB10 this crosses NVLink-C2C.
 * The delta vs gpu-scope is the coherence overhead.
 */
__global__ void uma_atomic_sys_kernel(
    uint32_t * __restrict__ data,
    uint64_t * __restrict__ latency,
    uint64_t n)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    uint32_t *ptr = data + tid;
    uint32_t result;

    uint64_t t0 = clock64();
    /* atom.global.sys: system-scope atomic add, traverses coherence protocol */
    asm volatile("atom.global.sys.add.u32 %0, [%1], 1;"
                 : "=r"(result) : "l"(ptr) : "memory");
    uint64_t t1 = clock64();

    (void)result;
    latency[tid] = t1 - t0;
}

/* ------------------------------------------------------------------ */
/* Stats                                                                */
/* ------------------------------------------------------------------ */

static double percentile(std::vector<uint64_t> &v, double pct) {
    if (v.empty()) return 0;
    size_t idx = (size_t)(pct / 100.0 * (v.size() - 1));
    return (double)v[std::min(idx, v.size()-1)];
}

static double cycles_to_ns(double cycles, int mhz) {
    return cycles / (double)mhz * 1000.0;
}

static void iso_ts(char *buf, size_t len) {
    time_t t = time(NULL);
    strftime(buf, len, "%Y-%m-%dT%H:%M:%SZ", gmtime(&t));
}

typedef struct {
    double p50_ns, p90_ns, p99_ns, min_ns, max_ns;
    double p50_cyc;
    size_t samples;
} PassResult;

/* ------------------------------------------------------------------ */
/* CPU contention thread                                                */
/* ------------------------------------------------------------------ */

typedef struct {
    uint32_t *data;
    size_t    n;
    volatile int stop;
} CpuArg;

static void *cpu_contention_fn(void *arg) {
    CpuArg *a = (CpuArg *)arg;
    uint32_t *data = a->data;
    size_t n = a->n;
    while (!a->stop) {
        for (size_t i = 0; i < n && !a->stop; i++)
            __atomic_fetch_add(&data[i], 1, __ATOMIC_SEQ_CST);
    }
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Run one pass                                                         */
/* ------------------------------------------------------------------ */

static PassResult run_pass_gpu(uint32_t *data, uint64_t *lat,
                                size_t n, int clock_mhz,
                                int device, int prefetch_to_gpu,
                                bool use_sys_scope) {
    CUDA_CHECK(cudaMemset(data, 0, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(lat,  0, n * sizeof(uint64_t)));

    if (prefetch_to_gpu) {
#if CUDART_VERSION >= 12020
        cudaMemLocation loc = {cudaMemLocationTypeDevice, device};
        CUDA_CHECK(cudaMemPrefetchAsync(data, n * sizeof(uint32_t), loc, 0));
#else
        CUDA_CHECK(cudaMemPrefetchAsync(data, n * sizeof(uint32_t), device, 0));
#endif
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    int blocks = (int)((n + THREADS_PER_BLK - 1) / THREADS_PER_BLK);
    if (use_sys_scope)
        uma_atomic_sys_kernel<<<blocks, THREADS_PER_BLK>>>(data, lat, (uint64_t)n);
    else
        uma_atomic_gpu_kernel<<<blocks, THREADS_PER_BLK>>>(data, lat, (uint64_t)n);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint64_t> host(n);
    CUDA_CHECK(cudaMemcpy(host.data(), lat,
                          n * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    std::vector<uint64_t> valid;
    valid.reserve(n);
    for (auto v : host) if (v > 0) valid.push_back(v);
    std::sort(valid.begin(), valid.end());

    PassResult r = {};
    if (valid.empty()) return r;
    r.samples = valid.size();
    r.p50_cyc = percentile(valid, 50.0);
    r.p50_ns  = cycles_to_ns(percentile(valid, 50.0), clock_mhz);
    r.p90_ns  = cycles_to_ns(percentile(valid, 90.0), clock_mhz);
    r.p99_ns  = cycles_to_ns(percentile(valid, 99.0), clock_mhz);
    r.min_ns  = cycles_to_ns(valid.front(), clock_mhz);
    r.max_ns  = cycles_to_ns(valid.back(),  clock_mhz);
    return r;
}

/* ------------------------------------------------------------------ */
/* JSON writer                                                          */
/* ------------------------------------------------------------------ */

static void write_json(const char *path,
                       const Platform *p,
                       PassResult *gpu_scope,
                       PassResult *sys_scope,
                       PassResult *contention) {
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path); return; }

    char ts[64]; iso_ts(ts, sizeof(ts));
    double ratio = (gpu_scope->p50_ns > 0) ?
                   sys_scope->p50_ns / gpu_scope->p50_ns : 0.0;

    fprintf(f, "{\n");
    fprintf(f, "  \"tool\": \"uma-atomic-coherence-probe\",\n");
    fprintf(f, "  \"version\": \"%s\",\n", TOOL_VERSION);
    fprintf(f, "  \"timestamp\": \"%s\",\n", ts);
    fprintf(f, "  \"platform\": {\n");
    fprintf(f, "    \"gpu_name\": \"%s\",\n", p->name);
    fprintf(f, "    \"sm_major\": %d,\n", p->sm_major);
    fprintf(f, "    \"sm_minor\": %d,\n", p->sm_minor);
    fprintf(f, "    \"uma_type\": \"%s\",\n", plat_name(p->type));
    fprintf(f, "    \"hw_coherent\": %s,\n",
            p->hw_coherent ? "true" : "false");
    fprintf(f, "    \"clock_mhz\": %d,\n", p->clock_mhz);
    fprintf(f, "    \"n_elements\": %d\n", N_ELEMENTS);
    fprintf(f, "  },\n");

    fprintf(f, "  \"results\": {\n");
    fprintf(f, "    \"gpu_scope\": {\n");
    fprintf(f, "      \"ptx_op\": \"atom.global.gpu.add.u32\",\n");
    fprintf(f, "      \"p50_ns\": %.1f, \"p90_ns\": %.1f, \"p99_ns\": %.1f,\n",
            gpu_scope->p50_ns, gpu_scope->p90_ns, gpu_scope->p99_ns);
    fprintf(f, "      \"min_ns\": %.1f, \"max_ns\": %.1f,\n",
            gpu_scope->min_ns, gpu_scope->max_ns);
    fprintf(f, "      \"p50_cycles\": %.0f, \"samples\": %zu\n",
            gpu_scope->p50_cyc, gpu_scope->samples);
    fprintf(f, "    },\n");

    fprintf(f, "    \"sys_scope\": {\n");
    fprintf(f, "      \"ptx_op\": \"atom.global.sys.add.u32\",\n");
    fprintf(f, "      \"p50_ns\": %.1f, \"p90_ns\": %.1f, \"p99_ns\": %.1f,\n",
            sys_scope->p50_ns, sys_scope->p90_ns, sys_scope->p99_ns);
    fprintf(f, "      \"min_ns\": %.1f, \"max_ns\": %.1f,\n",
            sys_scope->min_ns, sys_scope->max_ns);
    fprintf(f, "      \"p50_cycles\": %.0f, \"samples\": %zu\n",
            sys_scope->p50_cyc, sys_scope->samples);
    fprintf(f, "    },\n");

    fprintf(f, "    \"contention\": {\n");
    fprintf(f, "      \"ptx_op\": \"atom.global.sys.add.u32 + concurrent CPU __atomic_fetch_add\",\n");
    fprintf(f, "      \"p50_ns\": %.1f, \"p90_ns\": %.1f, \"p99_ns\": %.1f,\n",
            contention->p50_ns, contention->p90_ns, contention->p99_ns);
    fprintf(f, "      \"min_ns\": %.1f, \"max_ns\": %.1f,\n",
            contention->min_ns, contention->max_ns);
    fprintf(f, "      \"p50_cycles\": %.0f, \"samples\": %zu\n",
            contention->p50_cyc, contention->samples);
    fprintf(f, "    }\n");
    fprintf(f, "  },\n");

    fprintf(f, "  \"interpretation\": {\n");
    fprintf(f, "    \"sys_gpu_ratio\": %.2f,\n", ratio);
    fprintf(f, "    \"platform_note\": \"%s\",\n",
            p->type == PLAT_HW_COHERENT_UMA ?
            "HW_COHERENT_UMA: sys/gpu ratio measures NVLink-C2C coherence cost." :
            "DISCRETE_PCIE: no coherence protocol.");
    fprintf(f, "    \"coherence_overhead_ns\": %.1f\n",
            sys_scope->p50_ns - gpu_scope->p50_ns);
    fprintf(f, "  }\n");
    fprintf(f, "}\n");
    fclose(f);
}

/* ------------------------------------------------------------------ */
/* Main                                                                 */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv) {
    int json_only = 0;
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], "--json-only") == 0)
            json_only = 1;

    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));

    Platform p = detect_platform(device);
    int verbose = !json_only;

    if (verbose) {
        printf("=== UMA Atomic Coherence Probe v%s ===\n", TOOL_VERSION);
        printf("GPU      : %s (SM %d.%d)\n",
               p.name, p.sm_major, p.sm_minor);
        printf("Platform : %s\n", plat_name(p.type));
        printf("Coherent : %s\n",
               p.hw_coherent ? "yes (hardware)" : "no");
        printf("Clock    : %d MHz\n", p.clock_mhz);
        printf("Elements : %d\n", N_ELEMENTS);
        printf("Warmup   : %d runs  Measure: %d runs\n",
               WARMUP_RUNS, MEASURE_RUNS);
        printf("Kernel   : inline PTX atomics, nvcc native\n");
        printf("PTX gpu  : atom.global.gpu.add.u32\n");
        printf("PTX sys  : atom.global.sys.add.u32\n\n");
    }

    uint32_t *data = nullptr;
    uint64_t *lat  = nullptr;
    CUDA_CHECK(cudaMallocManaged(&data, N_ELEMENTS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMallocManaged(&lat,  N_ELEMENTS * sizeof(uint64_t)));

    PassResult gpu_scope = {}, sys_scope = {}, contention = {};

    /* --- GPU-scope pass --- */
    if (verbose) { printf("GPU-scope pass (atom.global.gpu):\n"); fflush(stdout); }
    for (int i = 0; i < WARMUP_RUNS; i++)
        run_pass_gpu(data, lat, N_ELEMENTS, p.clock_mhz, device, 1, false);
    for (int i = 0; i < MEASURE_RUNS; i++)
        gpu_scope = run_pass_gpu(data, lat, N_ELEMENTS, p.clock_mhz, device, 1, false);
    if (verbose)
        printf("  p50: %8.1f ns  p90: %8.1f ns  p99: %8.1f ns\n\n",
               gpu_scope.p50_ns, gpu_scope.p90_ns, gpu_scope.p99_ns);

    /* --- SYS-scope pass --- */
    if (verbose) { printf("SYS-scope pass (atom.global.sys):\n"); fflush(stdout); }
    for (int i = 0; i < WARMUP_RUNS; i++)
        run_pass_gpu(data, lat, N_ELEMENTS, p.clock_mhz, device, 1, true);
    for (int i = 0; i < MEASURE_RUNS; i++)
        sys_scope = run_pass_gpu(data, lat, N_ELEMENTS, p.clock_mhz, device, 1, true);
    if (verbose)
        printf("  p50: %8.1f ns  p90: %8.1f ns  p99: %8.1f ns\n\n",
               sys_scope.p50_ns, sys_scope.p90_ns, sys_scope.p99_ns);

    /* --- Contention pass: sys-scope atomic + concurrent CPU --- */
    if (verbose) { printf("CONTENTION pass (sys-scope + CPU concurrent):\n"); fflush(stdout); }

#if CUDART_VERSION >= 12020
    cudaMemLocation loc2 = {cudaMemLocationTypeDevice, device};
    CUDA_CHECK(cudaMemPrefetchAsync(data,
               (N_ELEMENTS/2) * sizeof(uint32_t), loc2, 0));
#else
    CUDA_CHECK(cudaMemPrefetchAsync(data,
               (N_ELEMENTS/2) * sizeof(uint32_t), device, 0));
#endif
    CUDA_CHECK(cudaMemPrefetchAsync(data + N_ELEMENTS/2,
               (N_ELEMENTS/2) * sizeof(uint32_t), cudaCpuDeviceId, 0));
    CUDA_CHECK(cudaDeviceSynchronize());

    CpuArg cpu_arg = { data, N_ELEMENTS, 0 };
    pthread_t tid;
    pthread_create(&tid, NULL, cpu_contention_fn, &cpu_arg);

    for (int i = 0; i < WARMUP_RUNS; i++)
        run_pass_gpu(data, lat, N_ELEMENTS, p.clock_mhz, device, 0, true);
    for (int i = 0; i < MEASURE_RUNS; i++)
        contention = run_pass_gpu(data, lat, N_ELEMENTS, p.clock_mhz, device, 0, true);

    cpu_arg.stop = 1;
    pthread_join(tid, NULL);

    if (verbose)
        printf("  p50: %8.1f ns  p90: %8.1f ns  p99: %8.1f ns\n\n",
               contention.p50_ns, contention.p90_ns, contention.p99_ns);

    /* --- Summary --- */
    if (verbose) {
        double ratio = (gpu_scope.p50_ns > 0) ?
                       sys_scope.p50_ns / gpu_scope.p50_ns : 0.0;
        double overhead = sys_scope.p50_ns - gpu_scope.p50_ns;
        printf("=== Summary ===\n");
        printf("GPU-scope p50 : %8.1f ns  (%6.0f cycles) [atom.global.gpu]\n",
               gpu_scope.p50_ns, gpu_scope.p50_cyc);
        printf("SYS-scope p50 : %8.1f ns  (%6.0f cycles) [atom.global.sys]\n",
               sys_scope.p50_ns, sys_scope.p50_cyc);
        printf("CONTENTION p50: %8.1f ns  (%6.0f cycles) [sys + CPU concurrent]\n",
               contention.p50_ns, contention.p50_cyc);
        printf("SYS/GPU ratio : %.2fx\n", ratio);
        printf("Coherence cost: %.1f ns overhead\n", overhead);
        printf("\nPlatform : %s\n", plat_name(p.type));
        printf("JSON     : %s\n", JSON_OUTPUT);
    }

    write_json(JSON_OUTPUT, &p, &gpu_scope, &sys_scope, &contention);
    if (verbose) printf("Done.\n");

    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(lat));
    return 0;
}
