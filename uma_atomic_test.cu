/*
 * uma_atomic_test.cu
 * UMA Atomic Coherence Latency Probe v1.0 — Host Launcher
 *
 * Measures cycle-accurate latency of GPU-scope vs system-scope atomics
 * on unified managed memory. The delta between gpu-scope and sys-scope
 * is the NVLink-C2C coherence protocol cost on GB10.
 *
 * On discrete PCIe (Pascal to Ada): gpu-scope ~= sys-scope (~1.0x ratio)
 * On hardware-coherent UMA (GB10):  sys-scope >> gpu-scope (coherence cost)
 *
 * PTX kernel: uma_atomic_probe.ptx (must be in same directory)
 *
 * Build:
 *   x86_64:  nvcc -O2 -std=c++17 uma_atomic_test.cu -o uma_atomic -lcudart -lcuda
 *   aarch64: nvcc -O2 -std=c++17 uma_atomic_test.cu -o uma_atomic -lcudart -lcuda -lpthread
 *
 * Run:
 *   ./uma_atomic
 *   ./uma_atomic --json-only
 */

#include <cuda.h>
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

#define TOOL_VERSION    "1.0.0"
#define N_ELEMENTS      (1024 * 64)   /* 64K elements — fits in L2 on all targets */
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

#define CU_CHECK(call) do { \
    CUresult r = (call); \
    if (r != CUDA_SUCCESS) { \
        const char *s; cuGetErrorString(r, &s); \
        fprintf(stderr, "CU error %s:%d: %s\n", \
                __FILE__, __LINE__, s); \
        exit(1); \
    } \
} while(0)

/* ------------------------------------------------------------------ */
/* Platform detection — same logic as uma_probe and uma_bw             */
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

typedef struct {
    double p50_ns, p90_ns, p99_ns, min_ns, max_ns;
    double p50_cyc;
    size_t samples;
} PassResult;

/* ------------------------------------------------------------------ */
/* PTX loader                                                           */
/* ------------------------------------------------------------------ */

static CUfunction load_ptx_fn(const char *path, const char *fn) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f); rewind(f);
    char *src = (char*)malloc(sz + 1);
    size_t nr = fread(src, 1, sz, f);
    (void)nr;
    src[sz] = '\0';
    fclose(f);
    CUmodule mod;
    CU_CHECK(cuModuleLoadData(&mod, src));
    free(src);
    CUfunction func;
    CU_CHECK(cuModuleGetFunction(&func, mod, fn));
    return func;
}

/* ------------------------------------------------------------------ */
/* Run one pass                                                         */
/* ------------------------------------------------------------------ */

static PassResult run_pass(CUfunction kernel,
                           uint32_t *data, uint64_t *lat,
                           size_t n, int clock_mhz,
                           int device, int prefetch_to_gpu) {
    /* Reset data and latency arrays */
    CUDA_CHECK(cudaMemset(data, 0, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(lat,  0, n * sizeof(uint64_t)));

    if (prefetch_to_gpu) {
        CUDA_CHECK(cudaMemPrefetchAsync(data,
                   n * sizeof(uint32_t), device, 0));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    uint64_t ne = (uint64_t)n;
    void *args[] = { &data, &lat, &ne };
    int blocks = (int)((n + THREADS_PER_BLK - 1) / THREADS_PER_BLK);

    CU_CHECK(cuLaunchKernel(kernel, blocks, 1, 1,
                            THREADS_PER_BLK, 1, 1,
                            0, 0, args, NULL));
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
    double raw_p50 = percentile(valid, 50.0);
    double raw_p90 = percentile(valid, 90.0);
    double raw_p99 = percentile(valid, 99.0);
    r.p50_ns  = cycles_to_ns(raw_p50, clock_mhz);
    r.p90_ns  = cycles_to_ns(raw_p90, clock_mhz);
    r.p99_ns  = cycles_to_ns(raw_p99, clock_mhz);
    r.min_ns  = cycles_to_ns(valid.front(), clock_mhz);
    r.max_ns  = cycles_to_ns(valid.back(),  clock_mhz);
    r.p50_cyc = raw_p50;
    return r;
}

/* ------------------------------------------------------------------ */
/* CPU contention thread                                                */
/* ------------------------------------------------------------------ */

typedef struct { uint32_t *data; size_t n; volatile int stop; } CpuArg;

static void *cpu_contention_fn(void *arg) {
    CpuArg *a = (CpuArg *)arg;
    uint32_t acc = 0;
    while (!a->stop) {
        for (size_t i = 0; i < a->n && !a->stop; i++)
            acc += __atomic_fetch_add(&a->data[i], 1, __ATOMIC_SEQ_CST);
    }
    (void)acc;
    return NULL;
}

/* ------------------------------------------------------------------ */
/* JSON output                                                          */
/* ------------------------------------------------------------------ */

static void iso_ts(char *buf, size_t len) {
    time_t t = time(NULL);
    strftime(buf, len, "%Y-%m-%dT%H:%M:%SZ", gmtime(&t));
}

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
            "HW_COHERENT_UMA: sys/gpu ratio > 1.0x expected — NVLink-C2C coherence cost." :
            "DISCRETE_PCIE: sys/gpu ratio ~1.0x expected — no coherence protocol.");
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

    CU_CHECK(cuInit(0));
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
        printf("PTX gpu  : atom.global.gpu.add.u32\n");
        printf("PTX sys  : atom.global.sys.add.u32\n\n");
    }

    /* Allocate managed memory */
    uint32_t *data = nullptr;
    uint64_t *lat  = nullptr;
    CUDA_CHECK(cudaMallocManaged(&data, N_ELEMENTS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMallocManaged(&lat,  N_ELEMENTS * sizeof(uint64_t)));

    /* Load PTX kernels */
    CUfunction kernel_gpu = load_ptx_fn("uma_atomic_probe.ptx",
                                         "uma_atomic_gpu_kernel");
    CUfunction kernel_sys = load_ptx_fn("uma_atomic_probe.ptx",
                                         "uma_atomic_sys_kernel");

    PassResult gpu_scope = {}, sys_scope = {}, contention = {};

    /* --- GPU-scope pass --- */
    if (verbose) { printf("GPU-scope pass (atom.global.gpu):\n"); fflush(stdout); }
    for (int i = 0; i < WARMUP_RUNS; i++)
        run_pass(kernel_gpu, data, lat, N_ELEMENTS,
                 p.clock_mhz, device, 1);
    /* Average over measure runs */
    std::vector<double> p50s;
    for (int i = 0; i < MEASURE_RUNS; i++) {
        PassResult r = run_pass(kernel_gpu, data, lat, N_ELEMENTS,
                                p.clock_mhz, device, 1);
        p50s.push_back(r.p50_ns);
        gpu_scope = r;
    }
    if (verbose)
        printf("  p50: %8.1f ns  p90: %8.1f ns  p99: %8.1f ns\n\n",
               gpu_scope.p50_ns, gpu_scope.p90_ns, gpu_scope.p99_ns);

    /* --- SYS-scope pass --- */
    if (verbose) { printf("SYS-scope pass (atom.global.sys):\n"); fflush(stdout); }
    for (int i = 0; i < WARMUP_RUNS; i++)
        run_pass(kernel_sys, data, lat, N_ELEMENTS,
                 p.clock_mhz, device, 1);
    for (int i = 0; i < MEASURE_RUNS; i++) {
        PassResult r = run_pass(kernel_sys, data, lat, N_ELEMENTS,
                                p.clock_mhz, device, 1);
        sys_scope = r;
    }
    if (verbose)
        printf("  p50: %8.1f ns  p90: %8.1f ns  p99: %8.1f ns\n\n",
               sys_scope.p50_ns, sys_scope.p90_ns, sys_scope.p99_ns);

    /* --- Contention pass: sys-scope atomic + concurrent CPU --- */
    if (verbose) { printf("CONTENTION pass (sys-scope + CPU concurrent):\n"); fflush(stdout); }

    /* Prefetch half to GPU, leave half CPU-accessible */
    CUDA_CHECK(cudaMemPrefetchAsync(data,
               (N_ELEMENTS/2) * sizeof(uint32_t), device, 0));
    CUDA_CHECK(cudaMemPrefetchAsync(data + N_ELEMENTS/2,
               (N_ELEMENTS/2) * sizeof(uint32_t), cudaCpuDeviceId, 0));
    CUDA_CHECK(cudaDeviceSynchronize());

    CpuArg cpu_arg = { data, N_ELEMENTS, 0 };
    pthread_t tid;
    pthread_create(&tid, NULL, cpu_contention_fn, &cpu_arg);

    for (int i = 0; i < WARMUP_RUNS; i++)
        run_pass(kernel_sys, data, lat, N_ELEMENTS,
                 p.clock_mhz, device, 0);
    for (int i = 0; i < MEASURE_RUNS; i++) {
        PassResult r = run_pass(kernel_sys, data, lat, N_ELEMENTS,
                                p.clock_mhz, device, 0);
        contention = r;
    }

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
        if (p.type == PLAT_DISCRETE_PCIE)
            printf("Expected : ratio ~1.0x (no coherence protocol on discrete GPU)\n");
        else if (p.type == PLAT_HW_COHERENT_UMA)
            printf("Expected : ratio > 1.0x (NVLink-C2C coherence overhead on GB10)\n");
        printf("JSON     : %s\n", JSON_OUTPUT);
    }

    write_json(JSON_OUTPUT, &p, &gpu_scope, &sys_scope, &contention);
    if (verbose) printf("Done.\n");

    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(lat));
    return 0;
}
