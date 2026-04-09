/*
 * probe_launcher.cu
 * UMA Fault Latency Probe v1.1 — Host Launcher
 *
 * Platform-aware. Outputs human summary + JSON log.
 * PTX kernel: uma_fault_probe.ptx (must be in same directory)
 *
 * Build:
 *   nvcc -O2 -std=c++17 probe_launcher.cu -o uma_probe -lcudart -lcuda
 *
 * Run:
 *   ./uma_probe
 *   ./uma_probe --json-only
 *
 * Share the JSON output for remote analysis:
 *   uma_probe_results.json
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <algorithm>
#include <vector>

#define TOOL_VERSION    "1.1.0"
#define BUFFER_MB       64
#define N_ELEMENTS      ((BUFFER_MB * 1024 * 1024) / sizeof(float))
#define THREADS_PER_BLK 256
#define JSON_OUTPUT     "uma_probe_results.json"

// --- Error checking ---
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

// --- Platform ---
typedef enum {
    PLAT_HW_COHERENT_UMA,   // GB10, GH200
    PLAT_DISCRETE_PCIE,     // Pascal, Turing, Ampere, Ada
    PLAT_SOFTWARE_UMA,      // Tegra, older SoCs
    PLAT_UNKNOWN
} PlatformType;

typedef struct {
    char         name[256];
    int          sm_major, sm_minor;
    PlatformType type;
    int          hw_coherent;
    int          managed;
    int          concurrent;
    int          host_page_tables;
    int          clock_mhz;
} Platform;

static Platform detect_platform(int device) {
    Platform p = {};
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    strncpy(p.name, prop.name, 255);
    p.sm_major   = prop.major;
    p.sm_minor   = prop.minor;
    p.managed    = prop.managedMemory;
    p.concurrent = prop.concurrentManagedAccess;

    int hpt = 0;
    cudaDeviceGetAttribute(&hpt,
        cudaDevAttrPageableMemoryAccessUsesHostPageTables, device);
    p.host_page_tables = hpt;
    p.hw_coherent      = hpt;

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

static const char *plat_note(PlatformType t) {
    switch(t) {
    case PLAT_HW_COHERENT_UMA:
        return "HW_COHERENT_UMA: One physical pool. HW fault active. COLD/WARM ratio 20-100x expected.";
    case PLAT_DISCRETE_PCIE:
        return "DISCRETE_PCIE: DRAM latency only. COLD=WARM ratio ~1.0x. No HW fault visible.";
    case PLAT_SOFTWARE_UMA:
        return "Software-managed UMA. "
               "Page migration overhead may be partially visible.";
    default:
        return "Platform not recognized.";
    }
}

// --- Stats ---
static double percentile(std::vector<uint64_t> &v, double pct) {
    if (v.empty()) return 0;
    size_t idx = (size_t)(pct / 100.0 * (v.size() - 1));
    return (double)v[std::min(idx, v.size()-1)];
}

static double cycles_to_ns(double cycles, int mhz) {
    return cycles / (double)mhz * 1000.0;
}

// --- Timestamp ---
static void iso_ts(char *buf, size_t len) {
    time_t t = time(NULL);
    strftime(buf, len, "%Y-%m-%dT%H:%M:%SZ", gmtime(&t));
}

// --- Load PTX ---
static CUfunction load_ptx(const char *path, const char *fn) {
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

// --- Pass types ---
typedef enum { PASS_COLD, PASS_WARM, PASS_PRESSURE } PassType;

typedef struct {
    double p50_ns, p90_ns, p99_ns, min_ns, max_ns;
    double p50_cyc, p99_cyc;
    double cold_warm_ratio;
    size_t samples;
    uint64_t raw_p50, raw_p90, raw_p99;
} PassResult;

// --- Run one pass ---
static PassResult run_pass(CUfunction kernel,
                           float *data, uint64_t *lat,
                           size_t n, PassType pass,
                           int clock_mhz, int device,
                           int verbose) {
    if (pass == PASS_COLD) {
        if (verbose) { printf("  touching pages from CPU... "); fflush(stdout); }
        for (size_t i = 0; i < n; i++) data[i] = (float)i;
        if (verbose) printf("done\n");
    } else if (pass == PASS_WARM) {
        if (verbose) { printf("  prefetching to GPU... "); fflush(stdout); }
        CUDA_CHECK(cudaMemPrefetchAsync(data, n*sizeof(float), device, 0));
        CUDA_CHECK(cudaDeviceSynchronize());
        if (verbose) printf("done\n");
    } else {
        if (verbose) { printf("  mixed CPU/GPU residency... "); fflush(stdout); }
        for (size_t i = 0; i < n/2; i++) data[i] = (float)i;
        CUDA_CHECK(cudaMemPrefetchAsync(data + n/2,
                   (n/2)*sizeof(float), device, 0));
        CUDA_CHECK(cudaDeviceSynchronize());
        if (verbose) printf("done\n");
    }

    CUDA_CHECK(cudaMemset(lat, 0, n * sizeof(uint64_t)));

    uint64_t ne = (uint64_t)n;
    void *args[] = { &data, &lat, &ne };
    int blocks = (int)((n + THREADS_PER_BLK - 1) / THREADS_PER_BLK);

    if (verbose) { printf("  running kernel...      "); fflush(stdout); }
    CU_CHECK(cuLaunchKernel(kernel, blocks, 1, 1,
                            THREADS_PER_BLK, 1, 1,
                            0, 0, args, NULL));
    CUDA_CHECK(cudaDeviceSynchronize());
    if (verbose) printf("done\n");

    // Copy results
    std::vector<uint64_t> host(n);
    CUDA_CHECK(cudaMemcpy(host.data(), lat,
                          n*sizeof(uint64_t), cudaMemcpyDeviceToHost));

    // Filter and sort
    std::vector<uint64_t> valid;
    valid.reserve(n);
    for (auto v : host) if (v > 0) valid.push_back(v);
    std::sort(valid.begin(), valid.end());

    PassResult r = {};
    if (valid.empty()) return r;

    r.samples  = valid.size();
    r.raw_p50  = (uint64_t)percentile(valid, 50.0);
    r.raw_p90  = (uint64_t)percentile(valid, 90.0);
    r.raw_p99  = (uint64_t)percentile(valid, 99.0);
    r.p50_ns   = cycles_to_ns(r.raw_p50, clock_mhz);
    r.p90_ns   = cycles_to_ns(r.raw_p90, clock_mhz);
    r.p99_ns   = cycles_to_ns(r.raw_p99, clock_mhz);
    r.min_ns   = cycles_to_ns(valid.front(), clock_mhz);
    r.max_ns   = cycles_to_ns(valid.back(),  clock_mhz);
    r.p50_cyc  = (double)r.raw_p50;
    r.p99_cyc  = (double)r.raw_p99;
    return r;
}

// --- JSON writer ---
static void write_json(const char *path,
                       const Platform *p,
                       PassResult *cold,
                       PassResult *warm,
                       PassResult *pressure) {
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path); return; }

    char ts[64]; iso_ts(ts, sizeof(ts));
    double ratio = (warm->p50_ns > 0) ?
                   cold->p50_ns / warm->p50_ns : 0.0;

    fprintf(f, "{\n");
    fprintf(f, "  \"tool\": \"uma-fault-latency-probe\",\n");
    fprintf(f, "  \"version\": \"%s\",\n", TOOL_VERSION);
    fprintf(f, "  \"timestamp\": \"%s\",\n", ts);
    fprintf(f, "  \"platform\": {\n");
    fprintf(f, "    \"gpu_name\": \"%s\",\n", p->name);
    fprintf(f, "    \"sm_major\": %d,\n", p->sm_major);
    fprintf(f, "    \"sm_minor\": %d,\n", p->sm_minor);
    fprintf(f, "    \"uma_type\": \"%s\",\n", plat_name(p->type));
    fprintf(f, "    \"hw_coherent\": %s,\n",
            p->hw_coherent ? "true" : "false");
    fprintf(f, "    \"host_page_tables\": %s,\n",
            p->host_page_tables ? "true" : "false");
    fprintf(f, "    \"clock_mhz\": %d,\n", p->clock_mhz);
    fprintf(f, "    \"buffer_mb\": %d\n", BUFFER_MB);
    fprintf(f, "  },\n");

    fprintf(f, "  \"results\": {\n");
    fprintf(f, "    \"cold\": {\n");
    fprintf(f, "      \"p50_ns\": %.1f, \"p90_ns\": %.1f, "
               "\"p99_ns\": %.1f,\n",
            cold->p50_ns, cold->p90_ns, cold->p99_ns);
    fprintf(f, "      \"min_ns\": %.1f, \"max_ns\": %.1f,\n",
            cold->min_ns, cold->max_ns);
    fprintf(f, "      \"p50_cycles\": %.0f, \"samples\": %zu\n",
            cold->p50_cyc, cold->samples);
    fprintf(f, "    },\n");

    fprintf(f, "    \"warm\": {\n");
    fprintf(f, "      \"p50_ns\": %.1f, \"p90_ns\": %.1f, "
               "\"p99_ns\": %.1f,\n",
            warm->p50_ns, warm->p90_ns, warm->p99_ns);
    fprintf(f, "      \"min_ns\": %.1f, \"max_ns\": %.1f,\n",
            warm->min_ns, warm->max_ns);
    fprintf(f, "      \"p50_cycles\": %.0f, \"samples\": %zu\n",
            warm->p50_cyc, warm->samples);
    fprintf(f, "    },\n");

    fprintf(f, "    \"pressure\": {\n");
    fprintf(f, "      \"p50_ns\": %.1f, \"p90_ns\": %.1f, "
               "\"p99_ns\": %.1f,\n",
            pressure->p50_ns, pressure->p90_ns, pressure->p99_ns);
    fprintf(f, "      \"min_ns\": %.1f, \"max_ns\": %.1f,\n",
            pressure->min_ns, pressure->max_ns);
    fprintf(f, "      \"p50_cycles\": %.0f, \"samples\": %zu\n",
            pressure->p50_cyc, pressure->samples);
    fprintf(f, "    }\n");
    fprintf(f, "  },\n");

    fprintf(f, "  \"interpretation\": {\n");
    fprintf(f, "    \"cold_warm_ratio\": %.2f,\n", ratio);
    fprintf(f, "    \"platform_note\": \"%s\",\n", plat_note(p->type));
    fprintf(f, "    \"ptx_load_op\": "
               "\"ld.global.cv (cache volatile, bypass all caches)\",\n");
    fprintf(f, "    \"measurement\": \"%%clock64 before/after ld.global\"\n");
    fprintf(f, "  }\n");
    fprintf(f, "}\n");
    fclose(f);
}

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
        printf("=== UMA Fault Latency Probe v%s ===\n", TOOL_VERSION);
        printf("GPU      : %s (SM %d.%d)\n",
               p.name, p.sm_major, p.sm_minor);
        printf("Platform : %s\n", plat_name(p.type));
        printf("Coherent : %s\n",
               p.hw_coherent ? "yes (hardware)" : "no");
        printf("Clock    : %d MHz\n", p.clock_mhz);
        printf("Buffer   : %d MB (%zu elements)\n",
               BUFFER_MB, (size_t)N_ELEMENTS);
        printf("PTX op   : ld.global.cv (bypass all caches)\n");
        printf("Measure  : %%clock64 before/after ld.global\n");
        printf("Note     : %s\n\n", plat_note(p.type));
    }

    // Allocate
    float    *data = nullptr;
    uint64_t *lat  = nullptr;
    CUDA_CHECK(cudaMallocManaged(&data,
               N_ELEMENTS * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&lat,
               N_ELEMENTS * sizeof(uint64_t)));

    // Load PTX
    CUfunction kernel = load_ptx("uma_fault_probe.ptx",
                                  "uma_fault_probe_kernel");

    // Run passes
    PassResult cold, warm, pressure;

    if (verbose) printf("COLD pass (CPU->GPU fault):\n");
    cold = run_pass(kernel, data, lat, N_ELEMENTS,
                    PASS_COLD, p.clock_mhz, device, verbose);
    if (verbose) {
        printf("  p50: %8.1f ns  p90: %8.1f ns  "
               "p99: %8.1f ns\n\n",
               cold.p50_ns, cold.p90_ns, cold.p99_ns);
    }

    if (verbose) printf("WARM pass (GPU resident):\n");
    warm = run_pass(kernel, data, lat, N_ELEMENTS,
                    PASS_WARM, p.clock_mhz, device, verbose);
    if (verbose) {
        printf("  p50: %8.1f ns  p90: %8.1f ns  "
               "p99: %8.1f ns\n\n",
               warm.p50_ns, warm.p90_ns, warm.p99_ns);
    }

    if (verbose) printf("PRESSURE pass (thrash):\n");
    pressure = run_pass(kernel, data, lat, N_ELEMENTS,
                        PASS_PRESSURE, p.clock_mhz, device, verbose);
    if (verbose) {
        printf("  p50: %8.1f ns  p90: %8.1f ns  "
               "p99: %8.1f ns\n\n",
               pressure.p50_ns, pressure.p90_ns, pressure.p99_ns);
    }

    // Summary
    if (verbose) {
        double ratio = (warm.p50_ns > 0) ?
                       cold.p50_ns / warm.p50_ns : 0.0;
        printf("=== Summary ===\n");
        printf("COLD  p50: %8.1f ns  (%6.0f cycles)\n",
               cold.p50_ns, cold.p50_cyc);
        printf("WARM  p50: %8.1f ns  (%6.0f cycles)\n",
               warm.p50_ns, warm.p50_cyc);
        printf("PRESS p50: %8.1f ns  (%6.0f cycles)\n",
               pressure.p50_ns, pressure.p50_cyc);
        printf("COLD/WARM ratio: %.2fx\n", ratio);
        printf("\nPlatform : %s\n", plat_name(p.type));
        if (p.type == PLAT_DISCRETE_PCIE)
            printf("Expected : ratio ~1.0x (no HW fault visible)\n");
        else if (p.type == PLAT_HW_COHERENT_UMA)
            printf("Expected : ratio 20-100x (HW UMA fault cost)\n");
        printf("JSON     : %s\n", JSON_OUTPUT);
    }

    write_json(JSON_OUTPUT, &p, &cold, &warm, &pressure);
    if (verbose) printf("Done.\n");

    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(lat));
    return 0;
}
