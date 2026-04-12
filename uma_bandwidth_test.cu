/*
 * uma_bandwidth_test.cu
 * UMA Bandwidth Test v2.0
 *
 * Platform-aware memory bandwidth measurement.
 * PTX cache operators for true DRAM measurement:
 *   ld.global.cg  — cache at L2, bypass L1 (read)
 *   st.global.cs  — bypass L2, true DRAM write
 *
 * Detects platform at runtime:
 *   HARDWARE_COHERENT_UMA — GB10, GH200
 *   DISCRETE_PCIE         — Pascal through Ada
 *   SOFTWARE_UMA          — Tegra, older SoCs
 *
 * Build:
 *   x86_64: nvcc -O2 -std=c++17 uma_bandwidth_test.cu -o uma_bw -lcudart
 *   aarch64: nvcc -O2 -std=c++17 uma_bandwidth_test.cu -o uma_bw -lcudart -lpthread
 *
 * Run:
 *   ./uma_bw
 *   ./uma_bw --json-only
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <math.h>

#define TOOL_VERSION      "2.0.0"
#define BUFFER_GB         4
#define BUFFER_BYTES      ((size_t)BUFFER_GB * 1024ULL * 1024ULL * 1024ULL)
#define THREADS_PER_BLOCK 256
#define WARMUP_RUNS       2
#define MEASURE_RUNS      5
#define MAX_RUNS          16
#define JSON_OUTPUT       "uma_bw_results.json"

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
    PLAT_HW_COHERENT_UMA,   /* GB10, GH200 — one physical pool        */
    PLAT_DISCRETE_PCIE,     /* Pascal→Ada  — separate VRAM + system RAM*/
    PLAT_SOFTWARE_UMA,      /* Tegra, older SoCs                       */
    PLAT_UNKNOWN
} PlatformType;

typedef struct {
    char         name[256];
    int          sm_major;
    int          sm_minor;
    PlatformType type;
    int          hw_coherent;
    int          managed;
    int          concurrent;
    int          host_page_tables;
    double       peak_bw_gbs;
    int          l2_kb;
    int          clock_mhz;
} Platform;

typedef struct {
    double runs[MAX_RUNS];
    double mean;
    double stddev;
} Stat;

typedef struct {
    Stat   gpu_read;
    Stat   gpu_write;
    Stat   gpu_copy;
    Stat   cpu_read;
    Stat   cpu_write;
    double conc_gpu;
    double conc_cpu;
    double conc_total;
} Results;

static double stat_mean(double *a, int n) {
    double s = 0;
    for (int i = 0; i < n; i++) s += a[i];
    return s / n;
}

static double stat_stddev(double *a, int n) {
    double m = stat_mean(a, n), s = 0;
    for (int i = 0; i < n; i++) s += (a[i]-m)*(a[i]-m);
    return sqrt(s / n);
}

static void stat_compute(Stat *s) {
    s->mean   = stat_mean(s->runs, MEASURE_RUNS);
    s->stddev = stat_stddev(s->runs, MEASURE_RUNS);
}

/* ------------------------------------------------------------------ */
/* Platform detection function                                          */
/* ------------------------------------------------------------------ */

static Platform detect_platform(int device) {
    Platform p = {};
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    strncpy(p.name, prop.name, 255);
    p.sm_major   = prop.major;
    p.sm_minor   = prop.minor;
    p.managed    = prop.managedMemory;
    p.concurrent = prop.concurrentManagedAccess;
    p.l2_kb      = prop.l2CacheSize / 1024;

    int clk = 0;
    cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, device);
    p.clock_mhz = clk / 1000;

    int hpt = 0;
    cudaDeviceGetAttribute(&hpt,
        cudaDevAttrPageableMemoryAccessUsesHostPageTables,
        device);
    p.host_page_tables = hpt;
    p.hw_coherent      = hpt;

    if (hpt && prop.concurrentManagedAccess) {
        p.type       = PLAT_HW_COHERENT_UMA;
        p.peak_bw_gbs = 0.0;   /* Memory clock N/A on GB10 driver 580.142 confirmed. Peak omitted. */
    } else if (prop.concurrentManagedAccess) {
        p.type = PLAT_DISCRETE_PCIE;
        /* Estimate peak by SM generation */
        int bus_bits = 0, mem_khz = 0;
        cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, device);
        cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, device);
        p.peak_bw_gbs = (bus_bits > 0 && mem_khz > 0) ?
            2.0 * (bus_bits / 8.0) * mem_khz * 1000.0 / 1e9 : 0.0;
    } else {
        p.type        = PLAT_SOFTWARE_UMA;
        p.peak_bw_gbs = 0.0;
    }
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
        return "HW_COHERENT_UMA: One LPDDR5X pool. NVLink-C2C. Peak BW not reported (memory clock N/A on this platform).";
    case PLAT_DISCRETE_PCIE:
        return "DISCRETE_PCIE: Separate VRAM. CPU via PCIe. PTX .cs = true DRAM write BW.";
    case PLAT_SOFTWARE_UMA:
        return "Software-managed unified memory. "
               "Page migration overhead included in measurements.";
    default:
        return "Platform not recognized.";
    }
}

/* ------------------------------------------------------------------ */
/* GPU kernels — PTX cache operators for true hardware measurement      */
/* ------------------------------------------------------------------ */

/* Read: ld.global.cg — cache at L2 only, bypass L1
 * Forces every access to go to L2 minimum.
 * On GB10 UMA, cold pages cause hardware fault before L2 fill. */
__global__ void gpu_read_kernel(const float * __restrict__ buf,
                                 float *sink, size_t n) {
    size_t idx    = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    float  acc    = 0.0f;
    for (size_t i = idx; i < n; i += stride) {
        float val;
        asm volatile("ld.global.cg.f32 %0, [%1];"
                     : "=f"(val) : "l"(buf + i));
        acc += val;
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) *sink = acc;
}

/* Write: st.global.cs — cache streaming, bypass L2
 * Forces writes directly toward DRAM.
 * Without .cs, write hits L2 at L2 bandwidth — not DRAM bandwidth.
 * This is the difference between 272 GB/s (L2) and 66 GB/s (DRAM). */
__global__ void gpu_write_kernel(float *buf, size_t n, float val) {
    size_t idx    = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = idx; i < n; i += stride)
        asm volatile("st.global.cs.f32 [%0], %1;"
                     : : "l"(buf + i), "f"(val));
}

/* Copy: .cg read + .cs write — true read+write DRAM bandwidth */
__global__ void gpu_copy_kernel(float * __restrict__ dst,
                                 const float * __restrict__ src,
                                 size_t n) {
    size_t idx    = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = idx; i < n; i += stride) {
        float val;
        asm volatile("ld.global.cg.f32 %0, [%1];"
                     : "=f"(val) : "l"(src + i));
        asm volatile("st.global.cs.f32 [%0], %1;"
                     : : "l"(dst + i), "f"(val));
    }
}

/* ------------------------------------------------------------------ */
/* CPU measurement — direct C, no CUDA overhead                         */
/* ------------------------------------------------------------------ */

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Single-threaded CPU read
 * On DISCRETE_PCIE: reads through PCIe to GPU VRAM after prefetch
 * On HW_COHERENT_UMA: reads from same LPDDR5X pool as GPU
 * This difference is why CPU read numbers matter on GB10 */
static double cpu_read_bw(const float *buf, size_t n) {
    volatile float sink = 0;
    double t0 = now_sec();
    float acc = 0;
    for (size_t i = 0; i < n; i++) acc += buf[i];
    sink = acc; (void)sink;
    return (double)(n * sizeof(float)) / (now_sec() - t0) / 1e9;
}

/* CPU write via memset
 * On HW_COHERENT_UMA: writes to shared pool, visible to GPU immediately
 * On DISCRETE_PCIE: writes to system RAM, GPU must migrate pages */
static double cpu_write_bw(float *buf, size_t n) {
    double t0 = now_sec();
    memset(buf, 0, n * sizeof(float));
    return (double)(n * sizeof(float)) / (now_sec() - t0) / 1e9;
}

/* GPU event-based measurement wrapper */
static double gpu_timed_run(void (*fn)(float*, size_t, float),
                             float *buf, size_t n, float val) {
    cudaEvent_t ev_s, ev_e;
    CUDA_CHECK(cudaEventCreate(&ev_s));
    CUDA_CHECK(cudaEventCreate(&ev_e));
    CUDA_CHECK(cudaEventRecord(ev_s));
    fn(buf, n, val);
    CUDA_CHECK(cudaEventRecord(ev_e));
    CUDA_CHECK(cudaEventSynchronize(ev_e));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_s, ev_e));
    CUDA_CHECK(cudaEventDestroy(ev_s));
    CUDA_CHECK(cudaEventDestroy(ev_e));
    return (double)(n * sizeof(float)) / (ms / 1000.0) / 1e9;
}

static float *g_sink = nullptr;


static void launch_read(float *a, size_t n, float v) {
    (void)v;
    int blk_r = (int)((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    gpu_read_kernel<<<blk_r, THREADS_PER_BLOCK>>>(a, g_sink, n);
}
static void launch_write(float *a, size_t n, float v) {
    int blk_w = (int)((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    gpu_write_kernel<<<blk_w, THREADS_PER_BLOCK>>>(a, n, v);
}

/* Concurrent CPU thread */
typedef struct { float *buf; size_t n; double bw; } CpuArg;

static void *cpu_thread_fn(void *arg) {
    CpuArg *a = (CpuArg *)arg;
    double s = 0;
    for (int i = 0; i < WARMUP_RUNS; i++) cpu_read_bw(a->buf, a->n);
    for (int i = 0; i < MEASURE_RUNS; i++) s += cpu_read_bw(a->buf, a->n);
    a->bw = s / MEASURE_RUNS;
    return NULL;
}

/* ------------------------------------------------------------------ */
/* JSON output — engineer-shareable diagnostic log                      */
/* ------------------------------------------------------------------ */

static void iso_ts(char *buf, size_t len) {
    time_t t = time(NULL);
    strftime(buf, len, "%Y-%m-%dT%H:%M:%SZ", gmtime(&t));
}

static void write_json(const char *path,
                       const Platform *p,
                       const Results  *r) {
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path); return; }

    char ts[64]; iso_ts(ts, sizeof(ts));

    fprintf(f, "{\n");
    fprintf(f, "  \"tool\": \"uma-bandwidth-test\",\n");
    fprintf(f, "  \"version\": \"%s\",\n", TOOL_VERSION);
    fprintf(f, "  \"timestamp\": \"%s\",\n", ts);

    fprintf(f, "  \"platform\": {\n");
    fprintf(f, "    \"gpu_name\": \"%s\",\n",      p->name);
    fprintf(f, "    \"sm_major\": %d,\n",           p->sm_major);
    fprintf(f, "    \"sm_minor\": %d,\n",           p->sm_minor);
    fprintf(f, "    \"uma_type\": \"%s\",\n",       plat_name(p->type));
    fprintf(f, "    \"hw_coherent\": %s,\n",
            p->hw_coherent ? "true" : "false");
    fprintf(f, "    \"host_page_tables\": %s,\n",
            p->host_page_tables ? "true" : "false");
    fprintf(f, "    \"peak_bw_gbs\": %.1f,\n",     p->peak_bw_gbs);
    fprintf(f, "    \"l2_cache_kb\": %d,\n",        p->l2_kb);
    fprintf(f, "    \"clock_mhz\": %d,\n",          p->clock_mhz);
    fprintf(f, "    \"buffer_gb\": %d\n",            BUFFER_GB);
    fprintf(f, "  },\n");

    fprintf(f, "  \"results\": {\n");
    fprintf(f, "    \"gpu_read_gbs\":        %.2f,\n", r->gpu_read.mean);
    fprintf(f, "    \"gpu_read_stddev\":     %.2f,\n", r->gpu_read.stddev);
    fprintf(f, "    \"gpu_write_gbs\":       %.2f,\n", r->gpu_write.mean);
    fprintf(f, "    \"gpu_write_stddev\":    %.2f,\n", r->gpu_write.stddev);
    fprintf(f, "    \"gpu_copy_gbs\":        %.2f,\n", r->gpu_copy.mean);
    fprintf(f, "    \"cpu_read_gbs\":        %.2f,\n", r->cpu_read.mean);
    fprintf(f, "    \"cpu_read_stddev\":     %.2f,\n", r->cpu_read.stddev);
    fprintf(f, "    \"cpu_write_gbs\":       %.2f,\n", r->cpu_write.mean);
    fprintf(f, "    \"concurrent_gpu_gbs\":  %.2f,\n", r->conc_gpu);
    fprintf(f, "    \"concurrent_cpu_gbs\":  %.2f,\n", r->conc_cpu);
    fprintf(f, "    \"concurrent_total_gbs\": %.2f\n",  r->conc_total);
    fprintf(f, "  },\n");

    fprintf(f, "  \"interpretation\": {\n");
    fprintf(f, "    \"gpu_read_pct_peak\": %.1f,\n",
            p->peak_bw_gbs > 0 ?
            r->gpu_read.mean / p->peak_bw_gbs * 100.0 : 0.0);
    fprintf(f, "    \"platform_note\": \"%s\",\n", plat_note(p->type));
    fprintf(f, "    \"ptx_read\":  \"ld.global.cg (L1 bypass)\",\n");
    fprintf(f, "    \"ptx_write\": \"st.global.cs (L2 bypass, true DRAM)\"\n");
    fprintf(f, "  },\n");

    fprintf(f, "  \"raw_runs\": {\n");
    fprintf(f, "    \"gpu_read\":  [");
    for (int i = 0; i < MEASURE_RUNS; i++)
        fprintf(f, "%.2f%s", r->gpu_read.runs[i],
                i < MEASURE_RUNS-1 ? "," : "");
    fprintf(f, "],\n");
    fprintf(f, "    \"gpu_write\": [");
    for (int i = 0; i < MEASURE_RUNS; i++)
        fprintf(f, "%.2f%s", r->gpu_write.runs[i],
                i < MEASURE_RUNS-1 ? "," : "");
    fprintf(f, "],\n");
    fprintf(f, "    \"cpu_read\":  [");
    for (int i = 0; i < MEASURE_RUNS; i++)
        fprintf(f, "%.2f%s", r->cpu_read.runs[i],
                i < MEASURE_RUNS-1 ? "," : "");
    fprintf(f, "]\n");
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
    Results  r = {};

    if (!json_only) {
        printf("=== UMA Bandwidth Test v%s ===\n", TOOL_VERSION);
        printf("GPU      : %s (SM %d.%d)\n",
               p.name, p.sm_major, p.sm_minor);
        printf("Platform : %s\n", plat_name(p.type));
        printf("Coherent : %s\n",
               p.hw_coherent ? "yes (hardware)" : "no");
        printf("Peak     : %.0f GB/s theoretical\n", p.peak_bw_gbs);
        printf("Buffer   : %d GB\n", BUFFER_GB);
        printf("Runs     : %d warmup + %d measured\n",
               WARMUP_RUNS, MEASURE_RUNS);
        printf("PTX read : ld.global.cg (L1 bypass)\n");
        printf("PTX write: st.global.cs (L2 bypass, true DRAM)\n");
        printf("Note     : %s\n\n", plat_note(p.type));
    }

    size_t n = BUFFER_BYTES / sizeof(float);
    float *buf_a, *buf_b, *sink;
    CUDA_CHECK(cudaMallocManaged(&buf_a, BUFFER_BYTES));
    CUDA_CHECK(cudaMallocManaged(&buf_b, BUFFER_BYTES));
    CUDA_CHECK(cudaMallocManaged(&sink,  sizeof(float)));
    g_sink = sink;

    if (!json_only) { printf("Initializing..."); fflush(stdout); }
    memset(buf_a, 0, BUFFER_BYTES);
    memset(buf_b, 1, BUFFER_BYTES);
    if (!json_only) printf(" done\n\n");

    /* --- GPU side --- */
    if (!json_only) printf("--- GPU (prefetched to GPU) ---\n");
    CUDA_CHECK(cudaMemPrefetchAsync(buf_a, BUFFER_BYTES, device, 0));
    CUDA_CHECK(cudaMemPrefetchAsync(buf_b, BUFFER_BYTES, device, 0));
    CUDA_CHECK(cudaDeviceSynchronize());

    /* GPU read */
    for (int i = 0; i < WARMUP_RUNS; i++) launch_read(buf_a, n, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < MEASURE_RUNS; i++)
        r.gpu_read.runs[i] = gpu_timed_run(launch_read, buf_a, n, 0);
    stat_compute(&r.gpu_read);
    if (!json_only)
        printf("GPU read  : %7.2f GB/s  stddev %.2f\n",
               r.gpu_read.mean, r.gpu_read.stddev);

    /* GPU write */
    for (int i = 0; i < WARMUP_RUNS; i++) launch_write(buf_a, n, 3.14f);
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < MEASURE_RUNS; i++)
        r.gpu_write.runs[i] = gpu_timed_run(launch_write,
                                             buf_a, n, 3.14f);
    stat_compute(&r.gpu_write);
    if (!json_only)
        printf("GPU write : %7.2f GB/s  stddev %.2f  [PTX .cs]\n",
               r.gpu_write.mean, r.gpu_write.stddev);

    /* GPU copy */
    if (!json_only) { printf("GPU copy..."); fflush(stdout); }
    cudaEvent_t ev_s, ev_e; float ms = 0;
    CUDA_CHECK(cudaEventCreate(&ev_s));
    CUDA_CHECK(cudaEventCreate(&ev_e));
    for (int i = 0; i < WARMUP_RUNS; i++)
        gpu_copy_kernel<<<(int)((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(buf_b,buf_a,n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ev_s));
    for (int i = 0; i < MEASURE_RUNS; i++)
        gpu_copy_kernel<<<(int)((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK),THREADS_PER_BLOCK>>>(buf_b,buf_a,n);
    CUDA_CHECK(cudaEventRecord(ev_e));
    CUDA_CHECK(cudaEventSynchronize(ev_e));
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_s, ev_e));
    CUDA_CHECK(cudaEventDestroy(ev_s));
    CUDA_CHECK(cudaEventDestroy(ev_e));
    r.gpu_copy.mean = (double)(n*sizeof(float)*2*MEASURE_RUNS)
                      / (ms/1000.0) / 1e9;
    if (!json_only)
        printf(" %7.2f GB/s  [read+write]\n\n", r.gpu_copy.mean);

    /* --- CPU side --- */
    if (!json_only) printf("--- CPU (prefetched to CPU) ---\n");
    CUDA_CHECK(cudaMemPrefetchAsync(buf_a, BUFFER_BYTES,
               cudaCpuDeviceId, 0));
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < WARMUP_RUNS; i++) cpu_read_bw(buf_a, n);
    for (int i = 0; i < MEASURE_RUNS; i++)
        r.cpu_read.runs[i] = cpu_read_bw(buf_a, n);
    stat_compute(&r.cpu_read);
    if (!json_only)
        printf("CPU read  : %7.2f GB/s  stddev %.2f\n",
               r.cpu_read.mean, r.cpu_read.stddev);

    for (int i = 0; i < WARMUP_RUNS; i++) cpu_write_bw(buf_a, n);
    for (int i = 0; i < MEASURE_RUNS; i++)
        r.cpu_write.runs[i] = cpu_write_bw(buf_a, n);
    stat_compute(&r.cpu_write);
    if (!json_only)
        printf("CPU write : %7.2f GB/s\n\n", r.cpu_write.mean);

    /* --- Concurrent CPU + GPU --- */
    if (!json_only) { printf("--- Concurrent CPU + GPU ---\n"); printf("measuring...\n"); fflush(stdout); }
    CUDA_CHECK(cudaMemPrefetchAsync(buf_a,
               BUFFER_BYTES/2, device, 0));
    CUDA_CHECK(cudaMemPrefetchAsync(buf_a + n/2,
               BUFFER_BYTES/2, cudaCpuDeviceId, 0));
    CUDA_CHECK(cudaDeviceSynchronize());

    CpuArg cpu_arg = { buf_a + n/2, n/2, 0.0 };
    pthread_t tid;
    pthread_create(&tid, NULL, cpu_thread_fn, &cpu_arg);

    double conc_runs[MAX_RUNS] = {};
    for (int i = 0; i < WARMUP_RUNS; i++) launch_read(buf_a, n/2, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < MEASURE_RUNS; i++)
        conc_runs[i] = gpu_timed_run(launch_read, buf_a, n/2, 0);
    r.conc_gpu = stat_mean(conc_runs, MEASURE_RUNS);

    pthread_join(tid, NULL);
    r.conc_cpu   = cpu_arg.bw;
    r.conc_total = r.conc_gpu + r.conc_cpu;

    if (!json_only) {
        printf("GPU concurrent: %7.2f GB/s\n", r.conc_gpu);
        printf("CPU concurrent: %7.2f GB/s\n", r.conc_cpu);
        printf("Total         : %7.2f GB/s\n\n", r.conc_total);
    }

    /* --- Summary --- */
    if (!json_only) {
        printf("=== Summary ===\n");
        printf("GPU read  : %7.2f GB/s  (%5.1f%% of %.0f GB/s peak)\n",
               r.gpu_read.mean,
               p.peak_bw_gbs > 0 ?
               r.gpu_read.mean / p.peak_bw_gbs * 100.0 : 0.0,
               p.peak_bw_gbs);
        printf("GPU write : %7.2f GB/s  [PTX .cs — true DRAM]\n",
               r.gpu_write.mean);
        printf("GPU copy  : %7.2f GB/s\n", r.gpu_copy.mean);
        printf("CPU read  : %7.2f GB/s\n", r.cpu_read.mean);
        printf("CPU write : %7.2f GB/s\n", r.cpu_write.mean);
        printf("Conc total: %7.2f GB/s\n", r.conc_total);
        printf("\nPlatform  : %s\n", plat_name(p.type));
        printf("JSON      : %s\n", JSON_OUTPUT);
    }

    write_json(JSON_OUTPUT, &p, &r);
    if (!json_only) printf("Done.\n");

    CUDA_CHECK(cudaFree(buf_a));
    CUDA_CHECK(cudaFree(buf_b));
    CUDA_CHECK(cudaFree(sink));
    return 0;
}
