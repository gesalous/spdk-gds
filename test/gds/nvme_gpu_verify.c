#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdbool.h>
#include <fcntl.h>
#include <pthread.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <gdrapi.h>

// SPDK & DPDK Headers
#include "spdk/env.h"
#include "spdk/nvme.h"
#include "spdk/vmd.h"
#include "spdk/stdinc.h"
#include <rte_config.h>
#include <rte_vfio.h>

#define GPU_ID 0
#define BUFFER_SIZE (64 * 1024 * 1024) // 2MB
#define GPUMAP_PATH "/proc/driver/gdrdrv/gpumap"
#define MAX_GDR_MAPPINGS 64

// ==========================================
// GDR Mapping Helper Logic
// ==========================================
typedef struct {
    uint64_t cpu_va_start;
    uint64_t gpu_va_start;
    uint64_t length;
} gdr_mapping_t;

static gdr_mapping_t g_cache[MAX_GDR_MAPPINGS];
static int g_cache_entries = 0;
static bool g_cache_populated = false;
static pthread_mutex_t g_cache_mutex = PTHREAD_MUTEX_INITIALIZER;

static void gdr_gpumap_populate_locked(void) {
    FILE *fp = fopen(GPUMAP_PATH, "r");
    char line[256];
    pid_t pid = getpid();

    g_cache_entries = 0;
    if (!fp) {
        perror("Failed to open gpumap");
        g_cache_populated = true;
        return;
    }

    while (fgets(line, sizeof(line), fp) && g_cache_entries < MAX_GDR_MAPPINGS) {
        int tgid;
        uint64_t cpu_va, gpu_va, len;
        if (sscanf(line, "tgid=%d cpu_va=0x%lx gpu_va=0x%lx len=%lu",
                   &tgid, &cpu_va, &gpu_va, &len) == 4 && tgid == pid) {
            g_cache[g_cache_entries].cpu_va_start = cpu_va;
            g_cache[g_cache_entries].gpu_va_start = gpu_va;
            g_cache[g_cache_entries].length = len;
            g_cache_entries++;
        }
    }
    fclose(fp);
    g_cache_populated = true;
}

static uint64_t gdr_gpumap_translate_cpu_va(uint64_t cpu_va) {
    pthread_mutex_lock(&g_cache_mutex);
    if (!g_cache_populated) {
        gdr_gpumap_populate_locked();
    }
    for (int i = 0; i < g_cache_entries; ++i) {
        uint64_t start = g_cache[i].cpu_va_start;
        uint64_t end = start + g_cache[i].length;
        if (cpu_va >= start && cpu_va < end) {
            uint64_t pa = g_cache[i].gpu_va_start + (cpu_va - start);
            pthread_mutex_unlock(&g_cache_mutex);
            return pa;
        }
    }
    pthread_mutex_unlock(&g_cache_mutex);
    return UINT64_MAX;
}

// ==========================================
// Main Logic
// ==========================================

struct app_context {
    void *bar_ptr;      // Virtual Address
    uint64_t phys_addr; // Physical Address
    bool vfio_mapped;
    struct spdk_nvme_ctrlr *ctrlr;
    struct spdk_nvme_ns *ns;
};

bool g_io_complete = false;

static void io_complete_cb(void *arg, const struct spdk_nvme_cpl *completion) {
    g_io_complete = true;
    if (spdk_nvme_cpl_is_error(completion)) {
        fprintf(stderr, "NVMe I/O Failed! SCT: %d, SC: %d\n", 
                completion->status.sct, completion->status.sc);
    }
}

static void attach_cb(void *cb_ctx, const struct spdk_nvme_transport_id *trid,
                      struct spdk_nvme_ctrlr *ctrlr, const struct spdk_nvme_ctrlr_opts *opts) {
    
    struct app_context *ctx = (struct app_context *)cb_ctx;
    ctx->ctrlr = ctrlr;
    ctx->ns = spdk_nvme_ctrlr_get_ns(ctrlr, 1);
    printf("Attached to NVMe: %s\n", trid->traddr);

    // VFIO MAP inside Callback
    printf("Mapping GPU memory to VFIO container...\n");
    int vfio_rc = rte_vfio_container_dma_map(
        RTE_VFIO_DEFAULT_CONTAINER_FD,
        (uint64_t)ctx->bar_ptr,  // vaddr
        0x900000000000,// ctx->phys_addr,          // iova
        BUFFER_SIZE
    );

    if (vfio_rc != 0) {
        fprintf(stderr, "FATAL: VFIO Map Failed (rc=%d)\n", vfio_rc);
        ctx->vfio_mapped = false;
    } else {
        printf("SUCCESS: VFIO Container Map Created.\n");
        ctx->vfio_mapped = true;
    }
    // 3. Register with SPDK (Now that VFIO is mapped, this should succeed)
    // We tell SPDK: "Here is a memory range. Trust us, it's mapped."
    if (spdk_mem_register(ctx->bar_ptr, BUFFER_SIZE) != 0) {
        fprintf(stderr, "Warning: spdk_mem_register failed even after VFIO map.\n");
        // We don't return -1 here because the VFIO map might be enough for the HW.
    }

}

static bool probe_cb(void *cb_ctx, const struct spdk_nvme_transport_id *trid,
                     struct spdk_nvme_ctrlr_opts *opts) {
    return true;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <pci_addr>\n", argv[0]);
        return 1;
    }

    struct app_context app_ctx = {0};

    // 1. Initialize SPDK
    struct spdk_env_opts opts;
    spdk_env_opts_init(&opts);
    opts.name = "p2p_verify";
    if (spdk_env_init(&opts) < 0) return 1;

    // 2. Initialize CUDA & GDRCopy
    CUdevice dev;
    CUcontext ctx;
    gdr_t gdr;
    gdr_mh_t mh;
    CUdeviceptr d_ptr;

    cuInit(0);
    cuDeviceGet(&dev, GPU_ID);
    if (cuCtxCreate(&ctx,NULL, 0, dev) != CUDA_SUCCESS) return 1;

    gdr = gdr_open();
    cuMemAlloc(&d_ptr, BUFFER_SIZE);
    
    // Pin and Map
    gdr_pin_buffer(gdr, d_ptr, BUFFER_SIZE, 0, 0, &mh);
    gdr_map(gdr, mh, &app_ctx.bar_ptr, BUFFER_SIZE);

    // 3. Resolve Physical Address
    app_ctx.phys_addr = gdr_gpumap_translate_cpu_va((uint64_t)app_ctx.bar_ptr);
    if (app_ctx.phys_addr == UINT64_MAX) {
        fprintf(stderr, "Failed to resolve GPU physical address.\n");
        return 1;
    }

    // 4. NVMe Probe (Triggers VFIO map in callback)
    struct spdk_nvme_transport_id trid = {};
    spdk_nvme_transport_id_parse(&trid, argv[1]);
    trid.trtype = SPDK_NVME_TRANSPORT_PCIE;
    
    if (spdk_nvme_probe(&trid, &app_ctx, probe_cb, attach_cb, NULL) != 0) return 1;
    if (!app_ctx.vfio_mapped) return 1;

    struct spdk_nvme_qpair *qpair = spdk_nvme_ctrlr_alloc_io_qpair(app_ctx.ctrlr, NULL, 0);
    char *cpu_buf = spdk_dma_zmalloc(BUFFER_SIZE, 4096, NULL);
    char *verify_buf = malloc(BUFFER_SIZE);

    // ---------------------------------------------------------
    // TEST 1: READ VERIFICATION (SSD -> GPU)
    // ---------------------------------------------------------
    printf("\n--- TEST 1: READ P2P (SSD -> GPU) ---\n");
    
    // A. Setup: CPU -> SSD (0xAA)
    memset(cpu_buf, 0xAA, BUFFER_SIZE);
    printf("[1.1] Seeding SSD with 0xAA (CPU Write)...\n");
    g_io_complete = false;
    spdk_nvme_ns_cmd_write(app_ctx.ns, qpair, cpu_buf, 0, BUFFER_SIZE/512, io_complete_cb, NULL, 0);
    while (!g_io_complete) spdk_nvme_qpair_process_completions(qpair, 0);

    // B. P2P Read: SSD -> GPU
    uint32_t zero = 0;
    cuMemsetD32(d_ptr, zero, BUFFER_SIZE / 4); // Clear GPU
    cuCtxSynchronize();

    printf("[1.2] Reading to GPU (P2P Read)...\n");
    g_io_complete = false;
    int rc = spdk_nvme_ns_cmd_read(app_ctx.ns, qpair, app_ctx.bar_ptr, 0, BUFFER_SIZE/512, io_complete_cb, NULL, 0);
    if(rc != 0) { fprintf(stderr, "Read Failed rc=%d\n", rc); return 1; }
    while (!g_io_complete) spdk_nvme_qpair_process_completions(qpair, 0);

    // C. Verify
    cuMemcpyDtoH(verify_buf, d_ptr, BUFFER_SIZE);
    if (verify_buf[0] == (char)0xAA) printf(">>> SUCCESS: P2P Read Verified (0xAA)\n");
    else printf(">>> FAILURE: P2P Read Mismatch (Got 0x%02hhX)\n", verify_buf[0]);


    // ---------------------------------------------------------
    // TEST 2: WRITE VERIFICATION (GPU -> SSD)
    // ---------------------------------------------------------
    printf("\n--- TEST 2: WRITE P2P (GPU -> SSD) ---\n");

    // A. Setup: GPU memory (0xBB)
    printf("[2.1] Filling GPU memory with 0xBB...\n");
    memset(verify_buf, 0xBB, BUFFER_SIZE); // Reuse verify buf to init pattern
    cuMemcpyHtoD(d_ptr, verify_buf, BUFFER_SIZE);
    cuCtxSynchronize();

    // B. P2P Write: GPU -> SSD
    printf("[2.2] Writing from GPU (P2P Write)...\n");
    g_io_complete = false;
    rc = spdk_nvme_ns_cmd_write(app_ctx.ns, qpair, app_ctx.bar_ptr, 0, BUFFER_SIZE/512, io_complete_cb, NULL, 0);
    if(rc != 0) { fprintf(stderr, "Write Failed rc=%d\n", rc); return 1; }
    while (!g_io_complete) spdk_nvme_qpair_process_completions(qpair, 0);

    // C. Verify: SSD -> CPU
    memset(cpu_buf, 0x00, BUFFER_SIZE); // Clear CPU buffer
    printf("[2.3] Reading back to CPU to verify...\n");
    g_io_complete = false;
    spdk_nvme_ns_cmd_read(app_ctx.ns, qpair, cpu_buf, 0, BUFFER_SIZE/512, io_complete_cb, NULL, 0);
    while (!g_io_complete) spdk_nvme_qpair_process_completions(qpair, 0);

    if (cpu_buf[0] == (char)0xBB) printf(">>> SUCCESS: P2P Write Verified (0xBB)\n");
    else printf(">>> FAILURE: P2P Write Mismatch (Got 0x%02hhX)\n", cpu_buf[0]);

    // ---------------------------------------------------------
    // Cleanup
    // ---------------------------------------------------------
    spdk_mem_unregister(app_ctx.bar_ptr,BUFFER_SIZE);
    rte_vfio_container_dma_unmap(RTE_VFIO_DEFAULT_CONTAINER_FD, (uint64_t)app_ctx.bar_ptr, 0x900000000000, BUFFER_SIZE);
    spdk_dma_free(cpu_buf);
    free(verify_buf);
    gdr_unmap(gdr, mh, app_ctx.bar_ptr, BUFFER_SIZE);
    gdr_unpin_buffer(gdr, mh);
    gdr_close(gdr);

    return 0;
}