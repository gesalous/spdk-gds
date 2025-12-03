#include <stdio.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <getopt.h>
#include <time.h>
#include <pthread.h>
#include <sched.h>



#include <cuda.h>
#include <gdrapi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "spdk/env.h"
#include "spdk/log.h"
#include "spdk/nvme.h"
#include "spdk/stdinc.h"
#include "spdk/string.h"
#include "spdk/vmd.h"
#include "spdk/cpuset.h" 
#include <cuda_runtime.h>
// #include <stdatomic.h>

// extern atomic_ullong g_vtophys_hook_successes;



/* =================================================================================================
 *                                              Configuration
 * ================================================================================================= */




#define GPU_ID 0



typedef struct {
    char *method;
    char *pci_addr;
    int gpu_id;
    size_t io_size;
    uint64_t total_size;
    int queue_depth;

    char *op;
    char *pattern;

    int num_threads;
    bool display_percentiles;
} benchmark_opts_t;

/**
 * @brief Holds resources that are initialized once and shared across all
 * worker threads.
 */
typedef struct global_state {
    //these are initiated once since we only have one drive and one GPU we are going to be talking to

    //Shared SPDK resources
    struct spdk_nvme_ctrlr* ctrlr;
    struct spdk_nvme_ns* ns;
    bool spdk_attached;

    //Shared CUDA/GDR resources
    CUcontext   cuda_context;
    //A pointer to the command-line options for easy access
    benchmark_opts_t* opts;

    //used for sync
    pthread_barrier_t barrier;
    pthread_mutex_t mutex;

} global_state_t;


typedef struct io_context io_context_t;


/**
 * @brief Holds all resources and state for a single worker thread.
 * Each thread gets its own instance of this struct.
 */

typedef struct{
    int t_id;

    struct spdk_nvme_qpair *qpair;  //individual queue per thread
    
    //will be allocated by spdk
    void*   cpu_dma_buffer;

    CUdeviceptr gpu_dest_buffer;

    // Per-thread benchmark state
    volatile uint64_t completed_ios;
    struct timespec start_time;
    struct timespec end_time;

    volatile int op_status; //0 = OK, -1 = error

    uint64_t*   latencies;              // Array to store latency values in cycles
    uint64_t    latencies_recorded;     // How many results we've stored
    io_context_t*   io_contexts;        // A pre-allocated pool of contexts
    uint64_t        total_ios;          //total ios a thread will do

    global_state_t* gstate; //pointer back to the shared state
    uint64_t rand_state;    //helper for rand 

}thread_ctx_t;

//a small helper struct to hold the start of each io
struct io_context {
    uint64_t        start_cycle;
    thread_ctx_t*   thread_ctx; //pointer back to the parent thread
    void*           cpu_buffer_slot;
};

/****DEPRICATED****/
typedef struct app_state{
    //spdk
    struct spdk_nvme_ctrlr* ctrlr;
    struct spdk_nvme_ns* ns;
    struct spdk_nvme_qpair* qpair;
    
    //cuda gdrcopy
    CUcontext context;
    CUdeviceptr d_ptr;
    gdr_t gdr_handle;
    gdr_mh_t gdr_mh;
    bool buf_pinned;



    void* bar_ptr; //the CPU-mapped GPU pointer
    size_t io_size;

    size_t mapped_size;


    volatile bool spdk_attached;
    volatile uint64_t completed_ios;
    volatile int op_status; //0 success, -1 error
    volatile int test_active;
}app_state_t;


void print_usage(const char *prog_name);
int init_global_spdk(benchmark_opts_t *opts, global_state_t *state);
int init_global_cuda_gdr(benchmark_opts_t *opts, global_state_t *state);
int prepare_gpu_data(global_state_t *ctx);
void *benchmark_thread_entry(void *arg);
static void write_complete_cb(void *arg, const struct spdk_nvme_cpl *completion);
static void read_complete_cb(void *arg, const struct spdk_nvme_cpl *completion);
static void global_attach_cb(void *cb_ctx, const struct spdk_nvme_transport_id *trid, struct spdk_nvme_ctrlr *ctrlr, const struct spdk_nvme_ctrlr_opts *opts);
static bool probe_cb(void *cb_ctx, const struct spdk_nvme_transport_id *trid, struct spdk_nvme_ctrlr_opts *opts);
static inline uint64_t read_tsc(void);
/* =================================================================================================
 *                                              SPDK callbacks
 * ================================================================================================= */


static bool probe_cb(void *cb_ctx, const struct spdk_nvme_transport_id *trid, struct spdk_nvme_ctrlr_opts *opts) {
    printf("Probing device at %s...\n", trid->traddr);
    return true;
}

static void global_attach_cb(void *cb_ctx, const struct spdk_nvme_transport_id *trid, struct spdk_nvme_ctrlr *ctrlr, const struct spdk_nvme_ctrlr_opts *opts) {
    global_state_t *state = (global_state_t*) cb_ctx;
    state->ctrlr = ctrlr;
    state->ns = spdk_nvme_ctrlr_get_ns(ctrlr, 1);
    printf("Attached to %s\n", trid->traddr);
    state->spdk_attached = true;
}

static void write_complete_cb(void *arg, const struct spdk_nvme_cpl *completion) {
    uint64_t end_cycle = read_tsc();
    io_context_t *ioctx = (io_context_t *)arg;
    thread_ctx_t *ctx = ioctx->thread_ctx;

    // It's still important to increment the completion counter first,
    // so the main loop knows this I/O is done.
    ctx->completed_ios++;

    // Check for SPDK errors.
    if (spdk_nvme_cpl_is_error(completion)) {
        fprintf(stderr, "Thread %d: SPDK Write I/O failed! Status: %s\n",
                ctx->t_id,
                spdk_nvme_cpl_get_status_string(&completion->status));
        ctx->op_status = -1;
        // Don't record latency for a failed I/O.
        return;
    }

    // The operation was successful. Now, calculate and record the latency.
    uint64_t lat_cycles = end_cycle - ioctx->start_cycle;
    if(ctx->latencies_recorded < ctx->total_ios){
        ctx->latencies[ctx->latencies_recorded++] = lat_cycles;
    }
}

static void read_complete_cb(void *arg, const struct spdk_nvme_cpl *completion) {
    
    io_context_t *ioctx = (io_context_t *)arg;
    thread_ctx_t *ctx = ioctx->thread_ctx;
    benchmark_opts_t *opts = ctx->gstate->opts;
    
    if(spdk_nvme_cpl_is_error(completion)){
        fprintf(stderr, "Thread %d: Read I/O failed! Status: %s\n",
                ctx->t_id,
                spdk_nvme_cpl_get_status_string(&completion->status));
        ctx->op_status = -1;
    }

    // 1. Calculate the correct destination slot in the GPU buffer.
    size_t buffer_slot_offset = (char*)ioctx->cpu_buffer_slot - (char*)ctx->cpu_dma_buffer;
    CUdeviceptr gpu_dest_slot = ctx->gpu_dest_buffer + buffer_slot_offset;

    // 2. Perform the synchronous memory copy from Host (CPU) to Device (GPU).
    CUresult res = cuMemcpyHtoD(gpu_dest_slot, ioctx->cpu_buffer_slot, opts->io_size);

    // 3. Stop the timer *after* the GPU copy is complete. This is the true end-to-end time.
    uint64_t end_cycle = read_tsc();

    // Now, handle the completion logic.
    ctx->completed_ios++;

    if(res != CUDA_SUCCESS){
        fprintf(stderr, "Thread %d: cuMemcpyHtoD failed after SPDK read! CUDA error code: %d\n", ctx->t_id, res);
        ctx->op_status = -1; // Mark the benchmark as failed
    } 
    else{
        // The full operation was successful. Record the latency.
        uint64_t lat_cycles = end_cycle - ioctx->start_cycle;
        if(ctx->latencies_recorded < ctx->total_ios){
            ctx->latencies[ctx->latencies_recorded++] = lat_cycles;
        }
    }
}


/* =================================================================================================
 *                                         Init and cleanup
 * ================================================================================================= */

int init_global_spdk(benchmark_opts_t *opts, global_state_t *state){
    struct spdk_env_opts spdk_opts;
    spdk_env_opts_init(&spdk_opts);
    spdk_opts.core_mask = "0x1";
    spdk_opts.name = "nvme2gpu-bench";
    
    if(spdk_env_init(&spdk_opts)<0){
        fprintf(stderr,"Failed to init SPDK\n");
        return -1;
    }

    struct spdk_nvme_transport_id trid = {};
    trid.trtype = SPDK_NVME_TRANSPORT_PCIE; //set the transport type

    if (spdk_nvme_transport_id_parse_trtype(&trid, opts->pci_addr) != 0) {
        fprintf(stderr, "Failed to parse PCIe address: %s\n", opts->pci_addr);
        return -1;
    }

    //try to attach the drive
    if(spdk_nvme_probe(&trid, state, probe_cb, global_attach_cb, NULL) != 0){
        fprintf(stderr, "Failed to probe for NVMe devices\n");
        return -1;
    }
    
    //spin until the ctrler is attached
    while(!state->ctrlr || !state->ns) usleep(10000);
    
    return (state->ctrlr && state->ns) ? 0 : -1;
}

int init_global_cuda_gdr(benchmark_opts_t *opts, global_state_t *state){
    CUdevice dev;
    cuInit(0);
    cuDeviceGet(&dev, opts->gpu_id);
    if(cuCtxCreate(&state->cuda_context, NULL, 0, dev) != CUDA_SUCCESS) return -1;
    
    return 0;
}


/* =================================================================================================
 *                                          Benchmarks
 * ================================================================================================= */

//helper
static inline uint64_t rand64(void) {
    uint64_t r = 0;
    //Call rand() 3 times to get enough bits for a large LBA
    r = (uint64_t)rand() << 30 | (uint64_t)rand() << 15 | (uint64_t)rand();
    return r;
}

//helper function that ensures all prev instructions are finished before reading the counter
static inline uint64_t
read_tsc(void)
{
	uint32_t lo, hi;
	__asm__ __volatile__("rdtscp" : "=a"(lo), "=d"(hi) :: "rcx");
	return ((uint64_t)hi << 32) | lo;
}




/**
 * @brief The main function for each worker thread.
 *
 * This function is responsible for:
 * 1. Setting its CPU core affinity.
 * 2. Allocating its own SPDK queue pair and GPU buffer.
 * 3. Running its portion of the benchmark I/O.
 * 4. Cleaning up its allocated resources.
 *
 * @param arg A pointer to this thread's thread_context_t.
 * @return void* NULL on success.
 */
void *benchmark_thread_entry(void *arg) {
    thread_ctx_t *ctx = (thread_ctx_t *)arg;
    global_state_t *g_state = ctx->gstate;
    benchmark_opts_t *opts = g_state->opts;
    void *ret = NULL;

    if (cuCtxSetCurrent(g_state->cuda_context) != CUDA_SUCCESS) {
        fprintf(stderr, "ERROR: Thread %d failed to set CUDA context.\n", ctx->t_id);
        ret = (void *)-1L;
        goto exit;
    }
    
    
    //--- 1. Set CPU Affinity ---
    cpu_set_t os_cpuset;
    CPU_ZERO(&os_cpuset);
    CPU_SET(ctx->t_id + 1, &os_cpuset);
    int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &os_cpuset);
    if (rc != 0) {
        fprintf(stderr, "Warning: Thread %d failed to set CPU affinity: %s\n",
                ctx->t_id, strerror(rc));
    }
    printf("Thread %d starting on core %d\n", ctx->t_id, sched_getcpu());

    // --- 2. Allocate Per-Thread Resources ---
    ctx->qpair = spdk_nvme_ctrlr_alloc_io_qpair(g_state->ctrlr, NULL, 0);
    if (ctx->qpair == NULL) {
        fprintf(stderr, "ERROR: Thread %d failed to allocate SPDK qpair.\n", ctx->t_id);
        ret = (void *)-1L;
        goto exit; //cannot proceed without a qpair.
    }

    size_t thread_buffer_size = (size_t)opts->queue_depth * opts->io_size;

    // 1. Allocate a CPU-side buffer that SPDK will use for DMA.
    //    spdk_dma_malloc ensures it has the correct memory alignment for the hardware.
    
    ctx->cpu_dma_buffer = spdk_dma_malloc(thread_buffer_size, 4096, NULL);
    if (!ctx->cpu_dma_buffer) {
        fprintf(stderr, "ERROR: Thread %d failed to allocate SPDK DMA memory.\n", ctx->t_id);
        ret = (void *)-1L;
        goto cleanup;
    }

    // 2. Register this existing SPDK buffer with CUDA.
    // This pins the memory from CUDA's perspective, allowing high-speed DMA 
    // (avoiding the internal host-side copy).
    // CU_MEMHOSTREGISTER_PORTABLE makes it usable by all CUDA contexts.
    CUresult cu_res = cuMemHostRegister(ctx->cpu_dma_buffer, thread_buffer_size, CU_MEMHOSTREGISTER_PORTABLE);
    if (cu_res != CUDA_SUCCESS) {
        fprintf(stderr, "ERROR: Thread %d failed to register SPDK memory with CUDA. Error: %d\n", ctx->t_id, cu_res);
        // Clean up the SPDK alloc before failing
        spdk_dma_free(ctx->cpu_dma_buffer); 
        ret = (void *)-1L;
        goto cleanup;
    }

    // 2. Allocate the corresponding buffer on the GPU. This will be the
    //    source for 'write' operations and the destination for 'read' operations.
    
    if (cuMemAlloc(&ctx->gpu_dest_buffer, thread_buffer_size) != CUDA_SUCCESS) {
        fprintf(stderr, "ERROR: Thread %d failed to allocate GPU memory.\n", ctx->t_id);
        ret = (void *)-1L;
        goto cleanup;
    }
    printf("Thread %d allocated %zu B on CPU and %zu B on GPU.\n", ctx->t_id, thread_buffer_size, thread_buffer_size);
    
    
    // void* end_addr = (char*)ctx->thread_bar_ptr + thread_buffer_size;
    // printf("Thread %d: Memory range [start: %p] to [end: %p] (Size: %zu bytes)\n", ctx->t_id, ctx->thread_bar_ptr, end_addr, thread_buffer_size);

    // --- 3. Prepare Data for 'write' Benchmark ---
    
    if (strcmp(opts->op, "write") == 0) {
        printf("Thread %d preparing initial data on GPU for writing...\n", ctx->t_id);

        // Stage 1: Create the data pattern on the host (CPU) first.
        //na valw thn cumemalloc host edw 
        char* host_pattern_buffer = (char*)malloc(thread_buffer_size);
        if (!host_pattern_buffer) {
            perror("Thread failed to allocate host pattern buffer");
            ret = (void *)-1L;
            goto cleanup;
        }
        
        // Create a small pattern chunk
        char* pattern_chunk = (char*)malloc(opts->io_size);
        if (!pattern_chunk) {
            perror("Thread failed to allocate pattern chunk");
            free(host_pattern_buffer);
            ret = (void *)-1L;
            goto cleanup;
        }
        for (size_t i = 0; i < opts->io_size; i++) {
            pattern_chunk[i] = (char)(i % 256); // Same pattern as original program
        }

        // Fill the large host buffer by stamping the pattern
        
        for (size_t offset = 0; offset < thread_buffer_size; offset += opts->io_size) {
            memcpy(host_pattern_buffer + offset, pattern_chunk, opts->io_size);
        }
        free(pattern_chunk);

        // Stage 2: Copy the prepared data from the host buffer to the thread's GPU buffer.
        // This cudaMemcpy is NOT part of the timed benchmark. This is just setup.

        if (cuMemcpyHtoD(ctx->gpu_dest_buffer, host_pattern_buffer, thread_buffer_size) != CUDA_SUCCESS) {
            fprintf(stderr, "Thread %d: Failed to cuMemcpyHtoD pattern data to GPU.\n", ctx->t_id);
            free(host_pattern_buffer);
            ret = (void *)-1L;
            goto cleanup;
        }

        // The temporary host buffer is no longer needed.
        free(host_pattern_buffer);
        
        // Good practice to ensure the GPU has finished the copy before the benchmark starts.
        cuCtxSynchronize();
        printf("Thread %d: GPU buffer is ready for writing.\n", ctx->t_id);
    }


    
    /**** general calculations used by everyone ****/

    //atomic unit of storage
    uint32_t sector_size = spdk_nvme_ns_get_sector_size(g_state->ns);
    
    //how many atomic units
    uint64_t ns_size_sectors = spdk_nvme_ns_get_num_sectors(g_state->ns);
    
    //how many atomic units per io
    uint32_t n_sectors_per_io = opts->io_size / sector_size;

    /**** thread-specific calculations ****/
    
    /* total io / num_threads */
    uint64_t total_ios_for_this_thread = (opts->total_size / opts->io_size) / opts->num_threads;
    
    //divide the disk space in each thread
    uint64_t lba_lane_size = ns_size_sectors / opts->num_threads;
    
    uint64_t lba_start_for_this_thread = lba_lane_size * ctx->t_id;
    
    //safety bound for random IO
    uint64_t max_lba_offset_in_lane = lba_lane_size - n_sectors_per_io;
    
    //safety bound for seq IO
    uint64_t num_io_slots_in_lane = lba_lane_size / n_sectors_per_io;

    ctx->total_ios = total_ios_for_this_thread;

    // --- LATENCY SETUP ---
    ctx->latencies = calloc(total_ios_for_this_thread, sizeof(uint64_t));
    ctx->latencies_recorded = 0;
    ctx->io_contexts = calloc(opts->queue_depth, sizeof(io_context_t));
    for (int i = 0; i < opts->queue_depth; i++) {
        ctx->io_contexts[i].thread_ctx = ctx; // Set back-pointer
    }


    uint64_t submitted_ios = 0;
    ctx->completed_ios = 0;
    ctx->rand_state = spdk_rand_xorshift64_seed(); 
    ctx->rand_state ^= (uint64_t)ctx->t_id;  

    // printf("Thread %d starting benchmark: %lu IOs to process.\n", ctx->t_id, total_ios_for_this_thread);
    int barrier_rc = pthread_barrier_wait(&g_state->barrier);
    

    if(barrier_rc == PTHREAD_BARRIER_SERIAL_THREAD)
        printf("\n--- All threads synchronized. Starting benchmark now! ---\n");
    else if (barrier_rc != 0) {
        fprintf(stderr, "ERROR: Thread %d failed to wait on barrier, code %d\n", ctx->t_id, barrier_rc);
        ret = (void *)-1L;
        goto cleanup;
    }
    
    clock_gettime(CLOCK_MONOTONIC, &ctx->start_time);

    while (ctx->completed_ios < total_ios_for_this_thread) {
         //if anything went wrong, abort
        if (ctx->op_status != 0) {
            fprintf(stderr, "Thread %d: Aborting due to I/O error.\n", ctx->t_id);
            ret = (void *)-1L; // Set the thread's return code to failure
            goto drain_completions;
        }
        
        while ((submitted_ios < total_ios_for_this_thread) &&
               (submitted_ios - ctx->completed_ios < (uint64_t)opts->queue_depth)) {

            uint64_t lba;
            if (strcmp(opts->pattern, "seq") == 0){
                uint64_t current_io_slot = submitted_ios % num_io_slots_in_lane;
                // Calculate the LBA based on this wrapped slot number.
                lba = lba_start_for_this_thread + (current_io_slot * n_sectors_per_io);
            }else if(strcmp(opts->pattern,"rand")== 0){
                uint64_t r = spdk_rand_xorshift64(&ctx->rand_state);
                uint64_t random_offset = r % max_lba_offset_in_lane; 
                lba = lba_start_for_this_thread + random_offset;
            }else    //"pseudo", imitating very fast ssd
                lba = lba_start_for_this_thread;

            int buffer_slot = submitted_ios % opts->queue_depth;
            
            // The buffer for the SPDK command is now a slot in our CPU DMA buffer.

            void* cpu_buffer = (char*)ctx->cpu_dma_buffer + (buffer_slot * opts->io_size);
            
            // The corresponding buffer on the GPU.
            
            CUdeviceptr gpu_buffer = ctx->gpu_dest_buffer + (buffer_slot * opts->io_size);

            // Get a pre-allocated io_ctx for this IO
            io_context_t* io_ctx = &ctx->io_contexts[buffer_slot];
            // IMPORTANT: Remember the CPU buffer address for this specific I/O.
            io_ctx->cpu_buffer_slot = cpu_buffer;

            int rc;
            if (strcmp(opts->op, "read") == 0){
                // For a read, the SPDK part is the first step. We start the timer here.
                
                io_ctx->start_cycle = read_tsc();
                rc = spdk_nvme_ns_cmd_read(g_state->ns, ctx->qpair, cpu_buffer, lba, n_sectors_per_io, read_complete_cb, io_ctx, 0);
            }else{ 
                // This is the 'write' operation
                // For a write, the GPU->CPU copy is the first step. We must do it
                // synchronously *before* submitting to SPDK.
                
                // We start the timer before the ENTIRE pipeline.
                io_ctx->start_cycle = read_tsc();

                // Step 1: GPU -> CPU
                if (cuMemcpyDtoH(cpu_buffer, gpu_buffer, opts->io_size) != CUDA_SUCCESS) {
                    fprintf(stderr, "Thread %d: cuMemcpyDtoH failed before write submission.\n", ctx->t_id);
                    ctx->op_status = -1; // Abort the benchmark
                    break; // Exit the submission loop
                }

                // Step 2: CPU -> NVMe
                rc = spdk_nvme_ns_cmd_write(g_state->ns, ctx->qpair, cpu_buffer, lba, n_sectors_per_io, write_complete_cb, io_ctx, 0);
            }
            if (rc != 0) {
                if (rc == -ENOMEM) {
                    // This is expected if the submission queue is full. Break and poll for completions.
                    break;
                }
                fprintf(stderr, "ERROR: Thread %d failed to submit I/O command, rc=%d\n", ctx->t_id, rc);
                ret = (void *)-1L;
                goto drain_completions; // Don't exit immediately, drain inflight IOs
            }
            submitted_ios++;
        }
        spdk_nvme_qpair_process_completions(ctx->qpair, 0);
    }


drain_completions:
    while (ctx->completed_ios < submitted_ios) {
        spdk_nvme_qpair_process_completions(ctx->qpair, 0);
    }
    clock_gettime(CLOCK_MONOTONIC, &ctx->end_time);
    printf("Thread %d finished benchmark.\n", ctx->t_id);

cleanup:
    if(ctx->qpair) spdk_nvme_ctrlr_free_io_qpair(ctx->qpair);
    if(ctx->io_contexts) free(ctx->io_contexts);
    
    // Free the SPDK-allocated CPU buffer
    if(ctx->cpu_dma_buffer) {
        spdk_dma_free(ctx->cpu_dma_buffer);
    }
    // Free the GPU buffer
    if(ctx->gpu_dest_buffer) {
        cuMemFree(ctx->gpu_dest_buffer);
    }

exit:
    return ret;
}


double calibrate_cpu_frequency(void) {
    struct timespec start_time, end_time;

    // Use CLOCK_MONOTONIC_RAW for a high-precision, steady clock
    // that isn't affected by NTP adjustments.
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &start_time) != 0) {
        perror("clock_gettime failed");
        return 0.0;
    }

    uint64_t start_cycles = read_tsc();

    // Wait for a measurable amount of time. 100ms is a good balance.
    usleep(100000); // 100,000 microseconds = 100ms

    if (clock_gettime(CLOCK_MONOTONIC_RAW, &end_time) != 0) {
        perror("clock_gettime failed");
        return 0.0;
    }
    
    uint64_t end_cycles = read_tsc();

    // Calculate elapsed time in nanoseconds
    uint64_t elapsed_ns = (end_time.tv_sec - start_time.tv_sec) * 1e9;
    elapsed_ns += (end_time.tv_nsec - start_time.tv_nsec);

    // Calculate elapsed cycles
    uint64_t elapsed_cycles = end_cycles - start_cycles;

    // Frequency in GHz = (cycles / nanoseconds)
    // Note: (cycles / ns) is the same as (cycles / (seconds / 1e9)) which is (cycles / seconds) * 1e9, i.e., GHz.
    double freq_ghz = (double)elapsed_cycles / (double)elapsed_ns;

    return freq_ghz;
}




void print_usage(const char *prog_name) {
    printf("Usage: %s --pci <addr> [options]\n", prog_name);
    printf("\nRequired:\n");
    printf("  --pci <addr>           PCIe address of the NVMe device (e.g., 0000:e3:00.0).\n");
    printf("\nOptions:\n");
    printf("  --operation <r|w>      I/O operation type: read or write (default: read).\n");
    printf("  --pattern <s|r|p>      Access pattern: seq, rand or pseudo(imitating very fast ssds) (default: seq).\n");
    printf("  --method <name>        Benchmark method to use. Currently only 'gdr' is supported.\n");
    printf("  --io-size <bytes>      Size of each I/O operation (default: 1MiB).\n");
    printf("  --total-size <bytes>   Total data to transfer (default: 10GiB).\n");
    printf("  --queue-depth <int>    Number of concurrent I/Os (default: 128).\n");
    printf("  --gpu-id <int>         GPU device ID to use (default: 0).\n");
}


int compare_u64(const void *a, const void *b) {
    uint64_t val_a = *(const uint64_t *)a;
    uint64_t val_b = *(const uint64_t *)b;
    if (val_a < val_b) return -1;
    if (val_a > val_b) return 1;
    return 0;
}

int main(int argc, char **argv) {
    benchmark_opts_t opts = {
        .method = "gdr",
        .pci_addr = NULL,
        .gpu_id = 0,
        .io_size = 1024 * 4,      //4KB
        .total_size = 1ULL * 1024 * 1024 * 1024, // 1 GiB
        .queue_depth = 32,
        .op = "read",
        .pattern = "seq",
        .num_threads = 1,
        .display_percentiles = false
    };

    static struct option long_options[] = {
        {"method",      required_argument, 0, 'm'},
        {"pci",         required_argument, 0, 'p'},
        {"io-size",     required_argument, 0, 's'},
        {"total-size",  required_argument, 0, 't'},
        {"queue-depth", required_argument, 0, 'q'},
        {"gpu-id",      required_argument, 0, 'g'},
        {"operation",   required_argument, 0, 'o'},
        {"pattern",     required_argument, 0, 'a'}, // 'a' for access pattern
        {"num-threads", required_argument, 0, 'n'},
        {"percentiles", no_argument,       0, 'P'},
        {"help",        no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    
    while ((opt = getopt_long(argc, argv, "m:p:s:t:q:g:o:a:h", long_options, NULL)) != -1) {
        switch (opt) {
            case 'm': opts.method = optarg; break;
            case 'p': opts.pci_addr = optarg; break;
            case 's': opts.io_size = atol(optarg); break;
            case 't': opts.total_size = atol(optarg); break;
            case 'q': opts.queue_depth = atoi(optarg); break;
            case 'g': opts.gpu_id = atoi(optarg); break;
            case 'o': opts.op = optarg; break;
            case 'a': opts.pattern = optarg; break;
            case 'n': opts.num_threads = atoi(optarg); break;
            case 'P': opts.display_percentiles = true; break; // SET THE FLAG WHEN OPTION IS FOUND
            case 'h': print_usage(argv[0]); return 0;
            default: print_usage(argv[0]); return 1;
        }
    }
    
    if(!opts.pci_addr){
        fprintf(stderr, "Error: --pci <addr> is a required argument.\n");
        print_usage(argv[0]);
        
        return 1;
    }

    if (strcmp(opts.op, "read") != 0 && strcmp(opts.op, "write") != 0) {
        fprintf(stderr, "Error: --operation must be either 'read' or 'write'.\n"); return 1;
    }

    if (strcmp(opts.pattern, "seq") != 0 && strcmp(opts.pattern, "rand") != 0 && strcmp(opts.pattern, "pseudo")!=0) {
        fprintf(stderr, "Error: --pattern must be either 'seq', 'rand' or 'pseudo'.\n"); return 1;
    }


    //global initialization

    global_state_t g_state = {0};
    g_state.opts = &opts;
    
    if (init_global_spdk(&opts, &g_state) != 0) {
        fprintf(stderr, "FATAL: Global SPDK initialization failed.\n");
        // No need to cleanup SPDK if spdk_env_init failed, but good practice
        spdk_env_fini(); 
        return 1;
    }
    
    if (init_global_cuda_gdr(&opts, &g_state) != 0) {
        fprintf(stderr, "FATAL: Global CUDA/GDR initialization failed.\n");
        spdk_nvme_detach(g_state.ctrlr);
        spdk_env_fini();
        return 1;
    }



    int rc = pthread_barrier_init(&g_state.barrier, NULL, opts.num_threads);
    if (rc != 0) {
        fprintf(stderr, "FATAL: Failed to initialize pthread barrier, code %d\n", rc);
        goto cleanup;
        return 1;
    }

    if (pthread_mutex_init(&g_state.mutex, NULL) != 0) {
        fprintf(stderr, "FATAL: Failed to initialize GDR mutex\n");
        // ... perform cleanup and exit ...
        goto cleanup;
        return 1;
    }


    pthread_t *threads = calloc(opts.num_threads, sizeof(pthread_t));
    thread_ctx_t *contexts = calloc(opts.num_threads, sizeof(thread_ctx_t));

    if (threads == NULL || contexts == NULL) {
        perror("Failed to allocate memory for threads/contexts");
        // goto cleanup;
    }
    
    printf("\n--- Starting benchmark with %d worker thread(s) ---\n", opts.num_threads);
    printf("Total Size: %.2f GiB | IO Size: %zu B | Queue Depth: %d\n",
           (double)opts.total_size / (1024*1024*1024), opts.io_size, opts.queue_depth);

    for (int i = 0; i < opts.num_threads; i++) {
        thread_ctx_t *ctx = &contexts[i];
        ctx->t_id = i;
        ctx->gstate = &g_state;
        ctx->op_status = 0; // Initialize status to OK

        int rc = pthread_create(&threads[i], NULL, benchmark_thread_entry, ctx);
        if (rc) {
            fprintf(stderr, "FATAL: Failed to create thread %d, pthread_create() returned %d\n", i, rc);
            // In a real-world app, you might try to cancel already created threads.
            // For this benchmark, we can exit.
            return 1;
        }
    }
    
    //waiting for completion of threads

    bool any_thread_failed = false;

    for (int i = 0; i < opts.num_threads; i++) {
        void *thread_ret;
        int rc = pthread_join(threads[i], &thread_ret);
        if (rc) {
            fprintf(stderr, "Warning: pthread_join() failed for thread %d with code %d\n", i, rc);
            any_thread_failed = true;
        }
        // Check the return value from the thread function itself
        if (thread_ret != NULL) {
            fprintf(stderr, "Warning: Thread %d exited with an error status.\n", i);
            any_thread_failed = true;
        }
    }
    
    if (any_thread_failed) {
        printf("\n--- Benchmark finished with errors. Results may be inaccurate. ---\n");
    } else {
        printf("\n--- All threads completed successfully. ---\n");
    }

    double total_iops = 0;
    double max_time_sec = 0;
    uint64_t total_completed_ios = 0;

    for (int i = 0; i < opts.num_threads; i++) {
        thread_ctx_t *ctx = &contexts[i];
        double elapsed_sec = (ctx->end_time.tv_sec - ctx->start_time.tv_sec) +
                             (ctx->end_time.tv_nsec - ctx->start_time.tv_nsec) / 1e9;
        
        // The total runtime is dictated by the slowest thread
        if (elapsed_sec > max_time_sec) {
            max_time_sec = elapsed_sec;
        }
        
        total_iops += (ctx->completed_ios / elapsed_sec);
        total_completed_ios += ctx->completed_ios;
    }
    
    // Throughput is total data transferred divided by the wall-clock time of the slowest thread.
    double total_bytes_transferred = (double)total_completed_ios * opts.io_size;
    double throughput_mibps = (total_bytes_transferred / (1024 * 1024)) / max_time_sec;

    printf("\n--- Aggregated Results ---\n");
    printf("Completed in:       %.2f seconds\n", max_time_sec);
    printf("Total Throughput:   %.2f MiB/s\n", throughput_mibps);
    printf("Total IOPS:         %.0f ops/sec\n", total_iops);
    // printf("Total successful vtophys hook translations: %llu\n", g_vtophys_hook_successes);


    if (opts.display_percentiles) {
        /****************************************************************
         * DETAILED PERCENTILE ANALYSIS
         ****************************************************************/
        printf("\n--- Detailed Latency Analysis (Percentiles) ---\n");

        uint64_t total_latencies_recorded = 0;
        for (int i = 0; i < opts.num_threads; i++) {
            total_latencies_recorded += contexts[i].latencies_recorded;
        }

        if (total_latencies_recorded == 0) {
            printf("No successful I/Os were recorded to calculate percentiles.\n");
        } else {
            uint64_t *all_latencies = malloc(total_latencies_recorded * sizeof(uint64_t));
            if (all_latencies == NULL) {
                fprintf(stderr, "FATAL: Failed to allocate memory for combined latency array.\n");
                goto cleanup;
            }

            uint64_t current_pos = 0;
            for (int i = 0; i < opts.num_threads; i++) {
                memcpy(all_latencies + current_pos, contexts[i].latencies, contexts[i].latencies_recorded * sizeof(uint64_t));
                current_pos += contexts[i].latencies_recorded;
            }

            printf("Sorting %lu latency entries...\n", total_latencies_recorded);
            qsort(all_latencies, total_latencies_recorded, sizeof(uint64_t), compare_u64);

            const double cpu_freq_ghz = calibrate_cpu_frequency();
            if (cpu_freq_ghz == 0.0) {
                fprintf(stderr, "FATAL: Could not determine CPU frequency.\n");
                free(all_latencies);
                goto cleanup;
            }
            printf("Calibrated CPU frequency: %.2f GHz\n", cpu_freq_ghz);
            const double cycles_per_microsecond = cpu_freq_ghz * 1000.0;
            
            uint64_t p50_index = (uint64_t)(0.50 * (total_latencies_recorded - 1));
            uint64_t p90_index = (uint64_t)(0.90 * (total_latencies_recorded - 1));
            uint64_t p99_index = (uint64_t)(0.99 * (total_latencies_recorded - 1));
            uint64_t p999_index = (uint64_t)(0.999 * (total_latencies_recorded - 1));

            double p50_us = all_latencies[p50_index] / cycles_per_microsecond;
            double p90_us = all_latencies[p90_index] / cycles_per_microsecond;
            double p99_us = all_latencies[p99_index] / cycles_per_microsecond;
            double p999_us = all_latencies[p999_index] / cycles_per_microsecond;
            double min_us = all_latencies[0] / cycles_per_microsecond;
            double max_us = all_latencies[total_latencies_recorded - 1] / cycles_per_microsecond;
            
            uint64_t total_cycles = 0;
            for(uint64_t i = 0; i < total_latencies_recorded; i++) total_cycles += all_latencies[i];
            double avg_us = (total_cycles / (double)total_latencies_recorded) / cycles_per_microsecond;

            printf("\nLatency Statistics (us):\n");
            printf("------------------------\n");
            printf("Average : %10.2f\n", avg_us);
            printf("Min     : %10.2f\n", min_us);
            printf("Median  : %10.2f (50th percentile)\n", p50_us);
            printf("90th    : %10.2f\n", p90_us);
            printf("99th    : %10.2f\n", p99_us);
            printf("99.9th  : %10.2f\n", p999_us);
            printf("Max     : %10.2f\n", max_us);

            free(all_latencies);
        }
    } else {
        /****************************************************************
         * BASIC AVERAGE-ONLY ANALYSIS (DEFAULT)
         ****************************************************************/
        printf("\n--- Latency Analysis ---\n");

        const double cpu_freq_ghz = calibrate_cpu_frequency();
        if(cpu_freq_ghz == 0.0){
            fprintf(stderr, "FATAL: Could not determine CPU frequency.\n");
            goto cleanup;
        }
        printf("Calibrated CPU frequency: %.2f GHz\n", cpu_freq_ghz);
        const double cycles_per_microsecond = cpu_freq_ghz * 1000.0;

        uint64_t final_total_cycles = 0;
        uint64_t final_io_count = 0;

        for(int i = 0; i < opts.num_threads; i++) {
            for(uint64_t j = 0; j < contexts[i].latencies_recorded; j++){
                final_total_cycles += contexts[i].latencies[j];
            }
            final_io_count += contexts[i].latencies_recorded;
        }

        if (final_io_count > 0) {
            double avg_us = (final_total_cycles / (double)final_io_count) / cycles_per_microsecond;
            printf("\nLatency Statistics (us):\n");
            printf("------------------------\n");
            printf("Average : %10.2f\n", avg_us);
            printf("(Use --percentiles for detailed analysis)\n");
        } else {
            printf("\nNo successful I/Os were completed to calculate latency.\n");
        }
    }


cleanup:

    if (g_state.cuda_context) {
        cuCtxDestroy(g_state.cuda_context);
    }

    spdk_env_fini();
    pthread_barrier_destroy(&g_state.barrier);
    pthread_mutex_destroy(&g_state.mutex);
    if(threads) free(threads);
    if (contexts){
        for(int i = 0; i < opts.num_threads; i++)
            if(contexts[i].latencies) free(contexts[i].latencies);
        free(contexts);
    }


    return any_thread_failed ? 1 : 0;
}