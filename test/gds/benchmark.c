#include <stdio.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <getopt.h>
#include <time.h>



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
} benchmark_opts_t;

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
    volatile int operation_status; //0 success, -1 error
    volatile int test_active;
}app_state_t;


/* =================================================================================================
 *                                              SPDK callbacks
 * ================================================================================================= */


static bool probe_cb(void *cb_ctx, const struct spdk_nvme_transport_id *trid, struct spdk_nvme_ctrlr_opts *opts) {
    printf("Probing device at %s...\n", trid->traddr);
    return true;
}

static void attach_cb(void *cb_ctx, const struct spdk_nvme_transport_id *trid, struct spdk_nvme_ctrlr *ctrlr, const struct spdk_nvme_ctrlr_opts *opts) {
    app_state_t *state = (app_state_t *)cb_ctx;
    state->ctrlr = ctrlr;
    state->ns = spdk_nvme_ctrlr_get_ns(ctrlr, 1);
    state->qpair = spdk_nvme_ctrlr_alloc_io_qpair(ctrlr, NULL, 0);
    printf("Attached to %s\n", trid->traddr);
    state->spdk_attached = true;
}

static void write_complete_cb(void *arg, const struct spdk_nvme_cpl *completion) {
    app_state_t *state = (app_state_t *)arg;
    state->completed_ios++;

    if (spdk_nvme_cpl_is_error(completion)) {
        fprintf(stderr, "Write I/O failed!\n");
        state->operation_status = -1;
    }
}

static void read_complete_cb(void *arg, const struct spdk_nvme_cpl *completion) {
    app_state_t *state = (app_state_t *)arg;
    state->completed_ios++;
    
    if (spdk_nvme_cpl_is_error(completion)) {
        fprintf(stderr, "Read I/O failed!\n");
        state->operation_status = -1;
    }
}


/* =================================================================================================
 *                                         Init and cleanup
 * ================================================================================================= */

int init_spdk(benchmark_opts_t *opts, app_state_t *state){
    struct spdk_env_opts spdk_opts;
    spdk_env_opts_init(&spdk_opts);
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
    state->spdk_attached = false;
    if(spdk_nvme_probe(&trid, state, probe_cb, attach_cb, NULL) != 0){
        fprintf(stderr, "Failed to probe for NVMe devices\n");
        return -1;
    }
    
    while(!state->ctrlr) usleep(10000);
    
    return (state->ctrlr && state->ns && state->qpair) ? 0 : -1;
}

int init_cuda_gdr(benchmark_opts_t *opts, app_state_t *state){
    CUdevice dev;
    cuInit(0);
    cuDeviceGet(&dev, opts->gpu_id);
    if(cuCtxCreate(&state->context, NULL, 0, dev) != CUDA_SUCCESS) return -1;

    //alloc enough memory for all in-flight operations
    state->mapped_size = opts->io_size * opts->queue_depth;
    if(cuMemAlloc(&state->d_ptr, state->mapped_size) != CUDA_SUCCESS) return -1; 

    state->gdr_handle = gdr_open();
    if(!state->gdr_handle) return -1;

    if(gdr_pin_buffer(state->gdr_handle, state->d_ptr, state->mapped_size, 0, 0, &state->gdr_mh) == 0)
        state->buf_pinned = true;
    else
        return -1;

    if(gdr_map(state->gdr_handle, state->gdr_mh, &state->bar_ptr, state->mapped_size) != 0)  return -1;
    
    //off for NOW

    // if(spdk_mem_register(state->bar_ptr, state->mapped_size) != 0) {
    //     fprintf(stderr, "Failed to register GDR-mapped buffer with SPDK\n");
    //     return -1;
    // }

    printf("Successfully initialized CUDA and mapped %zu bytes of GPU memory.\n", state->mapped_size);
    return 0;
}

void cleanup(app_state_t *state) {
    printf("\n--- Cleaning up ---\n");
    //cleanup SPDK
    if (state->qpair) spdk_nvme_ctrlr_free_io_qpair(state->qpair);
    if (state->ctrlr) spdk_nvme_detach(state->ctrlr);
    spdk_env_fini();

    //cleanup CUDA/GDRCopy
    if (state->bar_ptr) {
        spdk_mem_unregister(state->bar_ptr, state->mapped_size);
        gdr_unmap(state->gdr_handle, state->gdr_mh, state->bar_ptr, state->mapped_size);
    }
    if (state->buf_pinned) gdr_unpin_buffer(state->gdr_handle, state->gdr_mh);
    if (state->gdr_handle) gdr_close(state->gdr_handle);
    if (state->d_ptr) cuMemFree(state->d_ptr);
    if (state->context) cuCtxDestroy(state->context);
    printf("Cleanup complete.\n");
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

//helper function that fills the gpu with data to write to the nvme
int prepare_gpu_data(benchmark_opts_t *opts, app_state_t *state){
    void *host_buffer = malloc(opts->io_size);
    
    if(!host_buffer){
        perror("prepare_gpu_data(): Couldnt allocate host buffer");
        return -1;
    }

    char *pattern_ptr = (char*)host_buffer;
    for(size_t i = 0; i < opts->io_size; i++) pattern_ptr[i] =(char) (i%256);

    //populating all the slots of the gpu with the same pattern
    for(int i = 0; i < opts->queue_depth; i++){
        CUdeviceptr dest_ptr = state->d_ptr + (i*opts->io_size);
        
        CUresult res = cuMemcpyHtoD(dest_ptr, host_buffer, opts->io_size);
        if(res != CUDA_SUCCESS){
            fprintf(stderr,"cuMemcpyHtoD failed at slot %d\n",i);
            perror("REASON");
            free(host_buffer);
            return -1;
        }
    }

    free(host_buffer);
    cuCtxSynchronize();

    printf("GPU data ready\n");
    return 0;
}

void run_gdr_benchmark(benchmark_opts_t *opts, app_state_t *state) {
    printf("\n--- Running Benchmark [Method: gdr] ---\n");

    printf("Operation type: %s | Pattern: %s | IO Size: %zu B | Queue Depth: %d | Total Size: %.2f GiB\n",
           opts->op, opts->pattern, opts->io_size, opts->queue_depth, (double)opts->total_size / (1024*1024*1024));

    uint64_t total_ios_to_submit = opts->total_size / opts->io_size;
    uint64_t submitted_ios = 0;
    state->completed_ios = 0;
    state->test_active = true;


    uint64_t ns_size_sectors = spdk_nvme_ns_get_num_sectors(state->ns);
    uint32_t sector_size = spdk_nvme_ns_get_sector_size(state->ns);

    if (opts->io_size % sector_size != 0) {
        fprintf(stderr, "Error: IO size (%zu) must be a multiple of sector size (%u).\n", opts->io_size, sector_size);
        return;
    }
    uint32_t n_sectors_per_io = opts->io_size / sector_size;

    uint64_t max_lba = ns_size_sectors - n_sectors_per_io;
    
    srand(time(NULL));

    if (strcmp(opts->op, "write") == 0) {
        if (prepare_gpu_data(opts,state) != 0) {
            fprintf(stderr, "Failed to prepare GPU data for writing.\n");
            return;
        }
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    while (state->test_active) {
        //Submit new I/Os until the queue is full or we've submitted all work
        
        while ((submitted_ios < total_ios_to_submit) &&
               (submitted_ios - state->completed_ios < (uint64_t)opts->queue_depth)) {
            
            uint64_t lba;

            //check if pattern is random or sequential

            if(strcmp(opts->pattern,"rand")==0)
                lba = rand64() % (max_lba+1);
            else if(strcmp(opts->pattern, "seq")==0)
                lba = (submitted_ios * n_sectors_per_io) % ns_size_sectors;
            else    
                //pseudo, simulates very fast ssd
                lba = 0;
            
            int buffer_slot = submitted_ios % opts->queue_depth;
            
            //destination buffer changes for every IO
            void* buffer = (char*)state->bar_ptr + (buffer_slot * opts->io_size);
            int rc = 0;
            if(strcmp(opts->op,"read") == 0)
                rc = spdk_nvme_ns_cmd_read(state->ns, state->qpair, buffer, lba, n_sectors_per_io, read_complete_cb, state, 0);
            else
                rc = spdk_nvme_ns_cmd_write(state->ns, state->qpair, buffer, lba, n_sectors_per_io, write_complete_cb, state, 0);
            
            if (rc != 0) {
                fprintf(stderr, "Failed to submit read command (rc=%d). Aborting.\n", rc);
                state->test_active = false;
                break;
            }
            submitted_ios++;
        }

        //poll for completions
        spdk_nvme_qpair_process_completions(state->qpair, 0);

        //check if it is finished
        if (state->completed_ios >= total_ios_to_submit) {
            state->test_active = false;
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);

    //drain any remaining in-flight completions
    while (state->completed_ios < submitted_ios)
        spdk_nvme_qpair_process_completions(state->qpair, 0);
    
    // --- Calculate and print results ---
    double elapsed_sec = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double total_gb = (double)opts->total_size / (1024 * 1024 * 1024);
    double throughput_mibps = ((double)opts->total_size / (1024 * 1024)) / elapsed_sec;
    double iops = state->completed_ios / elapsed_sec;

    printf("\n--- Results ---\n");
    printf("Completed in: %.2f seconds.\n", elapsed_sec);
    printf("Throughput:   %.2f MiB/s\n", throughput_mibps);
    printf("IOPS:         %.0f ops/sec\n", iops);
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

int main(int argc, char **argv) {
    benchmark_opts_t opts = {
        .method = "gdr",
        .pci_addr = NULL,
        .gpu_id = 0,
        .io_size = 1024 * 4,      //4KB
        .total_size = 1ULL * 1024 * 1024 * 1024, // 1 GiB
        .queue_depth = 32,
        .op = "read",
        .pattern = "seq"
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
            case 'h': print_usage(argv[0]); return 0;
            default: print_usage(argv[0]); return 1;
        }
    }
    
    if (!opts.pci_addr) {
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


    app_state_t state = {0};



    if (strcmp(opts.method, "gdr") == 0) {
        if (init_cuda_gdr(&opts, &state) != 0 || init_spdk(&opts, &state) != 0) {
            fprintf(stderr, "Initialization failed.\n");
            cleanup(&state);
            return 1;
        }
        run_gdr_benchmark(&opts, &state);
    } else {
        fprintf(stderr, "Error: Method '%s' is not yet supported.\n", opts.method);
        return 1;
    }

    cleanup(&state);
    return 0;
}
