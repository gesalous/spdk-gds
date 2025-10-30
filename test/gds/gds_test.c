
#include <stdio.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>



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
#define CUDA_MEM_SIZE (4*1024UL)


// static void vtophys(unsigned long vaddr)
// {
// 	int fd = open("/dev/vtophys", O_RDONLY);
// 	if (fd < 0) { perror("open"); return 1; }

// 	unsigned long phys = (unsigned long)&vaddr;  /* any user VA */

// 	if (ioctl(fd, VTIOCTL_GET_PHYS, &phys) == -1) { perror("ioctl"); return 1; }

// 	printf("VA 0x%lx -> PA 0x%lx\n", vaddr, phys);
// 	close(fd);
// 	return 0;
// }


struct nvme_context {
	struct spdk_nvme_ctrlr *ctrlr;
	struct spdk_nvme_ns *ns;
	struct spdk_nvme_qpair *qpair;
	bool read_complete;
	bool write_complete;
	int read_status;
	int write_status;
};

static void
read_complete_cb(void *arg, const struct spdk_nvme_cpl *completion)
{
	struct nvme_context *ctx = arg;

	ctx->read_complete = true;
	ctx->read_status = spdk_nvme_cpl_is_error(completion) ? -1 : 0;

	if (ctx->read_status == 0) {
		printf("NVMe read completed successfully\n");
	} else {
		printf("NVMe read failed with status: 0x%x\n", completion->status.sc);
	}
}

static void
write_complete_cb(void *arg, const struct spdk_nvme_cpl *completion)
{
	struct nvme_context *ctx = arg;

	ctx->write_complete = true;
	ctx->write_status = spdk_nvme_cpl_is_error(completion) ? -1 : 0;

	if (ctx->write_status == 0) {
		printf("NVMe write completed successfully\n");
	} else {
		printf("NVMe write failed with status: 0x%x\n", completion->status.sc);
	}
}

static bool
probe_cb(void *cb_ctx, const struct spdk_nvme_transport_id *trid,
	 struct spdk_nvme_ctrlr_opts *opts)
{
	printf("Attaching to %s\n", trid->traddr);
	return true;
}

static void
attach_cb(void *cb_ctx, const struct spdk_nvme_transport_id *trid,
	  struct spdk_nvme_ctrlr *ctrlr, const struct spdk_nvme_ctrlr_opts *opts)
{
	struct nvme_context *ctx = cb_ctx;
	struct spdk_nvme_ns *ns;

	printf("Attached to %s\n", trid->traddr);

	ctx->ctrlr = ctrlr;

	// Get the first active namespace
	ns = spdk_nvme_ctrlr_get_ns(ctrlr, 1);
	if (ns == NULL || !spdk_nvme_ns_is_active(ns)) {
		printf("No active namespace found\n");
		return;
	}

	ctx->ns = ns;

	// Allocate I/O queue pair
	ctx->qpair = spdk_nvme_ctrlr_alloc_io_qpair(ctrlr, NULL, 0);
	if (ctx->qpair == NULL) {
		printf("Failed to allocate I/O queue pair\n");
		return;
	}

	printf("Namespace ID: %d, Size: %lu blocks, Block Size: %u bytes\n",
	       spdk_nvme_ns_get_id(ns),
	       spdk_nvme_ns_get_num_sectors(ns),
	       spdk_nvme_ns_get_sector_size(ns));
}

int main(int argc, char **argv)
{
	struct spdk_env_opts opts;
	struct spdk_nvme_transport_id trid = {};
	struct nvme_context ctx = {};
	void *read_buffer, *write_buffer;
	size_t buffer_size;
	int rc;

	if (argc != 2) {
		printf("Usage: %s <trtype:PCIe traddr:0000:xx:xx.x>\n", argv[0]);
		printf("Example: %s trtype:PCIe traddr:0000:e3:00.0\n", argv[0]);
		return 1;
	}

	// Parse transport ID from command line
	if (spdk_nvme_transport_id_parse(&trid, argv[1]) != 0) {
		printf("Failed to parse transport ID: %s\n", argv[1]);
		return 1;
	}


	CUdevice dev;
	CUcontext context;

	CUdeviceptr d_ptr;
	CUdeviceptr d_ptr2;

	void *bar_ptr = NULL;
	void *bar_ptr2 = NULL;
	gdr_t g = NULL;
	gdr_mh_t mh;
	int ret;

	cuInit(0);
	cuDeviceGet(&dev, 0);
	cuCtxCreate(&context, 0, dev);

	cuMemAlloc(&d_ptr, CUDA_MEM_SIZE);


	g = gdr_open();
	if (!g) {
		fprintf(stderr, "gdr_open failed reason follows:\n");
		perror("Reason");
		return 1;
	}

	ret = gdr_pin_buffer(g, d_ptr, CUDA_MEM_SIZE, 0, 0, &mh);
	if (ret) {
		fprintf(stderr, "gdr_pin_buffer failed\n");
		return 1;
	}

	ret = gdr_map(g, mh, &bar_ptr, CUDA_MEM_SIZE);
	if (ret) {
		fprintf(stderr, "gdr_map failed\n");
		return 1;
	}

	fprintf(stderr, "bar_ptr: %lu \n", bar_ptr);
  /*<gesalous>*/
	// cuMemAlloc(&d_ptr2, CUDA_MEM_SIZE);
 //  	ret = gdr_pin_buffer(g, d_ptr2, CUDA_MEM_SIZE, 0, 0, &mh);
	// if (ret) {
	// 	fprintf(stderr, "gdr_pin_buffer for d_ptr2 failed\n");
	// 	return 1;
	// }

	// ret = gdr_map(g, mh, &bar_ptr2, CUDA_MEM_SIZE);
	// if (ret) {
	// 	fprintf(stderr, "gdr_map (2) failed\n");
	// 	return 1;
	// }

	// fprintf(stderr, "bar_ptr2: %lu \n", bar_ptr2);

  /*</gesalous>*/

	char *cpu_ptr = (char *)bar_ptr;

	for (int i = 0; i < CUDA_MEM_SIZE; i++) {
		cpu_ptr[i] = (char)(i % 256);
	}

	char *host_buf;
	posix_memalign(&host_buf, 4096, CUDA_MEM_SIZE);
	if (!host_buf) {
		fprintf(stderr, "posix_memalign failed\n");
		return 1;
	}

	CUresult res = cuMemcpyDtoH(host_buf, d_ptr, CUDA_MEM_SIZE);
	//if (res != CUDA_SUCESS) {
	//	fprintf(stderr, "cuMemcpyDtoH failed\n");
	//	return 1;
	//}

	for (int i = 0; i < 4096; i++) {
		if (host_buf[i] != (char)(i % 256)) {
			fprintf(stderr, "verification failed\n");
			return 1;
		}
		//fprintf(stdout, "%d ", (unsigned char)host_buf[i]);
	}

	// fprintf(stderr, "bar_ptr: %p (decimal: %lu, hex: 0x%lx) pid: %d\n", bar_ptr, (unsigned long)bar_ptr,
	// 	(unsigned long)bar_ptr, getpid());
	// vtophys(bar_ptr);

	for (int i = 0; i < 4096; i++) {
		host_buf[i] = 66;
	}

	res = cuMemcpyHtoD(d_ptr, host_buf, 4096);

	sleep(1);

	for (int i = 0; i < 4096; i++) {
		unsigned char c = (unsigned char)cpu_ptr[i];
		if (c != 66) {
			fprintf(stderr, "verification failed\n");
			return 1;
		}
	}

	fprintf(stdout, "verification successfull starting the NVMe test with bar_ptr\n");

	// Initialize SPDK environment
	spdk_env_opts_init(&opts);
	opts.name = "nvme_rw_example";

	if (spdk_env_init(&opts) < 0) {
		printf("Failed to initialize SPDK environment\n");
		return 1;
	}

	printf("Initializing NVMe driver...\n");

	// Probe and attach to NVMe device
	if (spdk_nvme_probe(&trid, &ctx, probe_cb, attach_cb, NULL) != 0) {
		printf("Failed to probe NVMe device\n");
		spdk_env_fini();
		return 1;
	}

	if (ctx.ctrlr == NULL || ctx.ns == NULL || ctx.qpair == NULL) {
		printf("Failed to initialize NVMe device\n");
		spdk_env_fini();
		return 1;
	}

	// Allocate buffers (1 block size)
	buffer_size = spdk_nvme_ns_get_sector_size(ctx.ns);
	// read_buffer = spdk_zmalloc(buffer_size, 0x1000, NULL, SPDK_ENV_SOCKET_ID_ANY, SPDK_MALLOC_DMA);
  // if (spdk_mem_register(bar_ptr, CUDA_MEM_SIZE) != 0) {
  //   printf("Failed to register bar_ptr with SPDK\n");
  //   spdk_env_fini();
  //   return 1;
  // }
	read_buffer = bar_ptr;
  fprintf(stderr,"Set as destination buffer the GPU buffer (virtual address vtophys module will do the actual translation)\n");
	write_buffer = spdk_zmalloc(buffer_size, 0x1000, NULL, SPDK_ENV_SOCKET_ID_ANY, SPDK_MALLOC_DMA);

	if (read_buffer == NULL || write_buffer == NULL) {
		printf("Failed to allocate DMA buffers\n");
		goto cleanup;
	}

	// Fill write buffer with test data
	memset(write_buffer, 0xAA, buffer_size);

	printf("Starting NVMe operations...\n");

	// Issue read command (LBA 0)
	printf("Issuing read command...\n");
	ctx.read_complete = false;
	rc = spdk_nvme_ns_cmd_read(ctx.ns, ctx.qpair, read_buffer, 0, 1, read_complete_cb, &ctx, 0);
	if (rc != 0) {
		printf("Failed to submit read command\n");
		goto cleanup;
	}

	// Wait for read completion
	while (!ctx.read_complete) {
		spdk_nvme_qpair_process_completions(ctx.qpair, 0);
		usleep(1000);
	}

	if (ctx.read_status != 0) {
		printf("Read operation failed\n");
		goto cleanup;
	}

	// Issue write command (LBA 0)
	printf("Issuing write command...\n");
	ctx.write_complete = false;
	rc = spdk_nvme_ns_cmd_write(ctx.ns, ctx.qpair, write_buffer, 0, 1, write_complete_cb, &ctx, 0);
	if (rc != 0) {
		printf("Failed to submit write command\n");
		goto cleanup;
	}

	// Wait for write completion
	while (!ctx.write_complete) {
		spdk_nvme_qpair_process_completions(ctx.qpair, 0);
		usleep(1000);
	}

	if (ctx.write_status != 0) {
		printf("Write operation failed\n");
		goto cleanup;
	}

	printf("All operations completed successfully!\n");

cleanup:
	// if (read_buffer) {
	// 	spdk_free(read_buffer);
	// }
	if (write_buffer) {
		spdk_free(write_buffer);
	}
	if (ctx.qpair) {
		spdk_nvme_ctrlr_free_io_qpair(ctx.qpair);
	}
	if (ctx.ctrlr) {
		spdk_nvme_detach(ctx.ctrlr);
	}

	spdk_env_fini();
	return (ctx.read_status == 0 && ctx.write_status == 0) ? 0 : 1;
}

