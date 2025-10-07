#include <stdio.h>
#include <cuda.h>
#include <gdrapi.h>

int main(void) {
	CUdevice dev;
	CUcontext ctx;

	CUdeviceptr d_ptr;

	void *bar_ptr = NULL;
	gdr_t g = NULL;
	gdr_mh_t mh;
	int ret;

	cuInit(0);
	cuDeviceGet(&dev, 0);
	cuCtxCreate(&ctx, 0, dev);

	cuMemAlloc(&d_ptr, 4096);

	g = gdr_open();
	if (!g) {
		fprintf(stderr, "gdr_open failed\n");
		return 1;
	}

	ret = gdr_pin_buffer(g, d_ptr, 4096, 0, 0, &mh);
	if (ret) {
		fprintf(stderr, "gdr_pin_buffer failed\n");
		return 1;
	}

	ret = gdr_map(g, mh, &bar_ptr, 4096);
	if (ret) {
		fprintf(stderr, "gdr_map failed\n");
		return 1;
	}

	fprintf(stdout, "bar_ptr: %p \n", bar_ptr);

	char *cpu_ptr = (char *)bar_ptr;

	for (int i=0; i < 4096; i++) {
		cpu_ptr[i] = (char)(i % 256);
	}

	char *host_buf = (char *)malloc(4096);
	if (!host_buf) {
		fprintf(stderr, "malloc failed\n");
		return 1;
	}

	CUresult res = cuMemcpyDtoH(host_buf, d_ptr, 4096);
	//if (res != CUDA_SUCESS) {
	//	fprintf(stderr, "cuMemcpyDtoH failed\n");
	//	return 1;
	//}

	for (int i=0; i<4096; i++) {
		if (host_buf[i] != (char)(i % 256)) {
			fprintf(stderr, "verification failed\n");
			return 1;
		}
		//fprintf(stdout, "%d ", (unsigned char)host_buf[i]);
	}


	fprintf(stdout, "verification succesfull\n");

	for (int i=0; i<4096; i++) {
		host_buf[i] = 66;
	}

	res = cuMemcpyHtoD(d_ptr, host_buf, 4096);

	sleep(1);

	for (int i=0; i < 4096; i++) {
		unsigned char c = (unsigned char)cpu_ptr[i];
		if (c != 66) {
			fprintf(stderr, "verification failed\n");
			return 1;
		}
	}

	fprintf(stdout, "verification successfull\n");

	return 0;

}

