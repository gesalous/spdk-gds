APP ?=

MAKEFILE_THIS := $(firstword $(MAKEFILE_LIST))

APPS := nvme2gpu nvme2gpu_baseline 

SPDK_APP_nvme2gpu_SRCS := benchmark.c gdr_gpumap_helper.c
SPDK_APP_nvme2gpu_baseline_SRCS := baseline.c gdr_gpumap_helper.c


SPDK_ROOT_DIR := $(abspath $(CURDIR)/../..)
include $(SPDK_ROOT_DIR)/mk/spdk.common.mk
include $(SPDK_ROOT_DIR)/mk/spdk.modules.mk

CUDA_HOME ?= /usr/local/cuda-13.0
GDRCOPY_HOME ?= /usr/local
CFLAGS += -I$(CUDA_HOME)/include -I$(GDRCOPY_HOME)/include
CFLAGS += -I$(SPDK_ROOT_DIR)/dpdk/build/include
LDFLAGS += -L$(CUDA_HOME)/lib64 -L$(GDRCOPY_HOME)/lib -Wl,-rpath,$(GDRCOPY_HOME)/lib
SYS_LIBS += -lcuda -lcudart -lgdrapi -lpthread

ifeq ($(OS),Linux)
SYS_LIBS += -laio
CFLAGS += -DHAVE_LIBAIO
endif

ifneq ($(APP),)

C_SRCS := $(SPDK_APP_$(APP)_SRCS)
ifeq ($(strip $(C_SRCS)),)
$(error Unknown APP '$(APP)'. Choose from $(APPS))
endif

SPDK_LIB_LIST += $(SOCK_MODULES_LIST) nvme vmd keyring_file env_dpdk

include $(SPDK_ROOT_DIR)/mk/spdk.app.mk

else # top-level driver ---------------------------------------------------

.PHONY: all $(APPS) clean print-cflags

all: $(APPS)

$(APPS):
	$(MAKE) -f $(MAKEFILE_THIS) APP=$@

clean::
	@for app in $(APPS); do \
		$(MAKE) -f $(MAKEFILE_THIS) APP=$$app clean; \
	done

print-cflags:
	@echo $(CFLAGS)

endif
