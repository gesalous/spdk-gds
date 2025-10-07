SPDK_ROOT_DIR := $(abspath $(CURDIR)/../..)
include $(SPDK_ROOT_DIR)/mk/spdk.common.mk
include $(SPDK_ROOT_DIR)/mk/spdk.modules.mk

APP = gds_test
C_SRCS := gds_test.c


SPDK_LIB_LIST += $(SOCK_MODULES_LIST) nvme vmd keyring_file

CUDA_HOME ?= /usr/local/cuda-11.7
GDRCOPY_HOME ?= /usr/local
CFLAGS += -I$(CUDA_HOME)/include -I$(GDRCOPY_HOME)/include
LDFLAGS += -L$(CUDA_HOME)/lib64 -L$(GDRCOPY_HOME)/lib -Wl,-rpath,$(GDRCOPY_HOME)/lib 
SYS_LIBS += -lcuda -lcudart -lgdrapi

ifeq ($(OS),Linux)
SYS_LIBS += -laio
CFLAGS += -DHAVE_LIBAIO
endif

include $(SPDK_ROOT_DIR)/mk/spdk.app.mk
