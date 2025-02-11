NVCC := nvcc
CFLAGS := -O2 --std=c++11
EXTRA_NVCCFLAGS := --cudart=shared

all: conv2dV1 conv2dV2

conv2dV1: conv2dV1.cu
	$(NVCC) $(CFLAGS) $(EXTRA_NVCCFLAGS) -o conv2dV1 conv2dV1.cu

conv2dV2: conv2dV2.cu
	$(NVCC) $(CFLAGS) $(EXTRA_NVCCFLAGS) -o conv2dV2 conv2dV2.cu

clean:
	rm -f conv2dV1 conv2dV2
