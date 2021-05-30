CXX := nvcc
DEBUG := -O3
export DEBUG

OBJS := $(patsubst %.cpp,%.cpp.o,$(wildcard src/*.cpp))
OBJS += $(patsubst %.cu,%.cu.o,$(wildcard src/*.cu))

PROCESSOR_ARCHITECTURE := $(shell uname -i)
ifneq (,$(findstring arm,$(PROCESSOR_ARCHITECTURE)))
	TUNE_FOR_ARCH :=-m32 --target-cpu-architecture ARM --compiler-options=''
else
	TUNE_FOR_ARCH :=-march=native
endif

QUIET := 
export QUIET

CXXFLAGS :=-I/usr/local/cuda-6.5/include -std=c++11 -Xcompiler "$(DEBUG)" $(TUNE_FOR_ARCH) -gencode arch=compute_32,code=sm_32 $(DEVICE_DEBUG) $(QUIET)
HOST_LDFLAGS :=-L/src -lpthread
LDFLAGS :=-lcudart -L/usr/local/cuda-6.5/lib

.PHONY: clean
.DEFAULT: all

all: $(OBJS)
	$(CXX) $(OBJS) $(LDFLAGS) -Xlinker "$(HOST_LDFLAGS)" -o softmax

clean-objectfiles:
	rm -r $(OBJS)

clean: clean-objectfiles
	rm softmax

%.cpp.o: %.cpp
	$(CXX) $(CFLAGS) -c $< -o $@ $(CXXFLAGS)

%.cu.o: %.cu
	$(CXX) $(CFLAGS) -c $< -o $@ $(CXXFLAGS)
