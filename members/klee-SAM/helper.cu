#include <cstdio>                    // Gives us printf, which works on both the CPU and the GPU
#include <thrust/universal_vector.h> // Thrust vector using memory that both the CPU and GPU can access
#include <thrust/for_each.h>         // Thrust's for_each algorithm: "run a function on every element"
#include <thrust/execution_policy.h> // thrust::host and thrust::device, which choose CPU or GPU

namespace cv {  // A namespace groups names together; we call this function as cv::where_am_I

    // __host__   = also compile a CPU version of this function
    // __device__ = also compile a GPU version of this function
    // 'expected' is the label we pass in ("CPU" or "GPU") saying where we THINK it runs
    __host__ __device__ void where_am_I(const char* expected) {

    // nvcc compiles this file twice: once for the CPU and once for the GPU.
    // __CUDA_ARCH__ only exists during the GPU compilation, so this check
    // decides which printf goes into each version. It happens at compile time.
    #ifdef __CUDA_ARCH__
        // Only included in the GPU version of this function
        printf("Expected %s -> actually running on GPU\n", expected);
    #else
        // Only included in the CPU version of this function
        printf("Expected %s -> actually running on CPU\n", expected);
    #endif  // Ends the #ifdef / #else block
    }

    __host__ __device__ const char* execution_space() {
        #ifdef __CUDA_ARCH__
        // Only included in the GPU version of this function
        return "Device";
        #else
        // Only included in the CPU version of this function
        return "Host";
        #endif  // Ends the #ifdef / #else block
    }
}


/**
 * g++ compiling for CPU testing:
 * g++ <filename> -O2 -g -Wall -Wextra -o build/<outputname>
 * g++ exercise4CPU.cpp -O2 -g -Wall -Wextra -o build/e4
 */