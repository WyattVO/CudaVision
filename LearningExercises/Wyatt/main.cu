%%writefile main.cu
// ^ Colab "magic" (must be alone on line 1): save this cell as main.cu instead of running it as Python.

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
}

int main() {  // Program starts here, running on the CPU

    cv::where_am_I("CPU");  // Ordinary CPU code: prints once

    // Create a vector holding ONE element, the number 3.
    // (Braces {3} = one element with value 3. Parentheses (3) would mean three elements.)
    thrust::universal_vector<int> vec{3};

    // Run the lambda once for each element in vec, ON THE GPU (thrust::device).
    thrust::for_each(thrust::device,   // Execution policy: run on the GPU
                     vec.begin(),      // Start of the range
                     vec.end(),        // End of the range
        // The lambda: an inline, unnamed function.
        // []                  = captures nothing from outside
        // __host__ __device__ = can be compiled for CPU and GPU
        // (int)               = takes one element; it's unnamed because we don't use it
        [] __host__ __device__(int) { cv::where_am_I("GPU"); });

    // Wait for the GPU to finish, so its printf output appears now and in order
    cudaDeviceSynchronize();

    // Same thing, but thrust::host means it runs as a normal loop on the CPU
    thrust::for_each(thrust::host, vec.begin(), vec.end(),
        [] __host__ __device__(int) { cv::where_am_I("CPU"); });

    cv::where_am_I("CPU");  // Back to ordinary CPU code: prints once

    return 0;  // Tell the operating system the program finished successfully
}