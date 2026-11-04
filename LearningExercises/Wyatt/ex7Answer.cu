%%writefile ex7file.cu
// Exercise 7: ANSWER (divide while writing, calculated keys, no extra pass)
#include <cstdio>
#include <cmath>
#include <chrono>

#include <thrust/universal_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/discard_iterator.h>

// ===== EXERCISE 7 START =====
struct mean_functor {
    int width;
    __host__ __device__ float operator()(float x) const { return x / width; }
};

thrust::universal_vector<float> row_averages(int height, int width,
    const thrust::universal_vector<int>& row_ids,
    const thrust::universal_vector<float>& temp)
{
    thrust::universal_vector<float> means(height);
    // ANSWER TODO 1: each sum is divided by width as it is written
    auto means_output = thrust::make_transform_output_iterator(
        means.begin(), mean_functor{width});
    // ANSWER: same calculated keys as Exercise 6
    auto row_ids_begin = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0),
        [=] __host__ __device__ (int i) { return i / width; });
    auto row_ids_end = row_ids_begin + temp.size();
    thrust::reduce_by_key(thrust::device, row_ids_begin, row_ids_end,
        temp.begin(), thrust::make_discard_iterator(), means_output);
    // ANSWER TODO 2: the extra thrust::transform pass is deleted
    return means;
}
// ===== EXERCISE 7 END =====

thrust::universal_vector<float> make_grid(int height, int width)
{
    thrust::universal_vector<float> temp(static_cast<size_t>(height) * width, 15.0f);
    thrust::fill(thrust::device, temp.begin(), temp.begin() + width, 90.0f);
    return temp;
}

thrust::universal_vector<int> make_row_ids(int height, int width)
{
    thrust::universal_vector<int> row_ids(static_cast<size_t>(height) * width);
    thrust::tabulate(thrust::device, row_ids.begin(), row_ids.end(),
        [=] __host__ __device__ (int i) { return i / width; });
    return row_ids;
}

double peak_bandwidth_gbs()
{
    int device = 0, clock_khz = 0, bus_bits = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrMemoryClockRate, device);
    cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, device);
    return 2.0 * clock_khz * 1e3 * (bus_bits / 8.0) / 1e9;
}

int main()
{
    const int height = 16;
    const int width = 16777216;
    const int runs = 5;

    thrust::universal_vector<float> temp = make_grid(height, width);
    thrust::universal_vector<int> row_ids = make_row_ids(height, width);

    thrust::universal_vector<float> means = row_averages(height, width, row_ids, temp);
    cudaDeviceSynchronize();

    auto begin = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < runs; r++) {
        means = row_averages(height, width, row_ids, temp);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    const double seconds = std::chrono::duration<double>(end - begin).count() / runs;
    const double gigabytes = temp.size() * sizeof(float) / 1e9;
    const double throughput = gigabytes / seconds;
    const double peak = peak_bandwidth_gbs();

    const float r0 = means[0], r1 = means[1], r2 = means[2];
    std::printf("row 0 = %g\nrow 1 = %g\nrow 2 = %g\n", r0, r1, r2);
    const bool correct = std::fabs(r0 - 90.0) < 0.01 && std::fabs(r1 - 15.0) < 0.01;
    std::printf("averages are %s\n\n", correct ? "correct" : "WRONG");

    std::printf("time per run:         %.2f ms\n", seconds * 1e3);
    std::printf("achieved throughput:  %.1f GB/s\n", throughput);
    if (peak > 0) {
        std::printf("your GPU's limit:     %.1f GB/s\n", peak);
        std::printf("percent of limit:     %.1f%%\n", 100.0 * throughput / peak);
    }
}