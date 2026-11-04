%%writefile ex6file.cu
// Exercise 6: ANSWER (row keys are calculated on the fly, nothing stored)
#include <cstdio>
#include <cmath>
#include <chrono>

#include <thrust/universal_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/tabulate.h>
#include <thrust/reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/discard_iterator.h>

// ===== EXERCISE 6 START =====
thrust::universal_vector<float> row_temperatures(int height, int width,
    const thrust::universal_vector<int>& row_ids,
    const thrust::universal_vector<float>& temp)
{
    // ANSWER: count 0,1,2,... and turn each cell number i into its row (i / width)
    auto row_ids_begin = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0),
        [=] __host__ __device__ (int i) { return i / width; });
    auto row_ids_end = row_ids_begin + temp.size();
    thrust::universal_vector<float> sums(height);
    thrust::reduce_by_key(thrust::device,
        row_ids_begin, row_ids_end,       // keys: row of each cell
        temp.begin(),                     // values: temperatures
        thrust::make_discard_iterator(),  // keys out: not needed
        sums.begin());                    // one sum per row
    return sums;
}
// ===== EXERCISE 6 END =====

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

    thrust::universal_vector<float> sums = row_temperatures(height, width, row_ids, temp);
    cudaDeviceSynchronize();

    auto begin = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < runs; r++) {
        sums = row_temperatures(height, width, row_ids, temp);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    const double seconds = std::chrono::duration<double>(end - begin).count() / runs;
    const double gigabytes = temp.size() * sizeof(float) / 1e9;
    const double throughput = gigabytes / seconds;
    const double peak = peak_bandwidth_gbs();

    const float r0 = sums[0], r1 = sums[1], r2 = sums[2];
    std::printf("row 0 = %g\nrow 1 = %g\nrow 2 = %g\n", r0, r1, r2);
    const bool correct = std::fabs(r0 - 90.0 * width) < 1e-3 * 90.0 * width
                      && std::fabs(r1 - 15.0 * width) < 1e-3 * 15.0 * width;
    std::printf("sums are %s\n\n", correct ? "correct" : "WRONG");

    std::printf("time per run:         %.2f ms\n", seconds * 1e3);
    std::printf("achieved throughput:  %.1f GB/s\n", throughput);
    if (peak > 0) {
        std::printf("your GPU's limit:     %.1f GB/s\n", peak);
        std::printf("percent of limit:     %.1f%%\n", 100.0 * throughput / peak);
    }
}