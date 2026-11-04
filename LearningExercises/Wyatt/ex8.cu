%%writefile ex8file.cu
// Exercise 8: overlap the GPU's compute with the CPU's file writes
//
// Each round: copy the grid GPU -> CPU, run 500 heat steps on the GPU,
// then write the CPU copy to disk. Right now the CPU and GPU take turns.
// Run it once as is and write down "compute + write" and "write",
// then do the TODOs and run it again.
//
// Done when: "compute + write" is close to the slower of the two jobs,
// not the two added together.

#include <cstdio>
#include <chrono>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/tabulate.h>
#include <thrust/iterator/counting_iterator.h>
#include <cub/cub.cuh>

// ===== EXERCISE 8 START =====
void simulate(int width, int height,
              const thrust::device_vector<float>& in,
              thrust::device_vector<float>& out)
{
    const float* temp = thrust::raw_pointer_cast(in.data());

    // One heat step for cell id: average a little with its 4 neighbors.
    auto compute = [=] __host__ __device__ (int id) -> float {
        int row = id / width;
        int col = id % width;
        if (row == 0 || row == height - 1 || col == 0 || col == width - 1)
            return temp[id];                                  // edges stay fixed
        float d2tdx2 = temp[id - 1]     - 2.0f * temp[id] + temp[id + 1];
        float d2tdy2 = temp[id - width] - 2.0f * temp[id] + temp[id + width];
        return temp[id] + 0.2f * (d2tdx2 + d2tdy2);
    };

    // TODO 1: Thrust waits here until the GPU finishes. Replace this with
    //         cub::DeviceTransform::Transform(in, out, n, op), which returns right away.
    //   in:  your Exercise 6 answer, thrust::make_counting_iterator(0)
    //   out: out.begin()
    //   n:   width * height
    //   op:  compute
    thrust::tabulate(thrust::device, out.begin(), out.end(), compute);   //change THIS!!!
}
// ===== EXERCISE 8 END =====


// ---------- the TODO 2 line goes in main(), below ----------

// The heat grid: the top row is 90 degrees, everything else is 15.
thrust::device_vector<float> make_grid(int height, int width)
{
    thrust::device_vector<float> temp(static_cast<size_t>(height) * width, 15.0f);
    thrust::fill(thrust::device, temp.begin(), temp.begin() + width, 90.0f);
    return temp;
}

// Write the CPU copy of the grid to a file (this is the slow CPU job).
void store(int step, int height, int width, const thrust::host_vector<float>& h)
{
    char name[64];
    std::snprintf(name, sizeof(name), "heat_%d.bin", step);
    FILE* f = std::fopen(name, "wb");
    std::fwrite(h.data(), sizeof(float), static_cast<size_t>(height) * width, f);
    std::fclose(f);
}

int main()
{
    const int height = 2048;
    const int width  = 8192;
    const int compute_steps = 500;
    const int write_steps   = 3;

    thrust::device_vector<float> d_prev = make_grid(height, width);
    thrust::device_vector<float> d_next(static_cast<size_t>(height) * width);
    thrust::host_vector<float>   h_prev(static_cast<size_t>(height) * width);

    for (int write_step = 0; write_step < write_steps; write_step++) {
        auto step_start = std::chrono::high_resolution_clock::now();

        // copy GPU -> CPU, so the CPU has its own copy to write
        thrust::copy(d_prev.begin(), d_prev.end(), h_prev.begin());

        // GPU: run the simulation forward
        for (int s = 0; s < compute_steps; s++) {
            simulate(width, height, d_prev, d_next);
            d_prev.swap(d_next);
        }

        // CPU: write the copy to disk
        auto write_start = std::chrono::high_resolution_clock::now();
        store(write_step, height, width, h_prev);
        auto write_end = std::chrono::high_resolution_clock::now();

        // TODO 2: once simulate() uses CUB, nothing makes the CPU wait for the GPU.
        //         Add ONE cudaDeviceSynchronize() somewhere in main so the write
        //         overlaps the compute AND the timer below is honest.
        //         (Wrong spot = no overlap, or times that look impossibly fast.)

        auto step_end = std::chrono::high_resolution_clock::now();

        double write_ms = std::chrono::duration<double, std::milli>(write_end - write_start).count();
        double total_ms = std::chrono::duration<double, std::milli>(step_end - step_start).count();
        std::printf("round %d   compute + write: %7.1f ms   write: %7.1f ms\n",
                    write_step, total_ms, write_ms);
    }
}