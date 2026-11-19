# CUDA reaction-diffusion: paste into one Colab cell (Runtime > Change runtime type > GPU)
import subprocess, numpy as np, matplotlib
from PIL import Image
from IPython.display import Image as Show, display

code = r'''
#include <cuda_runtime.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

const int W = 1024, H = 1024, N = W * H;
const int STEPS_PER_FRAME = 500, FRAMES = 40;
struct Cell { float u, v; };   // two chemicals in every pixel

#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    printf("CUDA error: %s\n", cudaGetErrorString(e)); exit(1); } } while (0)

// Gray-Scott: the same physics runs on both the CPU and the GPU
__host__ __device__ inline Cell update(const Cell* g, int x, int y) {
    const float Du = 0.16f, Dv = 0.08f, F = 0.055f, k = 0.062f;
    int l = x ? x - 1 : W - 1, r = x < W - 1 ? x + 1 : 0;
    int t = y ? y - 1 : H - 1, b = y < H - 1 ? y + 1 : 0;
    Cell c = g[y * W + x], c1 = g[y * W + l], c2 = g[y * W + r], c3 = g[t * W + x], c4 = g[b * W + x];
    float lapU = c1.u + c2.u + c3.u + c4.u - 4 * c.u;
    float lapV = c1.v + c2.v + c3.v + c4.v - 4 * c.v;
    float react = c.u * c.v * c.v;
    return { c.u + Du * lapU - react + F * (1 - c.u),
             c.v + Dv * lapV + react - (F + k) * c.v };
}

// Each GPU thread updates one pixel
__global__ void stepKernel(const Cell* in, Cell* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = update(in, i % W, i / W);
}

double now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

int main() {
    // Same random starting spots for both runs
    std::vector<Cell> start(N, Cell{1, 0});
    srand(7);
    for (int s = 0; s < 250; s++) {
        int cx = rand() % (W - 20), cy = rand() % (H - 20);
        for (int y = 0; y < 20; y++) for (int x = 0; x < 20; x++)
            start[(cy + y) * W + cx + x] = Cell{0.5f, 0.25f};
    }

    // CPU: every core, one frame's worth of steps
    std::vector<Cell> a = start, b(N);
    double t0 = now();
    for (int s = 0; s < STEPS_PER_FRAME; s++) {
        #pragma omp parallel for
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) b[y * W + x] = update(a.data(), x, y);
        a.swap(b);
    }
    double cpuTime = now() - t0;

    // GPU: all frames, timing only the simulation
    Cell* in, *out;
    CHECK(cudaMalloc(&in, N * sizeof(Cell)));
    CHECK(cudaMalloc(&out, N * sizeof(Cell)));
    CHECK(cudaMemcpy(in, start.data(), N * sizeof(Cell), cudaMemcpyHostToDevice));

    const int threads = 256, blocks = (N + threads - 1) / threads;
    std::vector<Cell> gpu(N);
    std::vector<float> v(N);
    FILE* file = fopen("frames.bin", "wb");
    double gpuFirst = 0, gpuTotal = 0, maxDiff = 0;

    for (int f = 0; f < FRAMES; f++) {
        t0 = now();
        for (int s = 0; s < STEPS_PER_FRAME; s++) {
            stepKernel<<<blocks, threads>>>(in, out);
            std::swap(in, out);
        }
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        double t = now() - t0;
        gpuTotal += t;
        if (f == 0) gpuFirst = t;

        CHECK(cudaMemcpy(gpu.data(), in, N * sizeof(Cell), cudaMemcpyDeviceToHost));
        for (int i = 0; i < N; i++) {
            v[i] = gpu[i].v;
            if (f == 0) maxDiff = std::max(maxDiff, (double)std::fabs(gpu[i].v - a[i].v));
        }
        fwrite(v.data(), sizeof(float), N, file);
    }
    fclose(file);
    cudaFree(in);
    cudaFree(out);

    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, 0));
    double work = 1e-6 * N * STEPS_PER_FRAME;
    printf("Grid: %dx%d, benchmark: %d steps\n", W, H, STEPS_PER_FRAME);
    printf("CPU (%2d threads): %8.3f s  %8.0f Mcells/s\n", omp_get_max_threads(), cpuTime, work / cpuTime);
    printf("GPU (%s): %8.3f s  %8.0f Mcells/s\n", p.name, gpuFirst, work / gpuFirst);
    printf("Speedup: %.1fx   Largest CPU/GPU difference: %.6f\n", cpuTime / gpuFirst, maxDiff);
    printf("Full animation: %d steps in %.2f s on GPU, about %.0f s on CPU\n",
           FRAMES * STEPS_PER_FRAME, gpuTotal, cpuTime * FRAMES);
}
'''
open("gray_scott.cu", "w").write(code)

build = "nvcc -O3 -arch=native -Xcompiler -fopenmp -o gray_scott gray_scott.cu -lgomp && ./gray_scott"
result = subprocess.run(build, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(result.stdout)
if result.returncode:
    raise SystemExit("Build or run failed. Check that this runtime has a GPU.")

frames = np.fromfile("frames.bin", dtype=np.float32).reshape(-1, 1024, 1024)[:, ::2, ::2]
frames = (np.clip(frames / frames.max(), 0, 1) * 255).astype(np.uint8)
palette = (matplotlib.colormaps["magma"](np.arange(256))[:, :3] * 255).astype(np.uint8).ravel().tolist()
images = [Image.fromarray(f).convert("P") for f in frames]
for image in images:
    image.putpalette(palette)
images[0].save("gray_scott.gif", save_all=True, append_images=images[1:], duration=90, loop=0)
display(Show(filename="gray_scott.gif"))
