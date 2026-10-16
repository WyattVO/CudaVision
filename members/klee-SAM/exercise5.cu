%%writefile exercise5.cu
#include <cstdio>
#include <random>
#include <thrust/universal_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/transform_iterator.h>

// ===== EXERCISE 5 START =====
struct squared_diff {
    float mean;
    //
    __host__ __device__ float operator()(float xi) {
        return (xi - mean) * (xi - mean);
    }
};

float variance(const thrust::universal_vector<float>& x, float mean)
{
    //
    squared_diff sd{mean};
    auto it = thrust::make_transform_iterator(x.begin(), sd);
    return thrust::reduce(thrust::device, it, it + x.size()) / x.size();
}
// ===== EXERCISE 5 END =====

int main()
{
    const int n = 1 << 20;
    thrust::universal_vector<float> x(n);

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(50.0f, 10.0f);
    for (int i = 0; i < n; i++) x[i] = dist(rng);

    float mean = thrust::reduce(thrust::device, x.begin(), x.end()) / n;

    printf("n        = %d\n", n);
    printf("mean     = %f  (expected ~50)\n", mean);
    printf("variance = %f  (expected ~100)\n", variance(x, mean));
    return 0;
}