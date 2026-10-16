%%writefile exercise4.cu
#include <cstdio>
#include <random>

#include <thrust/universal_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/transform_iterator.h>

// ===== EXERCISE 4 START =====
float median(thrust::universal_vector<float> vec)
{
    //
    thrust::sort(thrust::device, vec.begin(), vec.end());
    return vec[vec.size()/2];
}

float variance(const thrust::universal_vector<float>& x, float mean)
{
    //
    auto it = thrust::make_transform_iterator(
        x.begin(), 
        [mean] __host__ __device__(float xi) {
        return (xi - mean) * (xi - mean);
    });
    return thrust::reduce(thrust::host, it, it + x.size()) / x.size();
}
// ===== EXERCISE 4 END =====

int main()
{
    const int n = 1 << 20;
    thrust::universal_vector<float> x(n);

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(50.0f, 10.0f);  // mean 50, stddev 10
    for (int i = 0; i < n; i++) x[i] = dist(rng);

    float mean = thrust::reduce(thrust::device, x.begin(), x.end()) / n;

    printf("n        = %d\n", n);
    printf("mean     = %f  (expected ~50)\n", mean);
    printf("median   = %f  (expected ~50)\n", median(x));
    printf("variance = %f  (expected ~100)\n", variance(x, mean));
    return 0;
}