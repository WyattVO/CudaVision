#include <cstdio>
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>




// ===== EXERCISE 5 START =====
struct squared_diff {
    float mean;
    // TODO: add a __host__ __device__ operator() that takes a float xi
    //       and returns (xi - mean) * (xi - mean)
    float operator()(float xi) {
        return (xi - mean) * (xi - mean);
    }
};

float variance(const std::vector<float>& x, float mean)
{
    // TODO: build a transform iterator using squared_diff{mean},
    //       then reduce it on the GPU and divide by x.size()
    squared_diff sd{mean};
    // bad
    std::vector<float> y(x);
    std::transform(x.begin(), x.end(), y.begin(), sd);
    return std::reduce(y.begin(), y.end()) / x.size();
}
// ===== EXERCISE 5 END =====

int main()
{
    const int n = 1 << 20;
    std::vector<float> x(n);

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(50.0f, 10.0f);
    for (int i = 0; i < n; i++) x[i] = dist(rng);

    float mean = std::reduce(x.begin(), x.end()) / n;

    printf("n        = %d\n", n);
    printf("mean     = %f  (expected ~50)\n", mean);
    printf("variance = %f  (expected ~100)\n", variance(x, mean));
    return 0;
}