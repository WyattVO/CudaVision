#include <cstdio>
#include <random>
#include <algorithm>
#include <vector>
#include <numeric>
#include <functional>

// ===== EXERCISE 4 START =====
float median(std::vector<float> vec)
{
    std::sort(vec.begin(), vec.end());
    return *(vec.begin() + vec.size()/2);
}


// mfw i reimplement cuda api in plain cpp for the learning experience
template <class F, class T> 
class transform_iterator 
{
private:
    T ptr;
    F op;
public:
    transform_iterator(F _func, T _iter) noexcept : op{_func}, ptr{_iter} {}

    T operator[](size_t i) 
    {
        return op(ptr[i]);
    }
};

template <class F, class T> 
inline constexpr auto make_transform_iterator(T _iter, F _func) 
{
    return transform_iterator<F, T>(_func, _iter);
}

float variance(const std::vector<float>& x, float mean)
{   
    auto op = [mean](float xi) {
        return (xi - mean) * (xi - mean);
    };
    // bad solution
    std::vector<float> y(x);
    std::transform(y.begin(), y.end(), y.begin(), op);
    return std::reduce(y.begin(), y.end()) / x.size();
}
// ===== EXERCISE 4 END =====

int main()
{
    const int n = 1 << 20;
    std::vector<float> x(n);

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(50.0f, 10.0f);  // mean 50, stddev 10
    for (int i = 0; i < n; i++) x[i] = dist(rng);

    float mean = std::reduce(x.begin(), x.end()) / n;

    printf("n        = %d\n", n);
    printf("mean     = %f  (expected ~50)\n", mean);
    printf("median   = %f  (expected ~50)\n", median(x));
    printf("variance = %f  (expected ~100)\n", variance(x, mean));
    return 0;
}
