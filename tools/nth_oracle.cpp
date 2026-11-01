// Test oracle: the element order GNU libstdc++'s std::nth_element leaves. Reads "n nth" then n floats per case from
// stdin; writes the resulting permutation of 0..n-1 per case. Compare with trianglebvh's selection (check_select.py).
#include <algorithm>
#include <cstdio>
#include <vector>
int main() {
    size_t n, nth;
    while (std::scanf("%zu %zu", &n, &nth) == 2) {
        std::vector<float> k(n);
        for (auto &x : k) std::scanf("%f", &x);
        std::vector<unsigned> p(n);
        for (size_t i = 0; i < n; i++) p[i] = i;
        std::nth_element(p.begin(), p.begin() + nth, p.end(), [&](unsigned a, unsigned b) { return k[a] < k[b]; });
        for (size_t i = 0; i < n; i++) std::printf("%u ", p[i]);
        std::printf("\n");
    }
}
