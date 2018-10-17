#include <iostream>
#include <immintrin.h>

int testAvx();

using namespace std;

int main() {
    return testAvx();
}

int testAvx() {
    float op1[8] = {2.2, 3.3, 4.4, 5.5, 5.5, 6.6, 7.7, 8.8};
    float op2[8] = {1.1, 2.2, 3.3, 4.4, 6.6, 7.7, 8.8, 9.9};
    float result[8];

    __m256 a = _mm256_loadu_ps(op1);
    __m256 b = _mm256_loadu_ps(op2);

    __m256 c = _mm256_sub_ps(a, b);   // c = a + b

    //c= _mm256_sqrt_ps(c);


    // Store
    _mm256_storeu_ps(result, c);

    for (int i = 0; i < 8; ++i) {
        printf("0: %lf\n", result[i]);
    }


    return 0;
}

