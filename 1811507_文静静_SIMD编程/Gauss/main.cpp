#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <windows.h>
#include <time.h>
#include <immintrin.h> // AVX
#include <xmmintrin.h> // SSE
#include <emmintrin.h> // SSE2
#include <pmmintrin.h> // SSE3
#include <tmmintrin.h> // SSSE3
#include <smmintrin.h> // SSE4.1
#include <nmmintrin.h> // SSSE4.2

using namespace std;

// 8,16,32,64,128,256,512,1024,2048
const int N = 1024;
long long head, tail, freq; // timers
double t;
float A[N][N];
const int epoch = 10; // 计算多次取平均比较

// 初始系数矩阵A，为了方便将其设置成单位矩阵/上三角矩阵
void init()
{
    for (int i = 0; i < N; i++)
    {
        A[i][i] = 1.0;
        // 上三角矩阵
        /*
        for (int j = i; i < N; i++)
        {
            A[i][j] = i + j + 1;
        }
        */
    }
}

// 比较两个矩阵判断误差
bool comp(float A[N][N], float B[N][N])
{
    for (int i = 0; i < N; i++)
    {
        for (int j = i; j < N; j++)
        {
            if (abs(A[i][j] - B[i][j]) > 1e-6)
            {
                cout << "误差超出10e-6，误差为：";
                cout << abs(A[i][j] - B[i][j]) << endl;
                return false;
            }
        }
    }
    cout << "误差未超出10e-6" << endl;
    return true;
}

// 高斯平凡算法
void Gauss_plain()
{
    for (int k = 0; k < N; k++)
    {
        for (int j = k + 1; j < N; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            for (int j = k + 1; j < N; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}

// SSE并行优化，对齐，4589向量化
void SSE_aligned_45_89()
{
    for (int k = 0; k < N; k++)
    {
        __m128 t1, t2, t3, t4;
        // 不能凑够4个的部分串行计算
        int remain = (N - k - 1) % 4;
        for (int j = k + 1; j <= k + remain; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        // 并行计算剩下4的倍数个，减少使用除法
        t1 = _mm_set1_ps(1 / A[k][k]);
        for (int j = k + remain + 1; j < N; j += 4)
        {
            t2 = _mm_load_ps(A[k] + j);
            t2 = _mm_mul_ps(t1, t2);
            _mm_store_ps(A[k] + j, t2);
        }
        A[k][k] = 1;
        for (int i = k + 1; i < N; i++)
        {
            // 不能凑够4个的部分串行计算
            for (int j = k + 1; j <= k + remain; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            t1 = _mm_set1_ps(A[i][k]);
            // 并行计算剩下4的倍数个float
            for (int j = k + remain + 1; j < N; j += 4)
            {
                t2 = _mm_load_ps(A[k] + j);
                t3 = _mm_mul_ps(t1, t2);
                t4 = _mm_load_ps(A[i] + j);
                t4 = _mm_sub_ps(t4, t3);
                _mm_store_ps(A[i] + j, t4);
            }
            A[i][k] = 0;
        }
    }
}

// SSE并行优化，对齐，45不向量化，89向量化
void SSE_aligned_u45_89()
{
    for (int k = 0; k < N; k++)
    {
        __m128 t1, t2, t3, t4;
        for (int j = k + 1; j < N; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            /*先把不能凑够4个的部分串行计算*/
            int remain = (N - k - 1) % 4;
            for (int j = k + 1; j <= k + remain; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            t1 = _mm_set1_ps(A[i][k]);
            /*并行计算剩下4的倍数个float*/
            for (int j = k + remain + 1; j < N; j += 4)
            {
                t2 = _mm_load_ps(A[k] + j);
                t3 = _mm_mul_ps(t1, t2);
                t4 = _mm_load_ps(A[i] + j);
                t4 = _mm_sub_ps(t4, t3);
                _mm_store_ps(A[i] + j, t4);
            }
            A[i][k] = 0;
        }
    }
}

// SSE并行优化，对齐，45向量化，89不向量化
void SSE_aligned_45_u89()
{
    for (int k = 0; k < N; k++)
    {
        __m128 t1, t2, t3, t4;
        // 不能凑够4个的部分串行计算
        int remain = (N - k - 1) % 4;
        for (int j = k + 1; j <= k + remain; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        // 并行计算剩下4的倍数个，减少使用除法
        t1 = _mm_set1_ps(1 / A[k][k]);
        for (int j = k + remain + 1; j < N; j += 4)
        {
            t2 = _mm_load_ps(A[k] + j);
            t2 = _mm_mul_ps(t1, t2);
            _mm_store_ps(A[k] + j, t2);
        }
        A[k][k] = 1;
        for (int i = k + 1; i < N; i++)
        {
            for (int j = k + 1; j < N; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}

// SSE并行优化，不对齐，4589向量化
void SSE_unaligned_45_89()
{
    for (int k = 0; k < N; k++)
    {
        __m128 t1, t2, t3, t4;
        // 不能凑够4个的部分串行计算
        int remain = (N - k - 1) % 4;
        for (int j = k + 1; j <= k + remain; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        // 并行计算剩下4的倍数个，减少使用除法
        t1 = _mm_set1_ps(1 / A[k][k]);
        for (int j = k + remain + 1; j < N; j += 4)
        {
            t2 = _mm_loadu_ps(A[k] + j);
            t2 = _mm_mul_ps(t1, t2);
            _mm_storeu_ps(A[k] + j, t2);
        }
        A[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            // 不能凑够4个的部分串行计算
            for (int j = k + 1; j <= k + remain; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            t1 = _mm_set1_ps(A[i][k]);
            // 并行计算剩下4的倍数个float
            for (int j = k + remain + 1; j < N; j += 4)
            {
                t2 = _mm_loadu_ps(A[k] + j);
                t3 = _mm_mul_ps(t1, t2);
                t4 = _mm_loadu_ps(A[i] + j);
                t4 = _mm_sub_ps(t4, t3);
                _mm_storeu_ps(A[i] + j, t4);
            }
            A[i][k] = 0;
        }
    }
}

// SSE并行优化，不对齐，45不向量化，89向量化
void SSE_unaligned_u45_89()
{
    for (int k = 0; k < N; k++)
    {
        __m128 t1, t2, t3, t4;
        for (int j = k + 1; j < N; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            // 先把不能凑够4个的部分串行计算
            int remain = (N - k - 1) % 4;
            for (int j = k + 1; j <= k + remain; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            t1 = _mm_set1_ps(A[i][k]);
            // 并行计算剩下4的倍数个float
            for (int j = k + remain + 1; j < N; j += 4)
            {
                t2 = _mm_loadu_ps(A[k] + j);
                t3 = _mm_mul_ps(t1, t2);
                t4 = _mm_loadu_ps(A[i] + j);
                t4 = _mm_sub_ps(t4, t3);
                _mm_storeu_ps(A[i] + j, t4);
            }
            A[i][k] = 0;
        }
    }
}

// SSE并行优化，不对齐，45向量化，89不向量化
void SSE_unaligned_45_u89()
{
    for (int k = 0; k < N; k++)
    {
        __m128 t1, t2, t3, t4;
        // 不能凑够4个的部分串行计算
        int remain = (N - k - 1) % 4;
        for (int j = k + 1; j <= k + remain; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        // 并行计算剩下4的倍数个，减少使用除法
        t1 = _mm_set1_ps(1 / A[k][k]);
        for (int j = k + remain + 1; j < N; j += 4)
        {
            t2 = _mm_loadu_ps(A[k] + j);
            t2 = _mm_mul_ps(t1, t2);
            _mm_storeu_ps(A[k] + j, t2);
        }
        A[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            for (int j = k + 1; j < N; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}

// AVX并行优化，对齐，向量化
void AVX_aligned_4589()
{
    for (int k = 0; k < N; k++)
    {
        __m256 t1, t2, t3, t4;
        // 不能凑够8个的部分串行计算
        int remain = (N - k - 1) % 8;
        for (int j = k + 1; j <= k + remain; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        // 并行计算剩下8的倍数个，减少使用除法
        t1 = _mm256_set1_ps(1 / A[k][k]);
        for (int j = k + remain + 1; j < N; j += 8)
        {
            t2 = _mm256_loadu_ps(A[k] + j);
            t2 = _mm256_mul_ps(t1, t2);
            _mm256_storeu_ps(A[k] + j, t2);
        }
        A[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            // 不能凑够8个的部分串行计算
            for (int j = k + 1; j <= k + remain; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            t1 = _mm256_set1_ps(A[i][k]);
            // 并行计算剩下8的倍数个float
            for (int j = k + remain + 1; j < N; j += 8)
            {
                t2 = _mm256_loadu_ps(A[k] + j);
                t3 = _mm256_mul_ps(t1, t2);
                t4 = _mm256_loadu_ps(A[i] + j);
                t4 = _mm256_sub_ps(t4, t3);
                _mm256_storeu_ps(A[i] + j, t4);
            }
            A[i][k] = 0;
        }
    }
}

// AVX并行优化，不对齐，向量化
void AVX_unaligned_4589()
{
    for (int k = 0; k < N; k++)
    {
        __m256 t1, t2, t3, t4;
        // 不能凑够8个的部分串行计算
        int remain = (N - k - 1) % 8;
        for (int j = k + 1; j <= k + remain; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        // 并行计算剩下8的倍数个，减少使用除法
        t1 = _mm256_set1_ps(1 / A[k][k]);
        for (int j = k + remain + 1; j < N; j += 8)
        {
            t2 = _mm256_loadu_ps(A[k] + j);
            t2 = _mm256_mul_ps(t1, t2);
            _mm256_storeu_ps(A[k] + j, t2);
        }
        A[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            // 不能凑够8个的部分串行计算
            for (int j = k + 1; j <= k + remain; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            t1 = _mm256_set1_ps(A[i][k]);
            // 并行计算剩下8的倍数个float
            for (int j = k + remain + 1; j < N; j += 8)
            {
                t2 = _mm256_loadu_ps(A[k] + j);
                t3 = _mm256_mul_ps(t1, t2);
                t4 = _mm256_loadu_ps(A[i] + j);
                t4 = _mm256_sub_ps(t4, t3);
                _mm256_storeu_ps(A[i] + j, t4);
            }
            A[i][k] = 0;
        }
    }
}

int main()
{
    cout << "当N=" << N << "时：" << endl;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    t = 0;
    for (int e = 0; e < epoch; e++)
    {
        init();                                          // 初始
        QueryPerformanceCounter((LARGE_INTEGER *)&head); //开始计时
        Gauss_plain();
        // SSE_aligned_45_89();
        // SSE_aligned_u45_89();
        // SSE_aligned_45_u89();
        // SSE_unaligned_45_89();
        // SSE_unaligned_u45_89();
        // SSE_unaligned_45_u89();
        // AVX_aligned_4589();
        // AVX_unaligned_4589();
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        t += (tail - head) * 1000.0 / freq;
    }
    cout << "总共耗时： " << t / epoch << "ms" << endl
         << endl;

    return 0;
}
