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
const int epoch = 10; // ������ȡƽ���Ƚ�

// ��ʼϵ������A��Ϊ�˷��㽫�����óɵ�λ����/�����Ǿ���
void init()
{
    for (int i = 0; i < N; i++)
    {
        A[i][i] = 1.0;
        // �����Ǿ���
        /*
        for (int j = i; i < N; i++)
        {
            A[i][j] = i + j + 1;
        }
        */
    }
}

// �Ƚ����������ж����
bool comp(float A[N][N], float B[N][N])
{
    for (int i = 0; i < N; i++)
    {
        for (int j = i; j < N; j++)
        {
            if (abs(A[i][j] - B[i][j]) > 1e-6)
            {
                cout << "����10e-6�����Ϊ��";
                cout << abs(A[i][j] - B[i][j]) << endl;
                return false;
            }
        }
    }
    cout << "���δ����10e-6" << endl;
    return true;
}

// ��˹ƽ���㷨
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

// SSE�����Ż������룬4589������
void SSE_aligned_45_89()
{
    for (int k = 0; k < N; k++)
    {
        __m128 t1, t2, t3, t4;
        // ���ܴչ�4���Ĳ��ִ��м���
        int remain = (N - k - 1) % 4;
        for (int j = k + 1; j <= k + remain; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        // ���м���ʣ��4�ı�����������ʹ�ó���
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
            // ���ܴչ�4���Ĳ��ִ��м���
            for (int j = k + 1; j <= k + remain; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            t1 = _mm_set1_ps(A[i][k]);
            // ���м���ʣ��4�ı�����float
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

// SSE�����Ż������룬45����������89������
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
            /*�ȰѲ��ܴչ�4���Ĳ��ִ��м���*/
            int remain = (N - k - 1) % 4;
            for (int j = k + 1; j <= k + remain; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            t1 = _mm_set1_ps(A[i][k]);
            /*���м���ʣ��4�ı�����float*/
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

// SSE�����Ż������룬45��������89��������
void SSE_aligned_45_u89()
{
    for (int k = 0; k < N; k++)
    {
        __m128 t1, t2, t3, t4;
        // ���ܴչ�4���Ĳ��ִ��м���
        int remain = (N - k - 1) % 4;
        for (int j = k + 1; j <= k + remain; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        // ���м���ʣ��4�ı�����������ʹ�ó���
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

// SSE�����Ż��������룬4589������
void SSE_unaligned_45_89()
{
    for (int k = 0; k < N; k++)
    {
        __m128 t1, t2, t3, t4;
        // ���ܴչ�4���Ĳ��ִ��м���
        int remain = (N - k - 1) % 4;
        for (int j = k + 1; j <= k + remain; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        // ���м���ʣ��4�ı�����������ʹ�ó���
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
            // ���ܴչ�4���Ĳ��ִ��м���
            for (int j = k + 1; j <= k + remain; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            t1 = _mm_set1_ps(A[i][k]);
            // ���м���ʣ��4�ı�����float
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

// SSE�����Ż��������룬45����������89������
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
            // �ȰѲ��ܴչ�4���Ĳ��ִ��м���
            int remain = (N - k - 1) % 4;
            for (int j = k + 1; j <= k + remain; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            t1 = _mm_set1_ps(A[i][k]);
            // ���м���ʣ��4�ı�����float
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

// SSE�����Ż��������룬45��������89��������
void SSE_unaligned_45_u89()
{
    for (int k = 0; k < N; k++)
    {
        __m128 t1, t2, t3, t4;
        // ���ܴչ�4���Ĳ��ִ��м���
        int remain = (N - k - 1) % 4;
        for (int j = k + 1; j <= k + remain; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        // ���м���ʣ��4�ı�����������ʹ�ó���
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

// AVX�����Ż������룬������
void AVX_aligned_4589()
{
    for (int k = 0; k < N; k++)
    {
        __m256 t1, t2, t3, t4;
        // ���ܴչ�8���Ĳ��ִ��м���
        int remain = (N - k - 1) % 8;
        for (int j = k + 1; j <= k + remain; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        // ���м���ʣ��8�ı�����������ʹ�ó���
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
            // ���ܴչ�8���Ĳ��ִ��м���
            for (int j = k + 1; j <= k + remain; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            t1 = _mm256_set1_ps(A[i][k]);
            // ���м���ʣ��8�ı�����float
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

// AVX�����Ż��������룬������
void AVX_unaligned_4589()
{
    for (int k = 0; k < N; k++)
    {
        __m256 t1, t2, t3, t4;
        // ���ܴչ�8���Ĳ��ִ��м���
        int remain = (N - k - 1) % 8;
        for (int j = k + 1; j <= k + remain; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        // ���м���ʣ��8�ı�����������ʹ�ó���
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
            // ���ܴչ�8���Ĳ��ִ��м���
            for (int j = k + 1; j <= k + remain; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            t1 = _mm256_set1_ps(A[i][k]);
            // ���м���ʣ��8�ı�����float
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
    cout << "��N=" << N << "ʱ��" << endl;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    t = 0;
    for (int e = 0; e < epoch; e++)
    {
        init();                                          // ��ʼ
        QueryPerformanceCounter((LARGE_INTEGER *)&head); //��ʼ��ʱ
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
    cout << "�ܹ���ʱ�� " << t / epoch << "ms" << endl
         << endl;

    return 0;
}
