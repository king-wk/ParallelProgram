#include <iostream>
#include <algorithm>
#include <math.h>
#include <omp.h>  // OpenMP �ش�ͷ�ļ�
#include <time.h>
#include <windows.h>
#include <immintrin.h> // AVX
#include <xmmintrin.h> // SSE
#include <emmintrin.h> // SSE2
#include <pmmintrin.h> // SSE3
#include <tmmintrin.h> // SSSE3
#include <smmintrin.h> // SSE4.1
#include <nmmintrin.h> // SSSE4.2
using namespace std;

// 16,32,64,128,256,512,1024,2048
const int N = 16; // ��ģ��С
float A[N][N];
long long head, tail, freq; // timers
double t; // ��¼�������ʱ��
const int epoch = 10; // ������ȡƽ���Ƚ�

// ��ʼϵ������A��Ϊ�˷��㽫�����ó������Ǿ���
void init()
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < i; j++)
		{
			A[i][j] = 0;
		}
        for (int j = i; j < N; j++)
        {
            A[i][j] = i + j + 1;
        }
    }
}

// ��ȷ�ԱȽ�
void comp(float *A[N], float *B[N])
{
	for (int i = 0; i < N; i++)
	{
		for (int j = i; j < N; j++)
		{
			if (fabs(A[i][j] - B[i][j]) > 1e-4)
			{
				cout << "i = " << i << ", j = " << j << ", ����1e-4�����Ϊ��";
				cout << fabs(A[i][j] - B[i][j]) << endl;
				cout << "A[i][j] = " << A[i][j] << ", B[i][j] = " << B[i][j] << endl;
				return;
			}
		}
	}
	cout << "���δ����1e-4" << endl;
}

// �����㷨
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

// ���л���
void omp_row()
{
    #pragma omp parallel
    for (int k = 0; k < N; k++)
    {
        #pragma omp single
        {
            for (int j = k + 1; j < N; j++)
            {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
        }
        #pragma omp for
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


// ���л���
void omp_col()
{
    #pragma omp parallel
    for (int k = 0; k < N; k++)
    {
        #pragma omp single
        {
            for (int j = k + 1; j < N; j++)
            {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
        }
        #pragma omp for
        for (int j = k + 1; j < N; j++)
        {
            for (int i = k + 1; i < N; i++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
        }
        #pragma omp single
        for (int i = k + 1; i < N; i++)
        {
            A[i][k] = 0;
        }
    }
}

// ���л��֣���̬��
void omp_row_static()
{
    #pragma omp parallel
    for (int k = 0; k < N; k++)
    {
        #pragma omp single
        {
            for (int j = k + 1; j < N; j++)
            {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
        }
        #pragma omp for schedule(static)
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

// ���л��֣���̬��
void omp_row_dynmaic()
{
    #pragma omp parallel
    for (int k = 0; k < N; k++)
    {
        #pragma omp single
        {
            for (int j = k + 1; j < N; j++)
            {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
        }
        #pragma omp for schedule(dynamic, 8)
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

// ���л��֣�����ʽ��
void omp_row_guided()
{
    #pragma omp parallel
    for (int k = 0; k < N; k++)
    {
        #pragma omp single
        {
            for (int j = k + 1; j < N; j++)
            {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
        }
        #pragma omp for schedule(guided, 1)
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

// SSE
void omp_sse()
{
    __m128 t1, t2, t3, t4;
    int remain = 0;
    #pragma omp parallel private(t1, t2, t3, t4)
    for (int k = 0; k < N; k++)
    {
        #pragma omp single
        {
            // ���ܴչ�4���Ĳ��ִ��м���
            remain = (N - k - 1) % 4;
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
        }
        #pragma omp for
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

// AVX
void omp_avx()
{
    __m256 t1, t2, t3, t4;
    int remain = 0;
    #pragma omp parallel private(t1, t2, t3, t4)
    for (int k = 0; k < N; k++)
    {
        #pragma omp single
        {
            // ���ܴչ�8���Ĳ��ִ��м���
            remain = (N - k - 1) % 8;
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
        }
        #pragma omp for
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
	omp_set_num_threads(4);
	cout << "when N = " << N << endl;
	QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
	t = 0;
	for (int e = 0; e < epoch; e++)
	{
        init();
        QueryPerformanceCounter((LARGE_INTEGER *)&head); //��ʼ��ʱ
        // omp_avx();
        // omp_sse();
        // omp_row_guided();
        // omp_row_dynmaic();
        // omp_row_static();
        // omp_col();
        // omp_row();
        Gauss_plain();
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        t += (tail - head) * 1000.0 / freq;
	}
	cout << "time = " << t / epoch << " ms" << endl;
	return 0;
}
