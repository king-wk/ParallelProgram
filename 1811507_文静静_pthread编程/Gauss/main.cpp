#include <iostream>
#include <algorithm>
#include <math.h>
#include <pthread.h>
#include <semaphore.h>
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
const int NUM_THREADS = 8; // �߳�����
// 8,16,32,64,128,256,512,1024,2048
const int N = 1024; // ��ģ��С
long long head, tail, freq; // timers
double t; // ��¼�������ʱ��
float A[N][N]; // ϵ������
const int epoch = 10; // ������ȡƽ���Ƚ�
bool flag; // ��־�߳��Ƿ����н���

typedef struct
{
    int threadId;
    int startPos;
} threadParm_t;

// �ź���
sem_t sem_parent;
sem_t sem_children;
sem_t sem_start;

// ��ʼϵ������A��Ϊ�˷��㽫�����ó������Ǿ���
void init()
{
    for (int i = 0; i < N; i++)
    {
        for (int j = i; j < N; j++)
        {
            A[i][j] = i + j + 1;
        }
    }
}

// ��ȷ�ԱȽ�
bool comp(float A[N][N], float B[N][N])
{
    for (int i = 0; i < N; i++)
    {
        for (int j = i; j < N; j++)
        {
            if (fabs(A[i][j] - B[i][j]) > 1e-6)
            {
                cout << "����10e-6�����Ϊ��";
                cout << fabs(A[i][j] - B[i][j]) << endl;
                return false;
            }
        }
    }
    cout << "���δ����10e-6" << endl;
    return true;
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
void *threadFunc_row(void *parm)
{
    threadParm_t *p = (threadParm_t *)parm;
    while (true)
    {
        sem_wait(&sem_start);
        if (!flag)
            break;
        for (int i = p->startPos + p->threadId + 1; i < N; i += NUM_THREADS)
        {
            for (int j = p->startPos + 1; j < N; j++)
            {
                A[i][j] = A[i][j] - A[i][p->startPos] * A[p->startPos][j];
            }
            A[i][p->startPos] = 0.0;
        }
        // ÿһ����ȥ֮�󣬻������̣߳������߳�˯��
        sem_post(&sem_parent);
        sem_wait(&sem_children);
    }
    pthread_exit(NULL); //���ؽ����������
}

void pt_row()
{
    threadParm_t threadParm[NUM_THREADS];
    pthread_t thread[NUM_THREADS];
    sem_init(&sem_parent, 0, 0);
    sem_init(&sem_start, 0, 0);
    sem_init(&sem_children, 0, 0);
    // ��ʼʱ���� NUM_THREADS �������߳�
    flag = true;
    for (int i = 0; i < NUM_THREADS; i++)
    {
        threadParm[i].threadId = i;
        pthread_create(&thread[i], NULL, threadFunc_row, (void *)&threadParm[i]);
    }
    for (int k = 0; k < N; k++)
    {
        for (int j = k + 1; j < N; j++)
        {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1;
        for (int i = 0; i < NUM_THREADS; i++)
        {
            threadParm[i].startPos = k;
        }
        for (int i = 0; i < NUM_THREADS; i++)
        {
            sem_post(&sem_start);
        }
        // ��ɳ���֮�󽫹����̻߳��ѣ����߳�˯��
        for (int i = 0; i < NUM_THREADS; i++)
        {
            sem_wait(&sem_parent);
        }
        for (int i = 0; i < NUM_THREADS; i++)
        {
            sem_post(&sem_children);
        }
    }
    flag = false;
    for (int i = 0; i < NUM_THREADS; i++)
    {
        sem_post(&sem_start);
    }
    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(thread[i], NULL);
    }
    sem_destroy(&sem_start);
    sem_destroy(&sem_children);
    sem_destroy(&sem_parent);
}

// ���л���
void *threadFunc_col(void *parm)
{
    threadParm_t *p = (threadParm_t *)parm;
    while (true)
    {
        sem_wait(&sem_start);
        if (!flag)
            break;
        for (int i = p->startPos + 1; i < N; i++)
        {
            for (int j = p->startPos + p->threadId + 1; j < N; j += NUM_THREADS)
            {
                A[i][j] = A[i][j] - A[i][p->startPos] * A[p->startPos][j];
            }
        }
        sem_post(&sem_parent);
        sem_wait(&sem_children);
    }
    pthread_exit(NULL);
}

void pt_col()
{
    threadParm_t threadParm[NUM_THREADS];
    pthread_t thread[NUM_THREADS];
    sem_init(&sem_parent, 0, 0);
    sem_init(&sem_start, 0, 0);
    sem_init(&sem_children, 0, 0);
    // ��ʼʱ���� NUM_THREADS �������߳�
    flag = true;
    for (int i = 0; i < NUM_THREADS; i++)
    {
        threadParm[i].threadId = i;
        pthread_create(&thread[i], NULL, threadFunc_col, (void *)&threadParm[i]);
    }
    for (int k = 0; k < N; k++)
    {
        for (int j = k + 1; j < N; j++)
        {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1;
        for (int i = 0; i < NUM_THREADS; i++)
        {
            threadParm[i].startPos = k;
        }
        for (int i = 0; i < NUM_THREADS; i++)
        {
            sem_post(&sem_start);
        }
        // ��ɳ���֮�󽫹����̻߳��ѣ����߳�˯��
        for (int i = 0; i < NUM_THREADS; i++)
        {
            sem_wait(&sem_parent);
        }
        // ���߳̽���k�е�k�����µ�Ԫ��ȫ��Ϊ0�������߳����֮�����
        for (int i = k + 1; i < N; i++)
        {
            A[i][k] = 0.0;
        }
        for (int i = 0; i < NUM_THREADS; i++)
        {
            sem_post(&sem_children);
        }
    }
    flag = false;
    for (int i = 0; i < NUM_THREADS; i++)
    {
        sem_post(&sem_start);
    }
    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(thread[i], NULL);
    }
    sem_destroy(&sem_start);
    sem_destroy(&sem_children);
    sem_destroy(&sem_parent);
}

// ���SSE�����л���
void *threadFunc_sse_row(void *parm)
{
    threadParm_t *p = (threadParm_t *)parm;
    __m128 t1, t2, t3, t4;
    while (true)
    {
        sem_wait(&sem_start);
        if (!flag)
            break;
        int k = p->startPos;
        for (int i = p->startPos + p->threadId + 1; i < N; i += NUM_THREADS)
        {
            int remain = (N - k - 1) % 4;
            // ���ܴչ�4���Ĳ��ִ��м���
            for (int j = k + 1; j <= k + remain; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            t1 = _mm_set1_ps(A[i][k]);
            // ���м���ʣ��4�ı�����float
            for (int j = p->startPos + remain + 1; j < N; j += 4)
            {
                t2 = _mm_load_ps(A[k] + j);
                t3 = _mm_mul_ps(t1, t2);
                t4 = _mm_load_ps(A[i] + j);
                t4 = _mm_sub_ps(t4, t3);
                _mm_store_ps(A[i] + j, t4);
            }
            A[i][p->startPos] = 0.0;
        }
        // ÿһ����ȥ֮�󣬻������̣߳������߳�˯��
        sem_post(&sem_parent);
        sem_wait(&sem_children);
    }
    pthread_exit(NULL); //���ؽ����������
}

void pt_sse_row()
{
    threadParm_t threadParm[NUM_THREADS];
    pthread_t thread[NUM_THREADS];
    sem_init(&sem_parent, 0, 0);
    sem_init(&sem_start, 0, 0);
    sem_init(&sem_children, 0, 0);
    // ��ʼʱ���� NUM_THREADS �������߳�
    flag = true;
    for (int i = 0; i < NUM_THREADS; i++)
    {
        threadParm[i].threadId = i;
        pthread_create(&thread[i], NULL, threadFunc_sse_row, (void *)&threadParm[i]);
    }
    for (int k = 0; k < N; k++)
    {
        __m128 t1, t2;
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
        for (int i = 0; i < NUM_THREADS; i++)
        {
            threadParm[i].startPos = k;
        }
        for (int i = 0; i < NUM_THREADS; i++)
        {
            sem_post(&sem_start);
        }
        // ��ɳ���֮�󽫹����̻߳��ѣ����߳�˯��
        for (int i = 0; i < NUM_THREADS; i++)
        {
            sem_wait(&sem_parent);
        }
        for (int i = 0; i < NUM_THREADS; i++)
        {
            sem_post(&sem_children);
        }
    }
    flag = false;
    for (int i = 0; i < NUM_THREADS; i++)
    {
        sem_post(&sem_start);
    }
    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(thread[i], NULL);
    }
    sem_destroy(&sem_start);
    sem_destroy(&sem_children);
    sem_destroy(&sem_parent);
}

// ���SSE�����л���
void *threadFunc_sse_col(void *parm)
{
    threadParm_t *p = (threadParm_t *)parm;
    __m128 t1, t2, t3, t4;
    float f1[4], f2[4];
    while (true)
    {
        sem_wait(&sem_start);
        if (!flag)
            break;
        int k = p->startPos;
        int remain = (N - k - 1) % 4;
        for (int i = k + 1; i < k + 1 + remain; i++)
        {
            // ���ܴչ�4���Ĳ��ִ��м���
            for (int j = k + p->threadId + 1; j < N; j += NUM_THREADS)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
        }
        for (int i = k + 1 + remain; i < N; i += 4)
        {
            // ���м���ʣ��4�ı�����float
            for (int j = k + p->threadId + 1; j < N; j += NUM_THREADS)
            {
                t1 = _mm_set1_ps(A[k][j]);
                for (int m = 0; m < 4; m++)
                {
                    f1[m] = A[i + m][k];
                    f2[m] = A[i + m][j];
                }
                t2 = _mm_load_ps(f1);
                t3 = _mm_mul_ps(t1, t2);
                t4 = _mm_load_ps(f2);
                t4 = _mm_sub_ps(t4, t3);
                _mm_store_ps(f1, t4);
                for (int m = 0; m < 4; m++)
                {
                    A[i + m][j] = f1[m];
                }
            }
        }
        // ÿһ����ȥ֮�󣬻������̣߳������߳�˯��
        sem_post(&sem_parent);
        sem_wait(&sem_children);
    }
    pthread_exit(NULL); //���ؽ����������
}

void pt_sse_col()
{
    threadParm_t threadParm[NUM_THREADS];
    pthread_t thread[NUM_THREADS];
    sem_init(&sem_parent, 0, 0);
    sem_init(&sem_start, 0, 0);
    sem_init(&sem_children, 0, 0);
    // ��ʼʱ���� NUM_THREADS �������߳�
    flag = true;
    for (int i = 0; i < NUM_THREADS; i++)
    {
        threadParm[i].threadId = i;
        pthread_create(&thread[i], NULL, threadFunc_sse_col, (void *)&threadParm[i]);
    }
    for (int k = 0; k < N; k++)
    {
        __m128 t1, t2;
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
        for (int i = 0; i < NUM_THREADS; i++)
        {
            threadParm[i].startPos = k;
        }
        for (int i = 0; i < NUM_THREADS; i++)
        {
            sem_post(&sem_start);
        }
        // ��ɳ���֮�󽫹����̻߳��ѣ����߳�˯��
        for (int i = 0; i < NUM_THREADS; i++)
        {
            sem_wait(&sem_parent);
        }
        // ���߳̽���k�е�k�����µ�Ԫ��ȫ��Ϊ0�������߳����֮�����
        for (int i = k + 1; i < N; i++)
        {
            A[i][k] = 0.0;
        }
        for (int i = 0; i < NUM_THREADS; i++)
        {
            sem_post(&sem_children);
        }
    }
    flag = false;
    for (int i = 0; i < NUM_THREADS; i++)
    {
        sem_post(&sem_start);
    }
    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(thread[i], NULL);
    }
    sem_destroy(&sem_start);
    sem_destroy(&sem_children);
    sem_destroy(&sem_parent);
}

// ���AVX�����л���
void *threadFunc_avx_row(void *parm)
{
    threadParm_t *p = (threadParm_t *)parm;
    __m256 t1, t2, t3, t4;
    while (true)
    {
        sem_wait(&sem_start);
        if (!flag)
            break;
        int k = p->startPos;
        for (int i = p->startPos + p->threadId + 1; i < N; i += NUM_THREADS)
        {
            int remain = (N - k - 1) % 8;
            // ���ܴչ�4���Ĳ��ִ��м���
            for (int j = k + 1; j <= k + remain; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            t1 = _mm256_set1_ps(A[i][k]);
            // ���м���ʣ��4�ı�����float
            for (int j = p->startPos + remain + 1; j < N; j += 8)
            {
                t2 = _mm256_loadu_ps(A[k] + j);
				t3 = _mm256_mul_ps(t1, t2);
				t4 = _mm256_loadu_ps(A[i] + j);
				t4 = _mm256_sub_ps(t4, t3);
				_mm256_storeu_ps(A[i] + j, t4);
            }
            A[i][p->startPos] = 0.0;
        }
        // ÿһ����ȥ֮�󣬻������̣߳������߳�˯��
        sem_post(&sem_parent);
        sem_wait(&sem_children);
    }
    pthread_exit(NULL); //���ؽ����������
}

void pt_avx_row()
{
    threadParm_t threadParm[NUM_THREADS];
    pthread_t thread[NUM_THREADS];
    sem_init(&sem_parent, 0, 0);
    sem_init(&sem_start, 0, 0);
    sem_init(&sem_children, 0, 0);
    // ��ʼʱ���� NUM_THREADS �������߳�
    flag = true;
    for (int i = 0; i < NUM_THREADS; i++)
    {
        threadParm[i].threadId = i;
        pthread_create(&thread[i], NULL, threadFunc_avx_row, (void *)&threadParm[i]);
    }
    for (int k = 0; k < N; k++)
    {
        __m256 t1, t2;
        // ���ܴչ�4���Ĳ��ִ��м���
        int remain = (N - k - 1) % 8;
        for (int j = k + 1; j <= k + remain; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        // ���м���ʣ��4�ı�����������ʹ�ó���
        t1 = _mm256_set1_ps(1 / A[k][k]);
        for (int j = k + remain + 1; j < N; j += 8)
        {
            t2 = _mm256_loadu_ps(A[k] + j);
			t2 = _mm256_mul_ps(t1, t2);
			_mm256_storeu_ps(A[k] + j, t2);
        }
        A[k][k] = 1;
        for (int i = 0; i < NUM_THREADS; i++)
        {
            threadParm[i].startPos = k;
        }
        for (int i = 0; i < NUM_THREADS; i++)
        {
            sem_post(&sem_start);
        }
        // ��ɳ���֮�󽫹����̻߳��ѣ����߳�˯��
        for (int i = 0; i < NUM_THREADS; i++)
        {
            sem_wait(&sem_parent);
        }
        for (int i = 0; i < NUM_THREADS; i++)
        {
            sem_post(&sem_children);
        }
    }
    flag = false;
    for (int i = 0; i < NUM_THREADS; i++)
    {
        sem_post(&sem_start);
    }
    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(thread[i], NULL);
    }
    sem_destroy(&sem_start);
    sem_destroy(&sem_children);
    sem_destroy(&sem_parent);
}

int main()
{
    cout << "when N = " << N << endl;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    t = 0;
    for (int e = 0; e < epoch; e++)
    {
        init();
        QueryPerformanceCounter((LARGE_INTEGER *)&head); //��ʼ��ʱ
        // pt_avx_row();
        // pt_sse_col();
        // pt_sse_row();
        // pt_col();
        // pt_row();
        Gauss_plain();
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        t += (tail - head) * 1000.0 / freq;
    }
    cout << "time = " << t / epoch << " ms" << endl;
    return 0;
}
