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
const int NUM_THREADS = 8; // 线程数量
// 8,16,32,64,128,256,512,1024,2048
const int N = 1024; // 规模大小
long long head, tail, freq; // timers
double t; // 记录多次运行时间
float A[N][N]; // 系数矩阵
const int epoch = 10; // 计算多次取平均比较
bool flag; // 标志线程是否运行结束

typedef struct
{
    int threadId;
    int startPos;
} threadParm_t;

// 信号量
sem_t sem_parent;
sem_t sem_children;
sem_t sem_start;

// 初始系数矩阵A，为了方便将其设置成上三角矩阵
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

// 正确性比较
bool comp(float A[N][N], float B[N][N])
{
    for (int i = 0; i < N; i++)
    {
        for (int j = i; j < N; j++)
        {
            if (fabs(A[i][j] - B[i][j]) > 1e-6)
            {
                cout << "误差超出10e-6，误差为：";
                cout << fabs(A[i][j] - B[i][j]) << endl;
                return false;
            }
        }
    }
    cout << "误差未超出10e-6" << endl;
    return true;
}

// 串行算法
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

// 按行划分
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
        // 每一次消去之后，唤醒主线程，工作线程睡眠
        sem_post(&sem_parent);
        sem_wait(&sem_children);
    }
    pthread_exit(NULL); //返回结果给调用者
}

void pt_row()
{
    threadParm_t threadParm[NUM_THREADS];
    pthread_t thread[NUM_THREADS];
    sem_init(&sem_parent, 0, 0);
    sem_init(&sem_start, 0, 0);
    sem_init(&sem_children, 0, 0);
    // 开始时建立 NUM_THREADS 个工作线程
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
        // 完成除法之后将工作线程唤醒，主线程睡眠
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

// 按列划分
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
    // 开始时建立 NUM_THREADS 个工作线程
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
        // 完成除法之后将工作线程唤醒，主线程睡眠
        for (int i = 0; i < NUM_THREADS; i++)
        {
            sem_wait(&sem_parent);
        }
        // 主线程将第k列的k行以下的元素全置为0，在子线程完成之后进行
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

// 结合SSE，按行划分
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
            // 不能凑够4个的部分串行计算
            for (int j = k + 1; j <= k + remain; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            t1 = _mm_set1_ps(A[i][k]);
            // 并行计算剩下4的倍数个float
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
        // 每一次消去之后，唤醒主线程，工作线程睡眠
        sem_post(&sem_parent);
        sem_wait(&sem_children);
    }
    pthread_exit(NULL); //返回结果给调用者
}

void pt_sse_row()
{
    threadParm_t threadParm[NUM_THREADS];
    pthread_t thread[NUM_THREADS];
    sem_init(&sem_parent, 0, 0);
    sem_init(&sem_start, 0, 0);
    sem_init(&sem_children, 0, 0);
    // 开始时建立 NUM_THREADS 个工作线程
    flag = true;
    for (int i = 0; i < NUM_THREADS; i++)
    {
        threadParm[i].threadId = i;
        pthread_create(&thread[i], NULL, threadFunc_sse_row, (void *)&threadParm[i]);
    }
    for (int k = 0; k < N; k++)
    {
        __m128 t1, t2;
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
        for (int i = 0; i < NUM_THREADS; i++)
        {
            threadParm[i].startPos = k;
        }
        for (int i = 0; i < NUM_THREADS; i++)
        {
            sem_post(&sem_start);
        }
        // 完成除法之后将工作线程唤醒，主线程睡眠
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

// 结合SSE，按列划分
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
            // 不能凑够4个的部分串行计算
            for (int j = k + p->threadId + 1; j < N; j += NUM_THREADS)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
        }
        for (int i = k + 1 + remain; i < N; i += 4)
        {
            // 并行计算剩下4的倍数个float
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
        // 每一次消去之后，唤醒主线程，工作线程睡眠
        sem_post(&sem_parent);
        sem_wait(&sem_children);
    }
    pthread_exit(NULL); //返回结果给调用者
}

void pt_sse_col()
{
    threadParm_t threadParm[NUM_THREADS];
    pthread_t thread[NUM_THREADS];
    sem_init(&sem_parent, 0, 0);
    sem_init(&sem_start, 0, 0);
    sem_init(&sem_children, 0, 0);
    // 开始时建立 NUM_THREADS 个工作线程
    flag = true;
    for (int i = 0; i < NUM_THREADS; i++)
    {
        threadParm[i].threadId = i;
        pthread_create(&thread[i], NULL, threadFunc_sse_col, (void *)&threadParm[i]);
    }
    for (int k = 0; k < N; k++)
    {
        __m128 t1, t2;
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
        for (int i = 0; i < NUM_THREADS; i++)
        {
            threadParm[i].startPos = k;
        }
        for (int i = 0; i < NUM_THREADS; i++)
        {
            sem_post(&sem_start);
        }
        // 完成除法之后将工作线程唤醒，主线程睡眠
        for (int i = 0; i < NUM_THREADS; i++)
        {
            sem_wait(&sem_parent);
        }
        // 主线程将第k列的k行以下的元素全置为0，在子线程完成之后进行
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

// 结合AVX，按行划分
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
            // 不能凑够4个的部分串行计算
            for (int j = k + 1; j <= k + remain; j++)
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            t1 = _mm256_set1_ps(A[i][k]);
            // 并行计算剩下4的倍数个float
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
        // 每一次消去之后，唤醒主线程，工作线程睡眠
        sem_post(&sem_parent);
        sem_wait(&sem_children);
    }
    pthread_exit(NULL); //返回结果给调用者
}

void pt_avx_row()
{
    threadParm_t threadParm[NUM_THREADS];
    pthread_t thread[NUM_THREADS];
    sem_init(&sem_parent, 0, 0);
    sem_init(&sem_start, 0, 0);
    sem_init(&sem_children, 0, 0);
    // 开始时建立 NUM_THREADS 个工作线程
    flag = true;
    for (int i = 0; i < NUM_THREADS; i++)
    {
        threadParm[i].threadId = i;
        pthread_create(&thread[i], NULL, threadFunc_avx_row, (void *)&threadParm[i]);
    }
    for (int k = 0; k < N; k++)
    {
        __m256 t1, t2;
        // 不能凑够4个的部分串行计算
        int remain = (N - k - 1) % 8;
        for (int j = k + 1; j <= k + remain; j++)
        {
            A[k][j] = A[k][j] / A[k][k];
        }
        // 并行计算剩下4的倍数个，减少使用除法
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
        // 完成除法之后将工作线程唤醒，主线程睡眠
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
        QueryPerformanceCounter((LARGE_INTEGER *)&head); //开始计时
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
