#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <sys/time.h>
using namespace std;

#define MY_PI 3.1415926
#define THREAD_NUM 4
double t;
const int width = 32;
const int height = 32;
float **RealTwo, **fxRealTwo, **fxImageTwo, **ifxRealTwo, **ifxImageTwo;

void init()
{
    RealTwo = new float *[width];
    fxRealTwo = new float *[width];
    fxImageTwo = new float *[width];
    ifxRealTwo = new float *[width];
    ifxImageTwo = new float *[width];
    for (int v = 0; v < width; v++)
    {
        RealTwo[v] = new float[height];
        fxRealTwo[v] = new float[height];
        fxImageTwo[v] = new float[height];
        ifxRealTwo[v] = new float[height];
        ifxImageTwo[v] = new float[height];
        for (int u = 0; u < height; u++)
        {
            RealTwo[v][u] = (rand() % 100 + 1) / 10;
            fxRealTwo[v][u] = 0.0;
            fxImageTwo[v][u] = 0.0;
            ifxRealTwo[v][u] = 0.0;
            ifxImageTwo[v][u] = 0.0;
        }
    }
}
void fft()
{
    for (int v = 0; v < width; v++)
    {
        for (int u = 0; u < height; u++)
        {
            for (int j = 0; j < width; j++)
            {
                for (int i = 0; i < height; i++)
                {
                    float w = 2 * MY_PI * (float)u * (float)i / (float)height + 2 * MY_PI * (float)v * (float)j / (float)width;
                    fxRealTwo[v][u] += RealTwo[j][i] * cos(w);
                    fxImageTwo[v][u] -= RealTwo[j][i] * sin(w);
                }
            }
        }
    }
}
void compare()
{
    for (int v = 0; v < width; v++)
    {
        for (int u = 0; u < height; u++)
        {
            if (fabs(ifxRealTwo[v][u] - RealTwo[v][u]) > 1e-2)
            {
                cout << "ifx: " << ifxRealTwo[v][u] << " real: " << RealTwo[v][u] << endl;
                cout << "Wrong!" << endl;
                return;
            }
        }
    }
    cout << "Correct!" << endl;
}

int main(int argc, char **argv)
{
    // 初始化
    MPI_Init(&argc, &argv);
    // rank 当前进程，size 进程数量
    int rank, size;
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double start_time = 0;
    double end_time = 0;
    int n = width / size;
    float **subReal = new float *[n];
    float **subImage = new float *[n];
    for (int i = 0; i < n; i++)
    {
        subReal[i] = new float[height];
        subImage[i] = new float[height];
    }
    init();
    fft();
    start_time = MPI_Wtime();
    // 开始计算
    for (int v = 0; v < width; v++)
    {
        if (rank == v / n)
        {
            int row = v % n;
            for (int u = 0; u < height; u++)
            {
                for (int j = 0; j < width; j++)
                {
                    for (int i = 0; i < height; i++)
                    {
                        float w = 2 * MY_PI * (float)u * (float)i / (float)height + 2 * MY_PI * (float)v * (float)j / (float)width;
                        subReal[row][u] += fxRealTwo[j][i] * cos(w) - fxImageTwo[j][i] * sin(w);
                        subImage[row][u] += fxImageTwo[j][i] * cos(w) + fxRealTwo[j][i] * sin(w);
                    }
                }
                subReal[row][u] /= (width * height);
                subImage[row][u] /= (width * height);
            }
        }
    }
    if (rank == 0)
    {
        for (int i = 1; i < size; i++)
        {
            for (int j = 0; j < n; j++)
            {
                MPI_Recv(&ifxRealTwo[i * n + j][0], height, MPI_FLOAT, i, j, MPI_COMM_WORLD, &status);
            }
        }
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < height; j++)
            {
                ifxRealTwo[i][j] = subReal[i][j];
            }
        }
        end_time = MPI_Wtime();
        cout << "Running time = " << (end_time - start_time) * 1000 << " ms" << endl;
        compare();
    }
    else
    {
        for (int i = 0; i < n; i++)
        {
            MPI_Send(&subReal[i][0], height, MPI_FLOAT, 0, i, MPI_COMM_WORLD);
        }
    }
    MPI_Finalize();
    return 0;
}