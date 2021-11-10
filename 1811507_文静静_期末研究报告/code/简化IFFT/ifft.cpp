#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <windows.h>
using namespace std;

#define MY_PI 3.1415926
#define THREAD_NUM 4
long long head, tail, freq; // timers
double t;
const int epoch = 10; // 计算多次取平均比较

const int width = 32;
const int height = 32;
float **RealTwo, **fxRealTwo, **fxImageTwo, **ifxRealTwo, **ifxImageTwo;

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

void ifft()
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
                    ifxRealTwo[v][u] += fxRealTwo[j][i] * cos(w) - fxImageTwo[j][i] * sin(w);
                    ifxImageTwo[v][u] += fxImageTwo[j][i] * cos(w) + fxRealTwo[j][i] * sin(w);
                }
            }
            ifxRealTwo[v][u] /= (width * height);
            ifxImageTwo[v][u] /= (width * height);
        }
    }
}

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

int main()
{
    t = 0.0;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    for (int e = 0; e < epoch; e++)
    {
        init();
        fft();
        QueryPerformanceCounter((LARGE_INTEGER *)&head); //开始计时
        ifft();
        QueryPerformanceCounter((LARGE_INTEGER *)&tail);
        t += (tail - head) * 1000.0 / freq;
        compare();
    }
    cout << "the time = " << t / epoch << " ms" << endl;
    system("pause");
    return 0;
}