#include <bits/stdc++.h>
#include <mpi.h>
#include <sys/time.h>
#include <omp.h>

using namespace std;

// �����ģ
const int N = 1024;

void Gauss_plain(float **B)
{
	for (int k = 0; k < N; k++)
	{
		for (int j = k + 1; j < N; j++)
		{
			B[k][j] = B[k][j] / B[k][k];
		}
		B[k][k] = 1.0;
		for (int i = k + 1; i < N; i++)
		{
			for (int j = k + 1; j < N; j++)
			{
				B[i][j] = B[i][j] - B[i][k] * B[k][j];
			}
			B[i][k] = 0;
		}
	}
}


void comp(float **A, float **B)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = i; j < N; j++)
        {
            if (fabs(A[i][j] - B[i][j]) > 1e-4)
            {
                cout << "error > 1e-4:";
                cout << fabs(A[i][j] - B[i][j]) << endl;
                return;
            }
        }
    }
    cout << "correct!" << endl;
}

int main(int argc, char **argv)
{
    // ��ʼ��
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    // rank ��ǰ���̣�size ��������
    int rank, size;
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double start_time = 0;
    double end_time = 0;
    float **A, **B;
    // 0 �Žڵ��ʼ������
    if (rank == 0)
    {
        cout << "N = " << N << " Number of Process: " << size << endl;
        A = new float *[N];
        for (int i = 0; i < N; i++)
            A[i] = new float[N];
        start_time = MPI_Wtime();
        //��ʼ������
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                if (i == j)
                    A[i][j] = 1;
                else
                    A[i][j] = 2;
            }
        }
    }
    else
    {
        A = new float *[1];
        A[0] = new float[1];
    }
    // �����������̵��Ӿ����СΪn * N
    int n = N / size;
    float **subA = new float *[n];
    for (int i = 0; i < n; i++)
        subA[i] = new float[N];
    // 0 �Žڵ�
    if (rank == 0)
    {
        for (int i = 0; i < n; i++)
            for (int j = 0; j < N; j++)
                subA[i][j] = A[i * size][j];
        for (int i = 0; i < N; i++)
        {
            // ѭ���黮��
            if (i % size != 0)
                // ����ÿһ�з��͵� dest = i % size, �������� tag = i / size + 1
                // size �������������� rank=0 ����
                MPI_Send(&A[i][0], N, MPI_FLOAT, i % size, i / size + 1, MPI_COMM_WORLD);
        }
        // cout << "subA:" << endl;
        // for (int i = 0; i < n; i++)
        // {
        //     for (int j = 0; j < N; j++)
        //         cout << subA[i][j] << " ";
        //     cout << endl;
        // }
    }
    else
    {
        // 0 �Ž��̽���
        for (int i = 0; i < n; i++)
            MPI_Recv(&subA[i][0], N, MPI_FLOAT, 0, i + 1, MPI_COMM_WORLD, &status);
    }
    float *line = new float[N];
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < size; j++)
        {
            // ���е���i����ĵ�j��
            int row = i * size + j;
            if (rank == j)
            {
                // �����������㲢������㲥������Ľڵ㣬�Ա����ڵ������ȥ
                for (int k = row + 1; k < N; k++)
                {
                    subA[i][k] = subA[i][k] / subA[i][row];
                    line[k] = subA[i][k];
                }
                subA[i][row] = 1;
                line[row] = 1;
                // cout << "line "<< row <<" to be bcasted: " << endl;
                // for (int t = 0; t < N; t++)
                // 	cout << line[t] << " ";
                // cout << endl;
            }
            MPI_Bcast(line, N, MPI_FLOAT, j, MPI_COMM_WORLD);
            // ��ȥ�������½�
            int kk = i;
            if (rank <= j)
            {
                kk = i + 1;
            }
#pragma omp parallel for num_threads(provided)
            for (int k = kk; k < n; k++)
            {
                for (int w = row + 1; w < N; w++)
                    subA[k][w] = subA[k][w] - line[w] * subA[k][row];
                subA[k][row] = 0;
            }
        }
    }
    // rank=0 �Ľ����ռ�������
    if (rank == 0)
    {
        for (int i = 1; i < size; i++)
        {
            for (int j = 0; j < n; j++)
            {
                MPI_Recv(&A[j * size + i][0], N, MPI_FLOAT, i, j, MPI_COMM_WORLD, &status);
            }
        }
        for (int i = 0; i < n; i++)
            for (int j = 0; j < N; j++)
                A[i * size][j] = subA[i][j];
        end_time = MPI_Wtime();
        cout << "Running Time: " << (end_time - start_time) * 1000 << "ms" << endl;

        // ������
		B = new float *[N];
		for (int i = 0; i < N; i++)
		{
			B[i] = new float[N];
		}
		timeval tv_start, tv_end;
    	gettimeofday(&tv_start, NULL);
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
				if (i == j)
					B[i][j] = 1;
				else
					B[i][j] = 2;
		}
		for (int k = 0; k < N; k++)
		{
			for (int j = k + 1; j < N; j++)
			{
				B[k][j] = B[k][j] / B[k][k];
			}
			B[k][k] = 1.0;
			for (int i = k + 1; i < N; i++)
			{
				for (int j = k + 1; j < N; j++)
				{
					B[i][j] = B[i][j] - B[i][k] * B[k][j];
				}
				B[i][k] = 0;
			}
		}
		gettimeofday(&tv_end, NULL);
    	cout << "Plain running time: " << 1000 * (tv_end.tv_sec - tv_start.tv_sec) + (tv_end.tv_usec - tv_start.tv_usec) / 1000 << "ms" << endl;
		comp(A, B);
    }
    else
    {
        for (int i = 0; i < n; i++)
            MPI_Send(&subA[i][0], N, MPI_FLOAT, 0, i, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return 0;
}