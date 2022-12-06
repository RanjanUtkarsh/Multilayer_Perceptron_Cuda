/*************************************************************************
/* ECE 277: GPU Programmming 2020
/* Author and Instructer: Cheolhong An
/* Copyright 2020
/* University of California, San Diego
/*************************************************************************/

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

__global__ void kernel_multiply(float* A, float *B, float* C, int M, int N, int K);

void main()
{
	int M = 3;
	int N = 3;
	int K = 3;
	float* d_a, * d_b, * d_c;
	float* A = (float*)malloc(N * M * sizeof(float));
	float* B = (float*)malloc(N * M * sizeof(float));
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < M; j++)
		{
			A[i * M + j] = 1;
			B[i * M + j] = 1;
		}
	}
	//*A = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
	//*B = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
	
	
	float* C = (float*)malloc(N * M * sizeof(float));

	dim3 blk;
	blk.x = 16; blk.y = 16; blk.z = 1;

	dim3 grid;
	grid.x = (M + blk.x - 1) / blk.x;
	grid.y = (N + blk.y - 1) / blk.y;
	grid.z = 1;

	int size_a = sizeof(float) * N * K;
	int size_b = sizeof(float) * K * M;
	int size_c = sizeof(float) * N * M;

	cudaMalloc((void**)&d_a, size_a);
	cudaMalloc((void**)&d_b, size_b);
	cudaMalloc((void**)&d_c, size_c);

	cudaMemcpy(d_a, A, size_a, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, B, size_b, cudaMemcpyHostToDevice);

	kernel_multiply << < grid, blk >> > (d_a, d_b, d_c, M, N, K);

	cudaMemcpy(C, d_c, size_c, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}

void cu_multiply(float* A, float *B, float* C, int M, int N, int K)
{
	// A = NxK, B = KxM matrix
	float* d_a, *d_b, *d_c;

	dim3 blk;
	blk.x = 1; blk.y = 1; blk.z = 1;

	dim3 grid;
	grid.x = (M + blk.x - 1) / blk.x;
	grid.y = (N + blk.y - 1) / blk.y;
	grid.z = 1;

	int size_a = sizeof(float) * N * K;
	int size_b = sizeof(float) * K * M;
	int size_c = sizeof(float) * N * M;

	cudaMalloc((void**)&d_a, size_a);
	cudaMalloc((void**)&d_b, size_b);
	cudaMalloc((void**)&d_c, size_c);

	cudaMemcpy(d_a, A, size_a, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, B, size_b, cudaMemcpyHostToDevice);

	//kernel_multiply << < grid, blk >> > (d_a, d_b, d_c, M, N, K);
	kernel_multiply << < 1, 1 >> > (d_a, d_b, d_c, M, N, K);

	cudaMemcpy(C, d_c, size_c, cudaMemcpyDeviceToHost);

	/*
	for (int I = 0; I < N; I++)
	{
		for (int J = 0; J < M; J++)
		{
			float _c = 0;
			for (unsigned int k = 0; k < K; k++)
			{
				float a = A[I * K + k];
				float b = B[k * M + J];
				_c += a * b;
				printf("C = %f\n", _c);
			}
			C[I * M + J] = _c;
		}
	}
	*/

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}

__global__ void kernel_multiply(float* A, float* B, float* C, int M, int N, int K)
{
	//int I = blockIdx.y * blockDim.y + threadIdx.y;
	//int J = blockIdx.x * blockDim.x + threadIdx.x;

	//if ((I < N) && (J < M)) 
	for (int I = 0; I < N; I++)
	{
		for (int J = 0; J < M; J++)
		{
			float _c = 0;
			for (unsigned int k = 0; k < K; k++)
			{
				float a = A[I * K + k];
				float b = B[k * M + J];
				_c += a * b;
				printf("C = %f\n", _c);
			}
			C[I * M + J] = _c;
		}
	}
}
