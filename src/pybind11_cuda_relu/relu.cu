/*************************************************************************
/* ECE 277: GPU Programmming 2020
/* Author and Instructer: Cheolhong An
/* Copyright 2020
/* University of California, San Diego
/*************************************************************************/

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

__global__ void kernel_relu(float* A, float* O, int M, int N);

void cu_relu(float* A, float* O, int M, int N)
{
	float* d_a, * d_o;

	dim3 blk;
	blk.x = 16; blk.y = 16; blk.z = 1;

	dim3 grid;
	grid.x = (M + blk.x - 1) / blk.x;
	grid.y = (N + blk.y - 1) / blk.y;
	grid.z = 1;

	int size = sizeof(unsigned int) * M * N;

	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_o, size);

	cudaMemcpy(d_a, A, size, cudaMemcpyHostToDevice);

	kernel_relu << < grid, blk >> > (d_a, d_o, M, N);

	cudaMemcpy(O, d_o, size, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_o);
}

__global__ void kernel_relu(float* A, float* O, int M, int N)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < M * N) {
		O[index] = fmaxf(A[index], 0);
	}
}
