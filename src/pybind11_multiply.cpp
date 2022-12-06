/*************************************************************************
/* ECE 277: GPU Programmming 2020
/* Author and Instructer: Cheolhong An
/* Copyright 2020
/* University of California, San Diego
/*************************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

extern void cu_multiply(float* A, float* B, float* C, int M, int N, int K);

namespace py = pybind11;


py::array_t<float> mm_wrapper(py::array_t<float> a1, py::array_t<float> a2 /*, int dimx, int dimy*/)
{
	auto buf1 = a1.request();
	auto buf2 = a2.request();

	// A = NxK, B = KxM matrix
	int N = a1.shape()[0];
	int K = a1.shape()[1];
	int M = a2.shape()[1];
	printf("M=%d, N=%d\n", M, N);

	auto result = py::array(py::buffer_info(
		nullptr,            /* Pointer to data (nullptr -> ask NumPy to allocate!) */
		sizeof(float),     /* Size of one item */
		py::format_descriptor<float>::value, /* Buffer format */
		buf1.ndim,          /* How many dimensions? */
		{ N, M },  /* Number of elements for each dimension */
		{ sizeof(int) * M, sizeof(int) }  /* Strides for each dimension */
	));

	auto buf3 = result.request();

	float* A = (float*)buf1.ptr;
	float* B = (float*)buf2.ptr;
	float* C = (float*)buf3.ptr;

	cu_multiply(A, B, C, M, N, K);
	

	return result;
}



PYBIND11_MODULE(cu_multiply_matrix, m) {
	m.def("multiply", &mm_wrapper, "Linear layer");

#ifdef VERSION_INFO
	m.attr("__version__") = VERSION_INFO;
#else
	m.attr("__version__") = "dev";
#endif
}