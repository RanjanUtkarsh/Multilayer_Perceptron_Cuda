/*************************************************************************
/* ECE 277: GPU Programmming 2020
/* Author and Instructer: Cheolhong An
/* Copyright 2020
/* University of California, San Diego
/*************************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

extern void cu_relu(float* A, float* O, int M, int N);

namespace py = pybind11;


py::array_t<float> relu_wrapper(py::array_t<float> a1)
{
	auto buf1 = a1.request();

	// NxM matrix
	int N = a1.shape()[0];
	int M = a1.shape()[1];
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
	float* O = (float*)buf3.ptr;

	cu_relu(A, O, M, N);

	return result;
}



PYBIND11_MODULE(cu_relu_lyr, m) {
	m.def("relu", &relu_wrapper, "ReLU layer");

#ifdef VERSION_INFO
	m.attr("__version__") = VERSION_INFO;
#else
	m.attr("__version__") = "dev";
#endif
}