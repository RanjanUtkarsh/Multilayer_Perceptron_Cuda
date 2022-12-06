import sys
sys.path.append("../build/src/pybind11_cpp_examples/release")
sys.path.append("../build/src/pybind11_cuda_examples/release")
# import cpp_add
# import cpp_export
# import cpp_matrix_add
#import cu_matrix_add
sys.path.append("../build/src/pybind11_cuda_relu/release")
import cu_relu_lyr
import tensorflow as tf
import numpy as np
import torch



A = np.random.randint(100, size=(3,3))
B = np.random.randint(-2, 2, (10, 2))
B = np.random.rand(3,2)
print(A)
print(B)
O = cu_relu_lyr.relu(A)
O2 = cu_relu_lyr.relu(B)
print(O)
print(O2)

def my_inference(data):
    model = torch.jit.load('model_scripted.pt')
    weight_1 = model.fc1.weight
    weight_2 = model.fc2.weight
    weight_3 = model.fc3.weight
    
    data_array = data.numpy()
    print(data_array)
    
    #data_l1 = cu_relu.relu(cu_my_project.mmpy(data_array, weight_1, (28*28), 512))
    #data_l2 = cu_my_project.relu(cu_my_project.mmpy(data_l1, weight_2, 512, 512))
    #data_l3 = cu_my_project.relu(cu_my_project.mmpy(data_l2, weight_3, 512, 10))
    
    #result = max(data_l3)