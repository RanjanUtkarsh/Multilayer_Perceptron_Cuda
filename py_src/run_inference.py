# -*- coding: utf-8 -*-


import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import os
import shutil
import sys
sys.path.append("../build/src/my_project/release")
import run_pybind

num_workers = 0
batch_size = 10
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307), (0.3081))])
test_data  = datasets.MNIST(root='data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
    num_workers=num_workers)


model = torch.jit.load('model_scripted.pt')


model.eval()  ## prepare for evaluation


test_loss = 0
correct = 0

with torch.no_grad():  # disabling the gradient calculation
    for data, target in test_loader:     ### using "test_loader" for evaluation
        output = model(data)
        #print(run_pybind.my_inference(data))
        pred = output.argmax(dim=1, keepdim=True)  
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)

print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



print(output)  # Not clean one hot encoding, # no negative numbers
#print(output.size())


print(output.size())
pred = output.argmax(dim=1, keepdim=True)
print(pred)


print(pred.size())
print(target.size())
print(    target.view_as(pred).size()   )


print(     pred.eq(   target.view_as(pred)  )             )
print(     pred.eq(target.view_as(pred)).sum().item()     )
