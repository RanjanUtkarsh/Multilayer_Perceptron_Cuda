# -*- coding: utf-8 -*-

# importing libraries
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import os
import shutil
import sys
import torch.nn as nn
import torch.nn.functional as F
sys.path.append("../build/src/my_project/release")


num_workers = 0
batch_size = 20

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307), (0.3081))])

train_data = datasets.MNIST(root='data', train=True, download=True,  transform=transform)
test_data  = datasets.MNIST(root='data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
    num_workers=num_workers)


dataiter = iter(train_loader)

images, labels = next(dataiter)

images = images.numpy()


dataiter = iter(train_loader)
images, labels = next(dataiter)


## Define the NN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(512, 512)
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

model = Net()
temp = images.view(-1, 28 * 28)  # 20*28*28 / (28*28) 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
n_epochs = 10  # suggest training between 20-50 epochs



model.train() # prep model for training


for epoch in range(n_epochs):
    # monitor training loss
    train_loss = 0.0
    
    ###################
    # train the model #
    ###################
    for data, target in train_loader:
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0) # as loss is tensor, .item() needed to get the value
        

        
    # print training statistics 
    # calculate average loss over an epoch
    train_loss = train_loss/len(train_loader.dataset)

    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss)) ## printing command
    

print(f'Epoch: {epoch} \tTraining Loss: {train_loss}') ## printing command


model_scripted = torch.jit.script(model) # Export to TorchScript
model_scripted.save('model_scripted.pt') # Save

