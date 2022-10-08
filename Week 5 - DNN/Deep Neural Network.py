from statistics import mode
import torch
import torch.nn as nn
from torch.nn import Linear
from torch import nn, optim, sigmoid
from torch.utils.data import Dataset, DataLoader 
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import os
import numpy as np
from matplotlib import pyplot as plt

torch.manual_seed(1)

######################################################
# DNN : Multiple Layer (Hidden) with Dropout
######################################################

class NNRelu(nn.Module):

    # Constructor
    def __init__(self, D_in, H1, H2, D_out, p=0):
        super(NNRelu, self).__init__()
        self.Dropout = nn.Dropout(p=p)
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)

    # Prediction
    def forward(self, x):
        # x = torch.relu(self.linear1(x))
        # x = self.Dropout(x)
        # x = torch.relu(self.linear2(x))
        # x = self.Dropout(x)
        x = torch.relu(self.Dropout(self.linear1(x)))
        x = torch.relu(self.Dropout(self.linear2(x)))
        x = self.linear3(x)
        return x
    
# # Sequential
# model_Seq = torch.nn.Sequential(
#     torch.nn.Linear(D_in, H1), 
#     torch.nn.Dropout(0.5),
#     torch.nn.Sigmoid(),
#     torch.nn.Linear(H1, H2),
#     torch.nn.Dropout(0.5), 
#     torch.nn.Sigmoid(),
#     torch.nn.Linear(H2, D_out)
# )


script_dir = os.path.dirname(__file__)
base = os.path.join(script_dir,"..","Dataset")

train_dataset = dsets.MNIST(root=base, train=True, download= True, transform= transforms.ToTensor())
val_dataset = dsets.MNIST(root=base, train=False, download= True, transform= transforms.ToTensor())

train_loader = DataLoader(dataset = train_dataset, batch_size = 100)
val_loader = DataLoader(dataset = val_dataset, batch_size = len(val_dataset))

print('Elements in Training dataset',len(train_dataset))
print('Shape of Each element',train_dataset[0][0].shape)    # torch.size([1,28,28])
print('Elements in Validation dataset',len(val_dataset))

input_size = train_dataset[0][0].shape[1] * train_dataset[0][0].shape[1]
output_size = 10
H1_neurons = 50 
H2_neurons = 10 
model = NNRelu(input_size, H1_neurons, H2_neurons, output_size, p=0.2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.6)

epochs = 5

def train(model, criterion, val_loader, train_loader, optimizer, epochs = 5):
    Loss = []
    Accuracy = []
    
    for epoch in range(epochs):
        total = 0
        model.train()
        for x,y in train_loader:
            optimizer.zero_grad()
            yhat = model(x.view(-1,input_size))
            loss = criterion(yhat,y)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            # cummulative loss
            total += loss.item()

        model.eval()
        correct = 0
        for X,Y in val_loader:            
            
            z = model(X.view(-1,input_size))
            _, yhat = z.max(1)
            # _, yhat = torch.max(z.data,1)

            correct += (yhat==Y).sum().item()
        
        accuracy = correct / len(Y)
        Accuracy.append(accuracy)
        Loss.append(loss.item())
    
    return Loss, Accuracy

def print_model_parameters(model):
    count = 0
    for ele in model.state_dict():
        count += 1
        if count % 2 != 0:
            print ("The following are the parameters for the layer ", count // 2 + 1)
        if ele.find("bias") != -1:
            print("The size of bias: ", model.state_dict()[ele].size())
        else:
            print("The size of weights: ", model.state_dict()[ele].size())

# print_model_parameters(model)
print(model.parameters)

loss, accuracy = train(model, criterion,val_loader, train_loader, optimizer, epochs)

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.plot(loss, color=color)
ax1.set_xlabel('epoch', color=color)
ax1.set_ylabel('total loss', color=color)
ax1.tick_params(axis='y', color=color)

ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('accuracy', color=color)  # we already handled the x-label with ax1
ax2.plot(accuracy, color=color)
ax2.tick_params(axis='y', color=color)
fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.show()

######################################################
# DNN using ModuleList : Dropout + Initialization(s)
######################################################


script_dir = os.path.dirname(__file__)
base = os.path.join(script_dir,"..","Dataset")

train_dataset = dsets.MNIST(root=base, train=True, download= True, transform= transforms.ToTensor())
val_dataset = dsets.MNIST(root=base, train=False, download= True, transform= transforms.ToTensor())

train_loader = DataLoader(dataset = train_dataset, batch_size = 100)
val_loader = DataLoader(dataset = val_dataset, batch_size = len(val_dataset))

print('Elements in Training dataset',len(train_dataset))
print('Shape of Each element',train_dataset[0][0].shape)    # torch.size([1,28,28])
print('Elements in Validation dataset',len(val_dataset))

input_size = train_dataset[0][0].shape[1] * train_dataset[0][0].shape[1]
output_size = 10
H1_neurons = 50 
H2_neurons = 10 
epochs = 30

#DEFAULT Initializiation
class Net(nn.Module):
    def __init__(self, Layers, p=0):
        super(Net, self).__init__()

        self.hidden = nn.ModuleList()
        self.Droupout = nn.Dropout(p=p)

        for in_size, out_size in zip(Layers, Layers[1:]):
            self.hidden.append(nn.Linear(in_size,out_size))
        
    def forward(self,x):

        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):

            if l < L-1:
                x = torch.relu(self.Droupout(linear_transform(x)))
            else:
                out = linear_transform(x)

        return out 

# Define the neural network with Uniform initialization

class Net_Uniform(nn.Module):
    
    # Constructor
    def __init__(self, Layers):
        super(Net_Uniform, self).__init__()
        self.hidden = nn.ModuleList()

        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            linear.weight.data.uniform_(0, 1)
            self.hidden.append(linear)
    
    # Prediction
    def forward(self, x):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                x = torch.relu(linear_transform(x))
            else:
                x = linear_transform(x)
        return x

# Define the neural network with Xavier initialization

class Net_Xavier(nn.Module):
    
    # Constructor
    def __init__(self, Layers):
        super(Net_Xavier, self).__init__()
        self.hidden = nn.ModuleList()

        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            torch.nn.init.xavier_uniform_(linear.weight)
            self.hidden.append(linear)
    
    # Prediction
    def forward(self, x):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                x = torch.tanh(linear_transform(x))
            else:
                x = linear_transform(x)
        return x

# Define the class for neural network model with He Initialization

class Net_He(nn.Module):
    
    # Constructor
    def __init__(self, Layers):
        super(Net_He, self).__init__()
        self.hidden = nn.ModuleList()

        for input_size, output_size in zip(Layers, Layers[1:]):
            linear = nn.Linear(input_size, output_size)
            torch.nn.init.kaiming_uniform_(linear.weight, nonlinearity='relu')
            self.hidden.append(linear)

    # Prediction
    def forward(self, x):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                x = torch.relu(linear_transform(x))
            else:
                x = linear_transform(x)
        return x

# function to Train the model

def train(model, criterion, train_loader, validation_loader, optimizer, epochs = 100):
    i = 0
    loss_accuracy = {'training_loss':[], 'validation_accuracy':[]}  
    
    for epoch in range(epochs):
        for i,(x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            z = model(x.view(-1, 28 * 28))
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            loss_accuracy['training_loss'].append(loss.data.item())
            
        correct = 0
        for x, y in validation_loader:
            yhat = model(x.view(-1, 28 * 28))
            _, label = torch.max(yhat, 1)
            correct += (label==y).sum().item()
        accuracy = 100 * (correct / len(val_dataset))
        loss_accuracy['validation_accuracy'].append(accuracy)
        
    return loss_accuracy

# Layers = [input_size, H1_neurons, H2_neurons, output_size]
Layers = [input_size, 100, 10, 100, 10, 100, output_size]
# Train the model with 2 hidden layers with 20 neurons

criterion = nn.CrossEntropyLoss()
# Train the model with default initialization

model = Net(Layers)
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
training_results = train(model, criterion, train_loader, val_loader, optimizer, epochs=epochs)

# Train the model with Xavier initialization

model_Xavier = Net_Xavier(Layers)
optimizer = torch.optim.SGD(model_Xavier.parameters(), lr=learning_rate)
training_results_Xavier = train(model_Xavier, criterion, train_loader, val_loader, optimizer, epochs=epochs)

# Train the model with Uniform initialization

model_Uniform = Net_Uniform(Layers)
optimizer = torch.optim.SGD(model_Uniform.parameters(), lr=learning_rate)
training_results_Uniform = train(model_Uniform, criterion, train_loader, val_loader, optimizer, epochs=epochs)

# Train the model with Uniform initialization

model_He = Net_He(Layers)
optimizer = torch.optim.SGD(model_He.parameters(), lr=learning_rate)
training_results_He = train(model_He, criterion, train_loader, val_loader, optimizer, epochs=epochs)

# Plot the loss
plt.figure()
plt.plot(training_results_Xavier['training_loss'], label='Xavier')
plt.plot(training_results['training_loss'], label='Default')
plt.plot(training_results_Uniform['training_loss'], label='Uniform')
plt.plot(training_results_He['training_loss'], label='He')
plt.ylabel('loss')
plt.xlabel('iteration ')  
plt.title('training loss iterations')
plt.legend()
plt.savefig('Training Loss.png')
plt.draw()

# Plot the accuracy
plt.figure()
plt.plot(training_results_Xavier['validation_accuracy'], label='Xavier')
plt.plot(training_results['validation_accuracy'], label='Default')
plt.plot(training_results_Uniform['validation_accuracy'], label='Uniform')
plt.plot(training_results_He['validation_accuracy'], label='He')  
plt.ylabel('validation accuracy')
plt.xlabel('epochs')   
plt.legend()
plt.savefig('Validation Accuracy.png')
plt.draw()

plt.show()