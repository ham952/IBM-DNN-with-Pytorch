import torch
import torch.nn as nn
from torch.nn import Linear
from torch import nn, optim, sigmoid
from torch.utils.data import Dataset, DataLoader 
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import os
import numpy as np

torch.manual_seed(1)
'''
######################################################
# NN in one dimesnion : Single Layer 
######################################################

class NN(nn.Module):

    # hidden_layer = number of neurons
    # Constructor
    def __init__(self, input_size, hidden_layer, output_size):
        super(NN, self).__init__()

        self.linear1 = nn.Linear(input_size, hidden_layer)
        self.linear2 = nn.Linear(hidden_layer, output_size)
    
    # Predictor
    def forward(self,x):
        
        x = sigmoid(self.linear1(x))
        out = sigmoid(self.linear2(x))

        return out

class Data(Dataset):
    def __init__(self):
        
        self.x = torch.linspace(-20,20,100).view(-1,1)
        self.y = torch.zeros(self.x.shape[0])

        self.y[ (self.x[:,0] > -10) & (self.x[:,0] < -5)] = 1
        self.y[ (self.x[:,0] > 5) & (self.x[:,0] < 10)] = 1
        self.y = self.y.view(-1,1)

        self.len = self.x.shape[0]

    def __getitem__(self, index):
        
        return self.x[index], self.y[index]
    
    def __len__(self):
        
        return self.len

model = NN(1,3,1)
x = torch.tensor([[0.0],[0.1],[10.0],[-10.0]])
yhat = model(x)
print(model.state_dict())
print(yhat)

yhat = yhat > 0.5
print(yhat.float())


# Training 
X = torch.arange(-20,20,1).view(-1,1).type(torch.FloatTensor)
Y = torch.zeros(X.shape[0])
Y[(X[:, 0] > -4) & (X[:, 0] < 4)] = 1.0

epochs = 1000
criterion = nn.BCELoss()

def criterion_cross(outputs, labels):
    out = -1 * torch.mean(labels * torch.log(outputs) + (1 - labels) * torch.log(1 - outputs))
    return out

def train(X,Y, model, optimizer, criterion, epochs=1000):
    cost = []
    total = 0

    for epoch in range(epochs):
        total = 0
        for x,y in zip(X,Y):
            yhat = model(x)
            loss = criterion(yhat, y)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            # cummulative loss
            total += loss.item()
        cost.append(total)
    
    return cost

def train_dataset(data, model, optimizer, criterion, epochs=1000):
    cost = []
    total = 0

    for epoch in range(epochs):
        total = 0
        for x,y in data:
            yhat = model(x)
            loss = criterion(yhat, y)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            # cummulative loss
            total += loss.item()
        cost.append(total)
    
    return cost

model = NN(1,6,1)
optimizer = optim.SGD(model.parameters(), lr= 0.1)
# cost_cross = train(X, Y, model, optimizer, criterion_cross, epochs=500)

data = Data()
train_loader = DataLoader(dataset=data, batch_size=1)
cost_dataset = train_dataset(train_loader, model, optimizer, criterion_cross, epochs=500)
# print(cost)

# Yhat = model(X)
Yhat = model(data.x)
Yhat = (Yhat > 0.5).float()

#accuracy = ( (Yhat==Y.view(-1,1)).sum().item() ) / len(Y)
accuracy = ( (Yhat==data.y.view(-1,1)).sum().item() ) / len(data.y)
print('Accuracy of model : ',accuracy)

# print('Ground Truth Vs Predicted Value :')
# for i in range(10):
#     print(Y[i], '\t', Yhat[i])

###################################################################
# NN in one dimesnion (1 class) : Single Layer : Multiple Input
##################################################################

# Define the class XOR_Data

class XOR_Data(Dataset):
    
    # Constructor
    def __init__(self, N_s=100):
        self.x = torch.zeros((N_s, 2))
        self.y = torch.zeros((N_s, 1))
        for i in range(N_s // 4):
            self.x[i, :] = torch.Tensor([0.0, 0.0]) 
            self.y[i, 0] = torch.Tensor([0.0])

            self.x[i + N_s // 4, :] = torch.Tensor([0.0, 1.0])
            self.y[i + N_s // 4, 0] = torch.Tensor([1.0])
    
            self.x[i + N_s // 2, :] = torch.Tensor([1.0, 0.0])
            self.y[i + N_s // 2, 0] = torch.Tensor([1.0])
    
            self.x[i + 3 * N_s // 4, :] = torch.Tensor([1.0, 1.0])
            self.y[i + 3 * N_s // 4, 0] = torch.Tensor([0.0])

            self.x = self.x + 0.01 * torch.randn((N_s, 2))
        self.len = N_s

    # Getter
    def __getitem__(self, index):    
        return self.x[index],self.y[index]
    
    # Get Length
    def __len__(self):
        return self.len

class NN(nn.Module):

    #Constructor
    def __init__(self, D_in, H, D_out):
        super(NN, self).__init__()

        # Hidden Layer
        self.linear1 = nn.Linear(D_in, H)
        
        # Output Layer
        self.linear2 = nn.Linear(H, D_out)
    
    # Prediction

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        out = torch.sigmoid(self.linear2(x))
    
        return out


def train(dataset, model, criterion, train_loader, optimizer, epochs = 500):
    Cost = []
    Accuracy = []

    for epoch in range(epochs):
        total = 0

        for x,y in train_loader:
            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat,y)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            # cummulative loss
            total += loss.item()

        Cost.append(total)
        # Yhat = model(dataset.x)
        # Yhat = (Yhat > 0.5).float()
        # Yhat = Yhat[:,0]
        # accuracy = ( (Yhat==dataset.y.view(-1)).sum()) / len(dataset.y)
        accuracy = torch.mean(( dataset.y.view(-1) == (model(dataset.x)> 0.5).float()[:,0]).float() )
        Accuracy.append(accuracy.item())
    
    return Cost, Accuracy

# def accuracy(model, data_set):
#     return np.mean(data_set.y.view(-1).numpy() == (model(data_set.x)[:, 0] > 0.5).numpy())

data = XOR_Data()
learning_rate = 0.1
criterion = nn.BCELoss()
model = NN(data.x.shape[-1], 2, 1)

optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate)
train_loader = DataLoader(data, batch_size=1)

cost, accuracy = train(data, model, criterion, train_loader, optimizer)

print('#'*10)
print("Cost .. \n", cost)
print('Accuracy ... \n', accuracy)

from matplotlib import pyplot as plt

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.plot(cost, color=color)
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

'''
###################################################################
# Multi Class : Single Layer : MNIST Dataset
##################################################################
# Build the model with Relu function

class NNRelu(nn.Module):

    # Constructor
    def __init__(self, D_in, H, D_out):
        super(NNRelu, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    # Prediction
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class NN(nn.Module):

    #Constructor
    def __init__(self, D_in, H, D_out):
        super(NN, self).__init__()

        # Hidden Layer
        self.linear1 = nn.Linear(D_in, H)
        
        # Output Layer
        self.linear2 = nn.Linear(H, D_out)
    
    # Prediction

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        out = self.linear2(x)
    
        return out

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
hlayer_neurons = 5 
model = NNRelu(input_size, hlayer_neurons, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)

epochs = 5

def train(model, criterion, val_loader, train_loader, optimizer, epochs = 5):
    Loss = []
    Accuracy = []

    for epoch in range(epochs):
        total = 0

        for x,y in train_loader:
            optimizer.zero_grad()
            yhat = model(x.view(-1,input_size))
            loss = criterion(yhat,y)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            # cummulative loss
            total += loss.item()

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

loss, accuracy = train(model, criterion,val_loader, train_loader, optimizer, epochs)

from matplotlib import pyplot as plt

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
 