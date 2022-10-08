import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F
import matplotlib.pylab as plt
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader 
torch.manual_seed(0)


######################################################
# DNN : Batch normalization
######################################################
# Define the Neural Network Model using Batch Normalization

class NetBatchNorm(nn.Module):
    
    # Constructor
    def __init__(self, in_size, n_hidden1, n_hidden2, out_size):
        super(NetBatchNorm, self).__init__()
        self.linear1 = nn.Linear(in_size, n_hidden1)
        self.linear2 = nn.Linear(n_hidden1, n_hidden2)
        self.linear3 = nn.Linear(n_hidden2, out_size)
        self.bn1 = nn.BatchNorm1d(n_hidden1)
        self.bn2 = nn.BatchNorm1d(n_hidden2)
        
    # Prediction
    def forward(self, x):
        x = self.bn1(torch.sigmoid(self.linear1(x)))
        x = self.bn2(torch.sigmoid(self.linear2(x)))
        x = self.linear3(x)
        return x
    
    # Activations, to analyze results 
    def activation(self, x):
        out = []
        z1 = self.bn1(self.linear1(x))
        out.append(z1.detach().numpy().reshape(-1))
        a1 = torch.sigmoid(z1)
        out.append(a1.detach().numpy().reshape(-1).reshape(-1))
        z2 = self.bn2(self.linear2(a1))
        out.append(z2.detach().numpy().reshape(-1))
        a2 = torch.sigmoid(z2)
        out.append(a2.detach().numpy().reshape(-1))
        return out

# Class Net for Neural Network Model

class Net(nn.Module):
    
    # Constructor
    def __init__(self, in_size, n_hidden1, n_hidden2, out_size):

        super(Net, self).__init__()
        self.linear1 = nn.Linear(in_size, n_hidden1)
        self.linear2 = nn.Linear(n_hidden1, n_hidden2)
        self.linear3 = nn.Linear(n_hidden2, out_size)
    
    # Prediction
    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        x = self.linear3(x)
        return x
    
    # Activations, to analyze results 
    def activation(self, x):
        out = []
        z1 = self.linear1(x)
        out.append(z1.detach().numpy().reshape(-1))
        a1 = torch.sigmoid(z1)
        out.append(a1.detach().numpy().reshape(-1).reshape(-1))
        z2 = self.linear2(a1)
        out.append(z2.detach().numpy().reshape(-1))
        a2 = torch.sigmoid(z2)
        out.append(a2.detach().numpy().reshape(-1))
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

# Define the function to train model

def train(model, criterion, train_loader, validation_loader, optimizer, epochs=10):
    i = 0
    useful_stuff = {'training_loss':[], 'validation_accuracy':[]}  

    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            z = model(x.view(-1, 28 * 28))
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            useful_stuff['training_loss'].append(loss.data.item())
            
        correct = 0
        for x, y in validation_loader:
            model.eval()
            yhat = model(x.view(-1, 28 * 28))
            _, label = torch.max(yhat, 1)
            correct += (label == y).sum().item()
            
        accuracy = 100 * (correct / len(val_dataset))
        useful_stuff['validation_accuracy'].append(accuracy)
    
    return useful_stuff


input_size = train_dataset[0][0].shape[1] * train_dataset[0][0].shape[1]
output_size = 10
H1_neurons = 50 
H2_neurons = 10 

# Create model, optimizer and train the model

model_norm  = NetBatchNorm(input_size, H1_neurons, H2_neurons, output_size)
optimizer_norm = torch.optim.Adam(model_norm.parameters(), lr = 0.1)

model  = Net(input_size, H1_neurons, H2_neurons, output_size)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)

criterion = nn.CrossEntropyLoss()
training_results_Norm=train(model_norm , criterion, train_loader, val_loader, optimizer_norm, epochs=5)
training_results=train(model , criterion, train_loader, val_loader, optimizer, epochs=5)

plt.figure()
model.eval()
model_norm.eval()
out=model.activation(val_dataset[0][0].reshape(-1,28*28))
plt.hist(out[2],label='model with no batch normalization' )
out_norm=model_norm.activation(val_dataset[0][0].reshape(-1,28*28))
plt.hist(out_norm[2],label='model with normalization')
plt.xlabel("activation ")
plt.legend()
plt.draw()

# Plot the diagram to show the loss
plt.figure()
plt.plot(training_results['training_loss'], label='No Batch Normalization')
plt.plot(training_results_Norm['training_loss'], label='Batch Normalization')
plt.ylabel('Cost')
plt.xlabel('iterations ')
plt.legend()   
plt.draw()

# Plot the diagram to show the accuracy
plt.figure()
plt.plot(training_results['validation_accuracy'],label='No Batch Normalization')
plt.plot(training_results_Norm['validation_accuracy'],label='Batch Normalization')
plt.ylabel('validation accuracy')
plt.xlabel('epochs ')   
plt.legend()

plt.draw()
plt.show()

