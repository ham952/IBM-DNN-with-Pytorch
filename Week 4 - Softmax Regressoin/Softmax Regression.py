import torch
import torch.nn as nn
from torch.nn import Linear
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader 
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import os

torch.manual_seed(1)
output_size = 4

######################################################
# Logistic Regression : Custom Module
######################################################

class Softmax(nn.Module):

    # Constructor
    def __init__(self, input_size, output_size):
        super(Softmax, self).__init__()

        self.linear = nn.Linear(input_size, output_size)
    
    # Predictor
    def forward(self,x):
        
        out = self.linear(x)
        return out

x = torch.tensor([[1.0, 2.0]])
x = torch.tensor([[1.0, -1.0],[0.0,-2.0],[4.0,3.0]])
model = Softmax(x.shape[1],output_size)
z = model(x)
print(z)

_,yhat = z.max(1)   # max with respect to column i.e output wrt each sample x

print(yhat)

######################################################
# Logistic Regression : MNIST Dataset
######################################################

script_dir = os.path.dirname(__file__)
base = os.path.join(script_dir,"..","Dataset")

train_dataset = dsets.MNIST(root=base, train=True, download= True, transform= transforms.ToTensor())
val_dataset = dsets.MNIST(root=base, train=False, download= True, transform= transforms.ToTensor())

train_loader = DataLoader(dataset = train_dataset, batch_size = 100)
val_loader = DataLoader(dataset = val_dataset, batch_size = len(val_dataset))

output_size = 10

class Softmax(nn.Module):

    # Constructor
    def __init__(self, input_size, output_size):
        super(Softmax, self).__init__()

        self.linear = nn.Linear(input_size, output_size)
    
    # Predictor
    def forward(self,x):
        
        out = self.linear(x)
        return out

print('Elements in Training dataset',len(train_dataset))
print('Shape of Each element',train_dataset[0][0].shape)    # torch.size([1,28,28])
print('Elements in Validation dataset',len(val_dataset))

input_size = train_dataset[0][0].shape[1] * train_dataset[0][0].shape[1]
model = Softmax(input_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)

epochs = 1
Accuracy = []
Loss = []

def train_model():

    for epoch in range(epochs):
        
        for x, y in train_loader:
            
            yhat = model(x.view(-1, input_size))
            loss = criterion(yhat, y)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        correct = 0
        for X,Y in val_loader:            
            
            z = model(X.view(-1,input_size))
            _, yhat = z.max(1)
            # _, yhat = torch.max(z.data,1)

            correct += (yhat==Y).sum().item()
        
        accuracy = correct / len(Y)
        Accuracy.append(accuracy)
        Loss.append(loss.item())

train_model()
print(Accuracy)
print(Loss)

# Plot the classified samples
Softmax_fn=nn.Softmax(dim=-1)
count = 0
for x, y in val_dataset:
    z = model(x.reshape(-1, 28 * 28))
    _, yhat = torch.max(z, 1)
    # if yhat != y:
    if yhat == y:
        # show_data((x, y))
        # plt.show()
        print("yhat:", yhat)
        print("probability of class wrt Softmax function : ", torch.max(Softmax_fn(z)).item())
        print("probability of class wrt Sigmoid Function : ", torch.max(torch.sigmoid(z)).item())
 
        # Verification
        _, ind_sof = torch.max(Softmax_fn(z),1)
        _, ind_sig = torch.max(torch.sigmoid(z),1)

        if (ind_sof != yhat).item() or (ind_sig != yhat).item():
            print('Error')
        count += 1
    if count >= 5:
        break  