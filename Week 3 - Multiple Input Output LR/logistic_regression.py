import torch
import torch.nn as nn
from torch.nn import Linear
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(1)
output_size = 1

######################################################
# Logistic Regression
######################################################

z = torch.arange(-100,100,0.1).view(-1,1)
sig = nn.Sigmoid()

yhat = sig(z)
yhat = torch.sigmoid(z)

x = torch.arange(-100,100,0.1).view(-1,1)

# Use sequential function to create model
model = nn.Sequential(nn.Linear(1,1), nn.Sigmoid())
yhat = model(x)

print(yhat)

######################################################
# Logistic Regression : Custom Module
######################################################

class logistic_regression(nn.Module):

    # Constructor
    def __init__(self, input_size, output_size):
        super(logistic_regression, self).__init__()

        self.linear = nn.Linear(input_size, output_size)
    
    # Predictor
    def forward(self,x):
        
        out = torch.sigmoid(self.linear(x))
        return out

x = torch.tensor([[1.0],[1.0]])
model = logistic_regression(x.shape[1],1)
yhat = model(x)
print(yhat)

class Data(Dataset):

    # Constructor
    def __init__(self):

        self.x = torch.zeros(20,1)
        self.x[:,0] = torch.arange(-1,1,0.1)
        
        self.y = torch.zeros(self.x.shape[0], 1)
        self.y[self.x[:, 0] > 0.2] = 1

        self.len = self.x.shape[0]
    
    # Getter
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # Get Length
    def __len__(self):
        return self.len

# Create Dataloader object
dataset = Data()
train_loader = DataLoader(dataset=dataset, batch_size=1)

# Build in cost function
def criterion_custom(y, yhat):
    out = -1 * torch.mean((y*torch.log(yhat)) + ( (1-y) * torch.log(1-yhat) ) )
    return out

criterion = nn.BCELoss()
criterion_rms = nn.MSELoss()

# Create model object
my_model = logistic_regression(dataset.x.shape[1],output_size)
# Create optimizer : get parameters from model
optimizer = optim.SGD(my_model.parameters(), lr = 0.1)

# Train Model
Cost = []
epochs = 5
def train_model():
    for epoch in range(epochs):

        for x, y in train_loader:
            
            # make the prediction
            yhat = my_model(x)
            # calculate the iteration loss
            # loss = criterion_rms(yhat, y)
            loss = criterion(yhat, y)

            # zero the gradients before running the backward pass
            optimizer.zero_grad()

            # backward pass: compute gradient of the loss with respect to all the learnable parameters
            loss.backward()

            # updata parameters
            optimizer.step()
        
        # training loss / Cost at nth epoch
        Yhat = my_model(dataset.x)
        cost = criterion(Yhat, dataset.y)
        Cost.append(cost.item())    
    
    labels = Yhat > 0.5
    return labels




print('Training ..')
labels = train_model()
print('Cost of Each Epoch',Cost)
print("The accuracy: ", torch.mean((labels == dataset.y.type(torch.ByteTensor)).type(torch.float)))

print('Ground Truth Vs Predicted Value :')
for i in range(len(dataset.y)):
    print(dataset.y[i], '\t', labels[i].float())
