import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(1)

w = torch.tensor(-15.0, requires_grad= True)
b = torch.tensor(-10.0, requires_grad= True)
X = torch.arange(-3,3,0.1).view(-1,1)
f = -3 * X
Y = f + 0.1 *torch.rand(X.size())
lr = 0.1

epochs = 5

class Data(Dataset):
    
    def __init__(self, length = 100, transform = None):
        
        self.X = torch.arange(-3,3,0.1).view(-1,1)
        f = -3 * self.X
        self.Y = f + 0.1 *torch.rand(X.size())
        self.transform = transform
        self.len = len(self.X)
    
    def __getitem__(self, index):
        
        sample = self.X[index] , self.Y[index]
        return sample
        
    def __len__(self):

        return self.len

class LR(nn.Module):
    
    def __init__(self, input_size, output_size):
        super(LR, self).__init__()
        
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        
        out = self.linear(x)
        return out

criterion = nn.MSELoss()
dataset = Data()
trainloader = DataLoader(dataset=dataset, batch_size=1)

model = LR(1,1)
optimizer = optim.SGD(model.parameters(), lr = 0.01)

# optimizer.state_dict()
Loss = []

def train_model(epochs):
    
    for epoch in range (epochs):
        
        Yhat = model(X)
        loss = criterion(Yhat, Y)
        Loss.append(loss.tolist())

        for x,y in trainloader:

            # make the prediction as we learned in the last lab
            yhat = model(x)
            
            # calculate the iteration
            loss = criterion(yhat, y)

            # zero the gradients before running the backward pass               
            optimizer.zero_grad()

            # backward pass: compute gradient of the loss with respect to all the learnable parameters
            loss.backward()

            # updata parameters
            optimizer.step()
            

train_model(epochs)
print('Training Batch Gradient Descent .... ')
print('Cost of each epoch == Loss of Each epoch')
print('\nLoss /Cost elements  == epochs :',Loss)
print('Number of elements in epoch / batch :',len(X))


######################################################
# Mini Batch Gradient Descent : Another Way
######################################################

dataset = Data()
trainloader = DataLoader(dataset = dataset, batch_size = 5)

w = torch.tensor(-15.0, requires_grad = True)
b = torch.tensor(-10.0, requires_grad = True)
LOSS_MINI = []
lr = 0.1

def forward(x):
    f = w*x +b
    return f

def criterion(y, yhat):
    # loss = 1/len(y) * torch.sum((y - yhat)**2)
    # Define the MSE Loss function
    loss = torch.mean((yhat - y) ** 2)
    return loss

def train_model_Mini(epochs):

    for epoch in range(epochs):
        Yhat = forward(X)
        loss = criterion(Yhat,Y)
        #LOSS_MINI.append(criterion(forward(X),Y).tolist())
        LOSS_MINI.append(loss.tolist())

        for x, y in trainloader:
            yhat = forward(x)
            loss = criterion(yhat, y)
            loss.backward()
            w.data = w.data - lr * w.grad.data
            b.data = b.data - lr * b.grad.data
            w.grad.data.zero_()
            b.grad.data.zero_()
            
train_model_Mini(5)
print ('\nTraining Mini Match different wway ....')
print("\nCost of Each Epoch :", LOSS_MINI)