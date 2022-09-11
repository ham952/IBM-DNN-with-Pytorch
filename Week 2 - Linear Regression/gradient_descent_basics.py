import torch
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(1)

w = torch.tensor(-15.0, requires_grad= True)
b = torch.tensor(-10.0, requires_grad= True)
X = torch.arange(-3,3,0.1).view(-1,1)
f = -3 * X
Y = f + 0.1 *torch.rand(X.size())
lr = 0.1

def forward(x):
    f = w * x + b
    return f

def criterion(y, yhat):
    # loss = 1/len(y) * torch.sum((y - yhat)**2)
    # Define the MSE Loss function
    loss = torch.mean((yhat - y) ** 2)
    return loss

######################################################
# Batch Gradient Descent
######################################################
Loss_BGD = []

def train_model(iter):
    for epoch in range (iter):

        # make the prediction as we learned in the last lab
        yhat = forward(X)
        
        # calculate the iteration
        loss = criterion(yhat,Y)
            
        # store the loss into list
        Loss_BGD.append(loss.item())
        
        # backward pass: compute gradient of the loss with respect to all the learnable parameters
        loss.backward()
        
        # updata parameters
        w.data = w.data - lr * w.grad.data
        b.data = b.data - lr * b.grad.data
        
        # zero the gradients before running the backward pass
        w.grad.data.zero_()
        b.grad.data.zero_()

train_model(5)
print('Training Batch Gradient Descent .... ')
print('Cost of each epoch == Loss of Each epoch')
print('\nLoss /Cost elements  == epochs :',Loss_BGD)
print('Number of elements in epoch / batch :',len(X))

######################################################
# SGD : Gradient descent with one sample at a time
######################################################

w = torch.tensor(-15.0, requires_grad = True)
b = torch.tensor(-10.0, requires_grad = True)

print('\nTraining Stochastic Gradient Descent .... ')
Loss = []
Cost = []
def train_model(iter):
    for epoch in range (iter):
        total = 0
        # Taking one sample at a time
        for x,y in zip(X,Y):
        
            # make the prediction as we learned in the last lab
            yhat = forward(x)
            
            # calculate the iteration
            loss = criterion(yhat,y)
                
            # store the loss into list
            Loss.append(loss.item())
            
            # backward pass: compute gradient of the loss with respect to all the learnable parameters
            loss.backward()
            
            # updata parameters
            w.data = w.data - lr * w.grad.data
            b.data = b.data - lr * b.grad.data
            
            # zero the gradients before running the backward pass
            w.grad.data.zero_()
            b.grad.data.zero_()

            total += loss.item()
        
        Cost.append(total)

train_model(5)
print('\nLoss elements  = len of X * epochs :',len(Loss))
print("\nCost of Each Epoch :", Cost)

######################################################
# SGD with Data Loader
######################################################
w = torch.tensor(-15.0, requires_grad = True)
b = torch.tensor(-10.0, requires_grad = True)
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

dataset = Data()

# Batch size control number of elements
# if batch_size = 1 : SGD with one element at a time
# if batch_size > 1 : Mini batch Gradient Descent
trainloader = DataLoader(dataset= dataset, batch_size=1)

print('\nTraining SGD with DataLoader.......')
Loss = []
Cost = []
def train_model(epochs):

    # Loop
    for epoch in range (epochs):
        total = 0
        # Taking one sample / batch size sample at a time
        for sample in trainloader:
            
            x, y = sample
            
            # make the prediction 
            yhat = forward(x)
            
            # calculate the loss
            loss = criterion(yhat,y)
                
            # store the loss into list
            Loss.append(loss.item())
            
            # backward pass: compute gradient of the loss with respect to all the learnable parameters
            loss.backward()
            
            # updata parameters slope
            w.data = w.data - lr * w.grad.data
            b.data = b.data - lr * b.grad.data
            
            # zero the gradients before running the backward pass
            # Clear gradients 
            w.grad.data.zero_()
            b.grad.data.zero_()

            total += loss.item()
        
        Cost.append(total)

train_model(5)
print('\nLoss elements  = len of X * epochs :',len(Loss))
print("\nCost of Each Epoch :", Cost)

######################################################
# Mini Batch Gradient Descent : Another Way
######################################################

dataset = Data()
trainloader = DataLoader(dataset = dataset, batch_size = 5)

w = torch.tensor(-15.0, requires_grad = True)
b = torch.tensor(-10.0, requires_grad = True)
COST_MINI = []
LOSS_MINI = []
lr = 0.1

def train_model_Mini(epochs):

    for epoch in range(epochs):
        X = dataset.X
        Y = dataset.Y
        Yhat = forward(X)
        loss = criterion(Yhat,Y)
        #LOSS_MINI.append(criterion(forward(X),Y).tolist())
        COST_MINI.append(loss.tolist())

        total = 0
        for x, y in trainloader:
            yhat = forward(x)
            loss = criterion(yhat, y)
            loss.backward()
            w.data = w.data - lr * w.grad.data
            b.data = b.data - lr * b.grad.data
            w.grad.data.zero_()
            b.grad.data.zero_()

            total+=loss.item()
        
        LOSS_MINI.append(total)
            
train_model_Mini(5)
print ('\nTraining Mini Match different wway ....')
print("\nCost of Each Epoch :", COST_MINI)
print("\nLoss wrt batch wise training :", LOSS_MINI)