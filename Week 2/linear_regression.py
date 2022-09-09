from cmath import cos
import torch
from torch.nn import Linear
import torch.nn as nn

torch.manual_seed(1)
######################################################
# Linear Regression using functions
######################################################

w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(-1.0, requires_grad=True)

def forward(x):
    
    y = w*x +b
    return y

x = torch.tensor([[1.0], [2.0]])
yhat = forward(x)

print('Output using functions model :\n',yhat)

######################################################
# Linear Regression using bulitin models
######################################################

model = Linear(in_features=1, out_features=1)
# print(list(model.parameters()))

x = torch.tensor([0.0])
x_ = torch.tensor([[1.0],[2.0]])

yhat = model(x_)

print('Output using builtin model :\n',yhat)

######################################################
# Custom models
######################################################

class LR(nn.Module):

    # Constructor
    def __init__(self, in_size, out_size):
        # Inherit from parent
        super(LR,self).__init__()
        self.linear = nn.Linear(in_size, out_size)
    
    # Prediction function
    def forward (self, x):
        out = self.linear(x)
        return out

model = LR(1,1)

model.state_dict()['linear.weight'].data[0] = torch.tensor([0.5153])
model.state_dict()['linear.bias'].data[0] = torch.tensor([-0.4414])
print(list(model.parameters()))

yhat = model(x_)

print('Output using custom model :\n',yhat)
print("Python Dictionary :", model.state_dict())
print('keys :', model.state_dict().keys())
print('values :', model.state_dict().values())


######################################################
# Linear Regression : Training
######################################################

w = torch.tensor(-15.0, requires_grad= True)
b = torch.tensor(-10.0, requires_grad= True)
x = torch.arange(-3,3,0.1).view(-1,1)
f = -3 * x
y = f + 0.1 *torch.rand(x.size())
lr = 0.1

def forward(x):
    f = w * x + b
    return f

def criterion(y, yhat):
    # loss = 1/len(y) * torch.sum((y - yhat)**2)
    loss = torch.mean((yhat - y) ** 2)
    return loss

print('\nTraining .......')
Loss = []
def train_model(iter):
    for epoch in range (iter):
        
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

train_model(10)
print(Loss)