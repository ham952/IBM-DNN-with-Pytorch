import torch
from torch.nn import Linear
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(1)

# in_features = # of columns of weights matrix = m
# out_featues = # of rows of weights matrix = n
# in_features == # of coumns of Input X

######################################################
# Multiple input Linear Regression : 
######################################################

model = Linear(in_features=2, out_features=2)
print(list(model.parameters()))
weights = model.state_dict()['weight'] 
print('Dimensions of w :',weights.shape)

# X = torch.tensor([[1.0,3.0,4.0],[4.0,5.0,7.0]])
X = torch.tensor([[1.0,3.0],[4.0,5.0], [10.0,8.8]])

# X = r x c
# w = n x m
# y = X.wT + b = r x n

print('Dimensions of X :', X.shape)
yhat = model(X)
print('Dimensions of Yhat :',yhat.shape)
print(yhat)


######################################################
# Multiple input/output Linear Regression : Custom Module
######################################################

output_size = 2

class LR(nn.Module):
    
    # Constructor
    def __init__(self, input_size, output_size):
        super(LR, self).__init__()
        
        self.linear = nn.Linear(input_size, output_size)
    
    # Prediction
    def forward(self, x):
        
        out = self.linear(x)
        return out

my_model = LR(X.shape[1],output_size)
print(list(my_model.parameters()))
output = my_model(X)
print(output)

class Data2D(Dataset):

    # Constructor
    def __init__(self):

        self.x = torch.zeros(20,3)
        self.x[:,0] = torch.arange(-1,1,0.1)
        self.x[:,1] = torch.arange(-1,1,0.1)
        
        self.w = torch.tensor([[1.0],[1.0],[1.0]])
        self.b = 1
        self.f = torch.mm(self.x, self.w) +self.b
        self.y = self.f + 0.1 * torch.randn((self.x.shape[0],output_size))

        self.len = self.x.shape[0]
    
    # Getter
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # Get Length
    def __len__(self):
        return self.len

# Create Dataloader object
dataset = Data2D()
train_loader = DataLoader(dataset=dataset, batch_size=1)

# Build in cost function
criterion = nn.MSELoss()

# Create model object
my_model = LR(dataset.x.shape[1],output_size)
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
            loss = criterion(yhat, y)

            # zero the gradients before running the backward pass
            optimizer.zero_grad()

            # backward pass: compute gradient of the loss with respect to all the learnable parameters
            loss.backward()

            # updata parameters
            optimizer.step()
        
        # training loss / Cost
        Yhat = my_model(dataset.x)
        cost = criterion(Yhat, dataset.y)
        Cost.append(cost.item())    
    
    return Yhat


print('Training ..')
yhat = train_model()
print(Cost)

print('Ground Truth Vs Predicted Value :')
for i in range(len(dataset.y)):
    print(dataset.y[i], '\t', yhat[i])
