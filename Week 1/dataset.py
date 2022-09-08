import torch
from torch.utils.data import Dataset
from torchvision import transforms

class toy_set(Dataset):

    def __init__(self, length = 100, transfrom = None):

        self.x = 2 * torch.ones(length,2)
        self.y = torch.ones(length,1)
        self.len = length
        self.transform = transfrom

    def __getitem__(self, index):

        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):

        return self.len

class add_mul(object):

    def __init__(self, addx= 1, muly= 1):
        
        self.addx = addx
        self.muly = muly
    
    def __call__(self, sample):

        x = sample[0]
        y = sample[1]
        x = x + self.addx
        y = y * self.muly

        sample = x, y
        return sample

class mult(object):

    def __init__(self, mul= 100):
        
        self.mul = mul
    
    def __call__(self, sample):

        x = sample[0]
        y = sample[1]
        x = x * self.mul
        y = y * self.mul

        sample = x, y
        return sample

##################################
# Constructing and calling dataset
##################################
length = 200
data = toy_set()

print(len(data))

# updating length of dataset 
data.len = length

print(data.len)
print(data[0])

for i in range(3):
    x,y = data[i]
    print(i, 'x:',x, 'y:',y )

##################################
# Applying transform to dataset
##################################


# Step 1 : form an Object
a_m = add_mul()

#Step 2 : pass object to constructor as argument
data_ = toy_set(transfrom=a_m)
# data_ = toy_set()
# data_.transform = a_m

print('\n-------------------------\n')
for i in range(3):
    x,y = data_[i]
    print(i, 'x:',x, 'y:',y )

#########################################
# Applying multuple transforms to dataset
#########################################

# Step 1 : form an Object of transform
data_transfrom = transforms.Compose([add_mul(), mult()])

# Step 2 : pass object to constructor as argument
data_trans = toy_set(transfrom=data_transfrom)
data_trans.len = 10

print('\n-------------------------\n')
for i, data in enumerate(data_trans):
    print(i, ' x:', data[0], 'y:', data[1])