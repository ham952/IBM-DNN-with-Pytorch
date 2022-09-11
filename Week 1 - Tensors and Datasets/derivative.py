import torch

def derivative(val):
    # initialize the tensor
    x = torch.tensor(val, dtype=float, requires_grad= True)

    # define the equation
    # y = x**2
    y = x**2 + 2*x + 1

    # calculate derivative
    y.backward()

    # calculate derivative at given value
    return x.grad

x = derivative(2)
print(x)

def partial_derivative(u_val,v_val):
    # initialize the tensor
    u = torch.tensor(u_val, dtype=float, requires_grad= True)
    v = torch.tensor(v_val, dtype=float, requires_grad= True)

    # define the equation
    f = u*v + u **2

    # calculate derivative
    f.backward()

    # calculate derivative at given value
    return u.grad, v.grad

u, v = partial_derivative(1,2)
print(u,v)
