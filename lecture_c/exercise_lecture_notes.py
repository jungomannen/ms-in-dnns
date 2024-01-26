import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim

# functional paradigm
W1 = torch.randn((20, 5), requires_grad=True)
b1 = torch.randn((20,), requires_grad=True)

W2 = torch.randn((20, 5), requires_grad=True)
b2 = torch.randn((20,), requires_grad=True)


def model(x):
    z1 = F.linear(x, W1, b1)
    y1 = F.relu(z1)
    z2 = F.linear(y1, W2, b2)


x = torch.randn(5)
print(model(x))


# modular paradigm
linear1 = nn.Linear(5, 20, bias=True)
linear2 = nn.Linear(20, 10, bias=True)


def modu(x):
    z1 = linear1(x)
    y1 = F.relu(z1)
    z2 = linear2(y1)
    return z2


x = torch.randn(5)
print(modu(x))


# implementation as a custom class
class SampleModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(5, 20, bias=True)
        self.linear2 = nn.Linear(20, 10, bias=True)

    def forward(self, x):
        z1 = self.linear1(x)
        y1 = F.relu(z1)
        z2 = self.linear2(y1)
        return z2


model = SampleModule()
x = torch.randn(5)
print(model(x))  # model(x) calls the forward function thanks to nn.Module

print(
    [param[0] for param in model.named_parameters()]
)  # also prints the params of linear1 and linear2!
print(model)

# compute loss
loss_func = nn.CrossEntropyLoss()
x = torch.randn(32, 5)
preds = model(x)
targets = torch.randint(10, (32,))
loss = loss_func(preds, targets)
loss.backward()
print(model.linear1.weight.grad)  # derivative of loss with respect to w_ij (aka theta_ij)

# model.train() # select train mode
# model.eval() # select eval mode (skips batch normalization etc)
# something.no_grad()

# important: register things as parameters or as buffers (= not buffer)(via create_buffer)


"""optimizer by hand"""

W1 = torch.randn((20, 5), requires_grad=True)
b1 = torch.randn((20,), requires_grad=True)

W2 = torch.randn((20, 5), requires_grad=True)
b2 = torch.randn((20,), requires_grad=True)


def model(x):
    z1 = F.linear(x, W1, b1)
    y1 = F.relu(z1)
    z2 = F.linear(y1, W2, b2)


x = torch.randn(32, 5)
preds = model(x)
targets = torch.randdint(10, (32,))
loss = F.cross_entropy(preds, targets)
loss.backward()

lr = 0.1  # learning rate

with torch.no_grad():
    W1 = W1 - lr * W1.grad
    W2 = W2 - lr * W2.grad
    b1 = b1 - lr * b1.grad
    b2 = b2 - lr * b2.grad


W1.requires_grad = True
W2.requires_grad = True
b1.requires_grad = True
b2.requires_grad = True


"""optimization in modular interface"""


class SampleModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(5, 20, bias=True)
        self.linear2 = nn.Linear(20, 10, bias=True)

    def forward(self, x):
        z1 = self.linear1(x)
        y1 = F.relu(z1)
        z2 = self.linear2(y1)
        return z2


model = SampleModule()
loss_func = nn.CrossEntropyLoss()
x = torch.randn(32, 5)
preds = model(x)
targets = torch.randint(10, (32,))
loss = loss_func(preds, targets)
loss.backward()

# optimizer
sgd = optim.SGD(model.parameters(), lr=0.1)  # replace SGD with adam etc
sgd.step()
sgd.zero_grad()


# checkpoining
[print(key, value) for key, value in model.state_dict().items()]
torch.save(model.state_dict(), "ckpt.pt")
# just unpickle it to look inside

model = SampleModule()
model.load_state_dict(torch.load("ckpt.pt"))  # recreates model from file
