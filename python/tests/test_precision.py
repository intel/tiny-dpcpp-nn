import torch

datatype = torch.bfloat16
# datatype = torch.float
param = torch.tensor([0.2412]).to(datatype)
lr = torch.tensor(1e-3, dtype=torch.float)  # Use higher precision for lr
grad = torch.tensor([1.3471 * 1e-2]).to(datatype)


# Perform the subtraction in higher precision
param_adjusted = (param.float() - grad.float() * lr).to(datatype) - param.float()

print("Param", param)
print("Grad", grad)
print("Param - grad*lr", param - grad * lr)
print("Param - grad*lr - param", param_adjusted)
