import torch
from torch import nn
from torch.optim import SGD

# 定义一个简单的线性模型
model = nn.Linear(1, 1)

# 定义损失函数
criterion = nn.MSELoss()

# 定义优化器
optimizer = SGD([model.bias], lr=0.01)

# 训练数据
x = torch.randn(10, 1)
y = 2 * x + 3

# 计算损失值
y_pred = model(x)
loss = criterion(y_pred, y)
print(loss.item())

# 计算梯度
optimizer.zero_grad()
loss.backward()

# 取相反数
for param in model.parameters():
    param.grad.data.neg_()

# 更新参数
optimizer.step()

# 输出更新后的损失值
y_pred = model(x)
loss = criterion(y_pred, y)
print(f'Loss: {loss.item()}')