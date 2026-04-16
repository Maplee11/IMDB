import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def create_data(n_samples=10000):
    # 在 [-2pi, 2pi] 之间随机采样
    x = torch.linspace(-2 * np.pi, 2 * np.pi, n_samples).view(-1, 1).to(device)
    y = torch.sin(x)
    # 划分训练集和测试集 (80% 训练, 20% 测试)
    indices = torch.randperm(n_samples)
    train_indices = indices[:int(n_samples * 0.8)]
    test_indices = indices[int(n_samples * 0.8):]
    
    return x[train_indices], y[train_indices], x[test_indices], y[test_indices]


class SineTower(nn.Module):
    def __init__(self, output_dims: list):
        super(SineTower, self).__init__()
        self.network = nn.Sequential()

        # 堆叠 MLP
        for i in range(0, len(output_dims) - 1):
            self.network.add_module(f"linear_{i}", nn.Linear(output_dims[i], output_dims[i+1]))
            if i < len(output_dims) - 2: # 最后一层不加 relu
                self.network.add_module(f"relu_{i}", nn.ReLU())

    def forward(self, x):
        return self.network(x)


# 设置 device
device = torch.device("mps") if torch.mps.is_available() else torch.device("cpu")
print("Using device: ", device)

# 模型定义，数据构造
train_x, train_y, test_x, test_y = create_data()
model = SineTower(output_dims=[1, 64, 64, 1]).to(device)
criterion = nn.MSELoss() # 回归任务常用均方误差
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练循环
model.train() # 将模型切换到训练模式，启用 dropout 等随机操作
TOTAL_EPOCHS = 1000
for epoch in range(TOTAL_EPOCHS):
    optimizer.zero_grad()
    outputs = model(train_x)
    loss = criterion(outputs, train_y)
    loss.backward()
    optimizer.step()
    
    if epoch % (TOTAL_EPOCHS // 10) == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# 测试打分
model.eval() # 将模型切换到推理模式，禁用 dropout 等
with torch.no_grad():
    predictions = model(test_x)
    mse = criterion(predictions, test_y).item()
    ss_res = torch.sum((test_y - predictions) ** 2)
    ss_tot = torch.sum((test_y - torch.mean(test_y)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)

print(f"\nFinal Test MSE: {mse:.6f}")
print(f"R-squared Score: {r2_score.item():.4f}")
