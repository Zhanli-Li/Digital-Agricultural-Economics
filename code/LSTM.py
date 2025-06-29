import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim
import matplotlib.pyplot as plt

# 从外部Excel文件导入数据
file_path = r'sales_threeyear.xlsx'
data = pd.read_excel(file_path, sheet_name='销量')
data['销售日期'] = pd.to_datetime(data['销售日期'])
data.set_index('销售日期', inplace=True)

# 替换即可
data = data[['花菜类']]

# LSTM模型定义
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=6, output_size=1, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  # 单层 LSTM，单元数为 6
        self.fc1 = nn.Linear(hidden_size, 7)  # 全连接层，神经元数为 7
        self.fc2 = nn.Linear(7, output_size)  # 最后一层输出 1 个值

    def forward(self, x):
        out, _ = self.lstm(x)  # LSTM 输出 (batch_size, seq_len, hidden_size)
        out = self.fc1(out)  # 全连接层1
        out = self.fc2(out)  # 最后一层输出
        return out

# 未来预测函数
def predict_future(model, last_sequence, steps, scaler):
    future_preds = []
    input_seq = last_sequence.copy()

    for _ in range(steps):
        input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        pred = model(input_tensor).detach().numpy()[-1, 0]  # 取最后一个时间步的预测值
        future_preds.append(pred)
        input_seq = np.append(input_seq, pred)[-len(last_sequence):]  # 滑动更新输入序列
    return scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

# 数据分割
train_size = int(len(data) * 0.7)
train, test = data[:train_size], data[train_size:]

# 对训练集和测试集分别归一化
scaler = MinMaxScaler(feature_range=(0, 1))
train_data = scaler.fit_transform(train[['花菜类']].values.reshape(-1, 1))  # 训练集归一化
test_data = scaler.transform(test[['花菜类']].values.reshape(-1, 1))  # 测试集归一化

# 构建输入张量 (batch_size=1, seq_len, input_size=1)
train_tensor = torch.tensor(train_data, dtype=torch.float32).unsqueeze(0)  # 增加 batch 维度
test_tensor = torch.tensor(test_data, dtype=torch.float32).unsqueeze(0)  # 增加 batch 维度

# LSTM训练与预测
for column in data.columns:
    print(f"Training LSTM model for {column}...")

    # 初始化模型
    model = LSTMModel(input_size=1, hidden_size=6, output_size=1, num_layers=3)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 保存训练损失
    losses = []

    # 训练模型
    model.train()
    for epoch in range(600):  # 训练600轮
        optimizer.zero_grad()
        outputs = model(train_tensor)
        loss = criterion(outputs.squeeze(), train_tensor.squeeze())
        loss.backward()
        optimizer.step()

        # 保存每轮的训练损失
        losses.append(loss.item())
        if (epoch + 1) % 50 == 0:  # 每50轮打印一次损失
            print(f"Epoch [{epoch + 1}/600], Loss: {loss.item():.6f}")

    # 训练集拟合
    model.eval()
    train_preds = model(train_tensor).detach().numpy().squeeze(0)  # 获取完整时间步预测值
    train_preds = scaler.inverse_transform(train_preds)  # 逆归一化
    train_true = scaler.inverse_transform(train_data)  # 训练集真实值

    # 测试集预测
    test_preds = model(test_tensor).detach().numpy().squeeze(0)
    test_preds = scaler.inverse_transform(test_preds)
    test_true = scaler.inverse_transform(test_data)

    # 合并训练集和测试集的真实值
    all_true = np.concatenate((train_true.flatten(), test_true.flatten()), axis=0)

    # 合并训练集和测试集的预测值
    all_preds = np.concatenate((train_preds.flatten(), test_preds.flatten()), axis=0)

    # 计算相对误差率
    relative_error = np.abs((all_true - all_preds) / all_true) * 100

    # 未来预测
    future_preds = predict_future(model, test_data.flatten(), steps=60, scaler=scaler)

    # 训练集评价指标
    train_r2 = r2_score(train_true, train_preds)
    train_rmse = sqrt(mean_squared_error(train_true, train_preds))
    train_mape = np.mean(np.abs((train_true - train_preds) / train_true)) * 100

    # 测试集评价指标
    test_r2 = r2_score(test_true, test_preds)
    test_rmse = sqrt(mean_squared_error(test_true, test_preds))
    test_mape = np.mean(np.abs((test_true - test_preds) / test_true)) * 100

    # 保存到Excel
    output_file = f"销量_{column}_lstm_predictions_with_metrics.xlsx"
    with pd.ExcelWriter(output_file) as writer:
        pd.DataFrame({'真实值': all_true}).to_excel(writer, sheet_name='真实值', index=False)
        pd.DataFrame({'拟合值': train_preds.flatten()}).to_excel(writer, sheet_name='拟合值', index=False)
        pd.DataFrame({'测试值': test_preds.flatten()}).to_excel(writer, sheet_name='测试值', index=False)
        pd.DataFrame({'相对误差率': relative_error.flatten()}).to_excel(writer, sheet_name='相对误差率', index=False)
        pd.DataFrame({'未来预测': future_preds.flatten()}).to_excel(writer, sheet_name='未来预测', index=False)
        pd.DataFrame({'训练集评价指标': ['R2', 'RMSE', 'MAPE'],
                      '值': [train_r2, train_rmse, train_mape]}).to_excel(writer, sheet_name='训练集评价指标', index=False)
        pd.DataFrame({'测试集评价指标': ['R2', 'RMSE', 'MAPE'],
                      '值': [test_r2, test_rmse, test_mape]}).to_excel(writer, sheet_name='测试集评价指标', index=False)
        pd.DataFrame({'训练过程损失': losses}).to_excel(writer, sheet_name='训练损失', index=False)

    print(f"Results for {column} saved to {output_file}")

print("\nAll predictions are complete.")
