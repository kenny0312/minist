import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from dataset.mnist_loader import load_mnist_data
from model.cnn_model import CNN

# 超参数
EPOCH = 1
LR = 0.001
BATCH_SIZE = 50

# 加载数据
train_loader, test_data = load_mnist_data(batch_size=BATCH_SIZE)

# 初始化模型
cnn = CNN()
optimizer = optim.Adam(cnn.parameters(), lr=LR)
loss_func = CrossEntropyLoss()

# 训练过程
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        output = cnn(b_x)
        loss = loss_func(output, b_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_x = test_data.data.type(torch.FloatTensor)[:2000] / 255.
            test_x = torch.unsqueeze(test_x, dim=1)
            test_y = test_data.targets[:2000]

            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].detach().numpy()
            accuracy = float((pred_y == test_y.numpy()).astype(int).sum()) / float(test_y.size(0))
            print(f'Epoch: {epoch}, Step: {step}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}')

# 保存模型
torch.save(cnn.state_dict(), './checkpoints/cnn2.pkl')
