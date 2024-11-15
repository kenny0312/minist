import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_mnist_data(root="./data", batch_size=50):
    """
    加载 MNIST 数据集并返回 DataLoader。

    Args:
        root (str): 数据存储路径。
        batch_size (int): 批次大小。

    Returns:
        tuple: (训练集 DataLoader, 测试集 DataLoader)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 归一化到 [-1, 1]
    ])
    
    # 加载数据集
    train_dataset = datasets.MNIST(root=root, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=root, train=False, transform=transform, download=True)
    
    # 创建 DataLoader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_dataset

