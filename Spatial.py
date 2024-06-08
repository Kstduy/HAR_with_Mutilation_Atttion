import torch
import torch.nn as nn
import torchvision
import numpy as np
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(224),#裁剪图片尺寸为224*224
    transforms.CenterCrop(224),#中心裁剪，确保图像中心的信息被保留
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载所有图片
full_dataset = datasets.ImageFolder(root='Data/RGB', transform=transform)

# 计算划分训练集和测试集的样本数量
train_size = int(0.7 * len(full_dataset))  # 训练集大小，占总数据集的70%
test_size = len(full_dataset) - train_size  # 测试集大小，占总数据集的30%

# 划分数据集
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义基本的卷积块（Convolution Block）
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(identity)
        out = self.relu(out)
        return out

# 定义spatial模型
class spatial(nn.Module):
    def __init__(self, num_classes=10):
        super(spatial, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(BasicBlock, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.dropout=nn.Dropout(p=0.5)

    def make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
spatial_stream_model = spatial(num_classes=10)

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
spatial_stream_model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(spatial_stream_model.parameters(), lr=0.001)

# 创建列表以存储训练损失和测试精度
train_losses = []
test_accuracies = []

#记录存储混淆矩阵的列表
confusion_matrices = []

num_epochs = 5
best_test_acc = 0.0

for epoch in range(num_epochs):
    #训练
    spatial_stream_model.train() 
    
    running_loss = 0.0
    train_pbar = tqdm(train_loader, total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs} - Training', ncols=100)
    for i, data in enumerate(train_pbar, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        #前向传播
        outputs = spatial_stream_model(inputs)

        #计算损失
        loss = criterion(outputs, labels)

        #反向传播和梯度更新
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    # 计算并打印每个epoch的平均损失
    train_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{num_epochs} - Training Loss: {train_loss}')
    train_losses.append(train_loss)

    # 评估模型在测试集上的性能
    spatial_stream_model.eval()  

    correct = 0
    total = 0

    with torch.no_grad():
        predicted_labels = []
        ground_truth = []
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            outputs = spatial_stream_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # 收集预测标签和真实标签
            predicted_labels.extend(predicted.cpu().numpy())
            ground_truth.extend(labels.cpu().numpy())

    # 计算并打印测试精度
    test_accuracy = correct / total
    print(f'Epoch {epoch+1}/{num_epochs} - Test Accuracy: {test_accuracy}')
    test_accuracies.append(test_accuracy)

    # 计算当前epoch的混淆矩阵并将其添加到列表中
    cm = confusion_matrix(ground_truth, predicted_labels)
    confusion_matrices.append(cm)

    # 如果当前模型在验证集上的性能优于之前的最佳性能，则保存模型参数
    if test_accuracy > best_test_acc:
        best_test_acc = test_accuracy
        best_model_name = 'spatial'+'_epoch'+str(epoch+1)+'test_Acc'+str(test_accuracy)+'.pth'
        torch.save(spatial_stream_model.state_dict(),best_model_name)


print('Finished Training')

epochs = np.arange(1, num_epochs + 1)
train_data = np.column_stack((epochs, train_losses))
test_data = np.column_stack((epochs,test_accuracies))

epochs = np.arange(1, num_epochs + 1)
train_data = np.column_stack((epochs, train_losses))
test_data = np.column_stack((epochs,test_accuracies))

np.savetxt("spatial_stream_train_data_without_dropout.txt", train_data)
np.savetxt("spatial_stream_test_data_without_dropout.txt", test_data)

# 将混淆矩阵列表保存为NumPy数组，并保存到文件中
confusion_matrices_array = np.stack(confusion_matrices)
np.save("Spatial_without_dropout_confusionmatrix.npy", confusion_matrices_array)
