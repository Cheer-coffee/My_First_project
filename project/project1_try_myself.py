import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset, random_split

# 基础设置（配置显卡，数据路径）
device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
print("使用设备：", device)

train_dataset = r'D:\BME\data\chest_xray\train'
test_dataset = r'D:\BME\data\chest_xray\test'

# 数据增强、转换
def add_noise(x):
    return x + 0.01 * torch.randn_like(x)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.01, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
train_dataset_with_aug = datasets.ImageFolder(root=train_dataset, transform=train_transform)
val_dataset_no_aug = datasets.ImageFolder(root=train_dataset, transform=val_transform)
test_dataset = datasets.ImageFolder(root=test_dataset, transform=val_transform)

# 划分训练集和验证集
total_size = len(train_dataset_with_aug)
train_size = int(total_size * 0.8)
val_size = total_size - train_size

torch.manual_seed(42)
train_indices, val_indices = random_split(range(total_size), [train_size, val_size])
train_subset = Subset(train_dataset_with_aug, train_indices)
val_subset = Subset(val_dataset_no_aug, val_indices)
print(f'训练集{len(train_subset)}张')
print(f'验证集{len(val_subset)}张')
print(f'测试集{len(test_dataset)}张')

# 创建DataLoader
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# 数据处理好了，就要创建模型了
# 先加载初始的ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

for param in model.parameters():
    param.requires_grad = False
for param in model.layer4.parameters():
    param.requires_grad = True

# 修改输出层
model.fc = nn.Linear(in_features=512, out_features=2)
model = model.to(device)

trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"可训练参数{trainable_count}")

# 训练阶段
class_weights = torch.tensor([2.89, 1]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0001
)

num_epochs = 3
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    for batch_index, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _ , predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()

        if (batch_index + 1)% 50 == 0:
            print(f"Epoch{epoch+1}  {batch_index+1}/{len(train_loader)}  "
                  f"Loss {loss.item():.4f}")

    train_acc = 100 * train_correct / train_total
    avg_train_loss = loss.item()/len(train_loader)
    # 这里train_total 和 len(train_loader)有什么区别，感觉都是一样的

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            pneumonia_prob = probs[:, 1]
            predicted = (pneumonia_prob > 0.7).long()
            # 这一段要再问问
            val_total += labels.size(0)
            val_correct +=predicted.eq(labels).sum().item()

    val_acc = 100 * val_correct / val_total
    avg_val_loss = val_loss / val_total

    print(f'Epoch{epoch+1}/{num_epochs}')
    print(f'训练集 Loss：{avg_train_loss:.4f}  Acc:{train_acc:.4f}')
    print(f'验证集 Loss：{avg_val_loss:.4f}  Acc：{val_acc:.4f}')






