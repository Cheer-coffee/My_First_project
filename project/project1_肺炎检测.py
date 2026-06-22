import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset, random_split

# ============================================
# 1. 基本配置
# ============================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#device配置到cuda
print(f"使用设备: {device}")
print(f"GPU名称: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

# 数据路径(本地)
train_dir = r'D:\BME\data\chest_xray\train'
test_dir = r'D:\BME\data\chest_xray\test'


# ============================================
# 2. 数据增强函数(普通函数,不用lambda)
# ============================================
def add_noise(x):
    return x + 0.01 * torch.randn_like(x)


# ============================================
# 3. 定义transforms
# ============================================
train_transform = transforms.Compose([ #用增强dataset来扩大dataset的数据量
    transforms.Resize((224, 224)), #将图像rsize到224,224。1.模型限制 2.小尺寸占显存更少 3.医学图像的特点，小尺寸病灶依然明显
    transforms.Grayscale(num_output_channels=3), #因为model是3通道的，能不改模型就不改模型，所以将图像复制3份
    transforms.RandomHorizontalFlip(p=0.5), #随机水平翻转，probability=0.5
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.1, contrast=0.2),
    transforms.ToTensor(),#转换成张量，因为前面都是图像，模型只认张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406],#归一化，1.将transforms.Totensor()后全是正数的数据转变成有正有负，使梯度更新平衡
                         std=[0.229, 0.224, 0.225]),#归一化，将数据按（x-mean)/std 处理，这组数据是大量图片的RGB均值
    transforms.Lambda(add_noise)
])

val_transform = transforms.Compose([ #验证集的数据只需要基本处理，不能做增强
    transforms.Resize((224, 224)),#resize缩放
    transforms.Grayscale(num_output_channels=3),
    # 拓展至3通道
    transforms.ToTensor(),
    # 将图像转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406],#做归一化处理，将转换张量后全是正数的数据变为有正有负的，利于梯度更新平衡
                         std=[0.229, 0.224, 0.225])
])


# ============================================
# 4. 加载数据
# ============================================
train_dataset_with_aug = datasets.ImageFolder(root=train_dir, transform=train_transform)
train_dataset_no_aug = datasets.ImageFolder(root=train_dir, transform=val_transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=val_transform)


# ============================================
# 5. 划分训练集和验证集
# ============================================
total_size = len(train_dataset_with_aug)
# 所有增强的训练集的总数
train_size = int(0.8 * total_size)
# 训练集是总数的80%
val_size = total_size - train_size

torch.manual_seed(42)
# 随机数种子，42是参数，让每次生成的随机数都是一样的，为后面random_split做准备
train_indices, val_indices = random_split( range(total_size), [train_size, val_size])
# train_indices,val_indices是标签。indices是index的复数  用random_split 将 range(total_size) 按照[train_size, val_size]分成两份
train_subset = Subset(train_dataset_with_aug, train_indices.indices)
val_subset = Subset(train_dataset_no_aug, val_indices.indices)
# 数据集的划分比较复杂，现在来解释一下：torch.manual_seed(42)生成同一串随机数，使random_split()的划分每次都相同。
# random_split()按照[train_size, val_size]将range(total_size)分成两部分
# train_subset在原始图像中按train_indices挑选图像，通过数据增强获得训练集
# val_subset在原始图像中按照val_indices挑选不重叠的图像，组成验证集
print(f"训练集: {len(train_subset)} 张")
print(f"验证集: {len(val_subset)} 张")
print(f"测试集: {len(test_dataset)} 张")
print(f"类别: {train_dataset_with_aug.classes}")


# ============================================
# 6. 创建DataLoader
# ============================================
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
# 例如train_loader 用Dataloader将train_subset按照32张每批(batch_size),打乱(shuffle=True),windows系统下并行子程序必须为0，不然会报错

# ============================================
# 7. 创建模型
# ============================================
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

for param in model.parameters():
    param.requires_grad = False

for param in model.layer4.parameters():
    param.requires_grad = True
# 上面的代码冻结了前3层神经网络，解冻了第4层神经网络
# 原因：前三层神经网络越靠前越通用，第1层处理边缘、线条，第2层处理纹理形状，第3层局部特征组合（还算通用）
# 第4层高级语义特征需要通过训练集训练
# 总而言之就是前3层的参数都是预训练好了的，不用改变，直接用就好

model.fc = nn.Linear(in_features=512, out_features=2)
model = model.to(device)
# fc层也要改变，原来out_features = 1000,但现在是2分类问题，所以out_features=2

# 验证可训练参数
trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"可训练参数: {trainable_count}")


# ============================================
# 8. 训练
# ============================================
class_weights = torch.tensor([2.89, 1.0]).to(device)
# 设置权重，按首字母排序，第一个是normal，第二个是pneumonia。这里给normal2.89权重，是因为normal图像是少数，模型会无脑判断为pneumonia
# 这样正确率也很高，因此要给normal更高的权重，这样在模型把pneumonia错判成normal时会受到更大的惩罚，更加认真地对待少数类
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0001
)

num_epochs = 3  # 本地先跑3个epoch看看
# 训练3轮，每轮训练一遍，验证一遍

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    # 训练阶段的标志
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        # loss是一个张量 .item()将张量变为python数字，累加到train_loss中，最后除以batch数得到平均loss
        _, predicted = outputs.max(1)
        # output的输出是[32, 2] 32张图(行)，每张图两个分数(列)，分别代表normal和pneumonia的分数。
        # .max(1)沿第一维挑出大的那个，就是挑出模型认为图像属于哪一类，在构成两个tensor
        # 第一个tensor是挑出来的得分，第二个tensor是该分数所在的标签，0代表normal，1代表pneumonia
        # _ 就是丢弃得分tensor，只保留预测的标签tensor
        train_total += labels.size(0)
        # 输出label.size第0维的数字，避免最后一组不满32张图却被按照32张图计算
        train_correct += predicted.eq(labels).sum().item()
        # .eq(labels)就是将predicted和labels对答案，看看有多少对的
        # .sum()是将对的数量统计出来，item()把张量转换成python数字
        if (batch_idx + 1) % 50 == 0: # 每50个batch打印一次进度
            print(f"  Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, "
                  f"Loss: {loss.item():.4f}")

    train_acc = 100 * train_correct / train_total
    avg_train_loss = train_loss / len(train_loader)

    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            pneumonia_prob = probs[:, 1]
            predicted = (pneumonia_prob > 0.7).long()
            # 这是用来替换_, predicted = outputs.max(1)的
            # torch.softmax()将outputs中的分数变成总和为1的比例值
            # probs[:,1]是只取第一列代表pneumonia的比例值，若pneumonia_prob>0.7才能判断该图像是pneumonia
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_acc = 100 * val_correct / val_total
    avg_val_loss = val_loss / len(val_loader)

    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print(f"  训练: Loss={avg_train_loss:.4f}, Acc={train_acc:.2f}%")
    print(f"  验证: Loss={avg_val_loss:.4f}, Acc={val_acc:.2f}%\n")


# ============================================
# 9. 保存模型(关键!避免重启后丢失)
# ============================================
torch.save(model.state_dict(), r'D:\BME\BME\project\pneumonia_model.pth')
print("模型已保存到 D:\\BME\\BME\\project\\pneumonia_model.pth")

model.eval()
test_correct = 0
test_total = 0

all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        pneumonia_prob = probs[:, 1]
        predicted = (pneumonia_prob > 0.7).long()
        # 这是用来替换_, predicted = outputs.max(1)的
        # torch.softmax()将outputs中的分数变成总和为1的比例值
        # probs[:,1]是只取第一列代表pneumonia的比例值，若pneumonia_prob>0.7才能判断该图像是pneumonia

        test_total += labels.size(0)
        test_correct += predicted.eq(labels).sum().item()

        all_predictions.append(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = 100 * test_correct/test_total
print(f"\n===测试结果===")
print(f"测试集准确率：{test_acc:.2f}%")
print(f"测试集总数：{test_total:.2f}")
print(f"测试集正确预测：{test_correct:.2f}")