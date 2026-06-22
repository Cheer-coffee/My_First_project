import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np

# ============================================
# 1. 基本配置
# ============================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

test_dir = r'D:\BME\data\chest_xray\test'

# ============================================
# 2. 测试集transforms(不带增强)
# ============================================
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ============================================
# 3. 加载测试集
# ============================================
test_dataset = datasets.ImageFolder(root=test_dir, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
print(f"测试集: {len(test_dataset)} 张")
print(f"类别: {test_dataset.classes}")

# ============================================
# 4. 重建模型结构 + 加载已训练的权重
# ============================================
model = models.resnet18(weights=None)

for param in model.parameters():
    param.requires_grad = False
for param in model.layer4.parameters():
    param.requires_grad = True

model.fc = nn.Linear(in_features=512, out_features=2)
model = model.to(device)

model.load_state_dict(torch.load(r'D:\BME\BME\project\pneumonia_model.pth'))
print("模型权重已加载")

# ============================================
# 5. 推理：只收集概率和标签，不在这里卡阈值
# ============================================
model.eval()
all_probs = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        pneumonia_prob = probs[:, 1]

        all_probs.extend(pneumonia_prob.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

# ============================================
# 6. 阈值扫描：同一批概率，套不同阈值
# ============================================
thresholds = np.arange(0.1, 1.0, 0.05)
normal_recalls = []
pneumonia_recalls = []

for t in thresholds:
    predicted = (all_probs > t).astype(int)

    normal_mask = (all_labels == 0)
    normal_recall = (predicted[normal_mask] == 0).sum() / normal_mask.sum()

    pneumonia_mask = (all_labels == 1)
    pneumonia_recall = (predicted[pneumonia_mask] == 1).sum() / pneumonia_mask.sum()

    normal_recalls.append(normal_recall)
    pneumonia_recalls.append(pneumonia_recall)
    print(f"阈值 {t:.2f}  ->  NORMAL recall {normal_recall:.2f}  |  PNEUMONIA recall {pneumonia_recall:.2f}")

# ============================================
# 7. 画图：两条曲线随阈值变化  —— 这部分你来填
# ============================================
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(8, 6))

# 关键填空1：画NORMAL recall曲线，x轴是thresholds，y轴是normal_recalls
plt.plot(thresholds, normal_recalls, label='NORMAL recall', marker='o')

# 关键填空2：仿照上面，画PNEUMONIA recall曲线
plt.plot(thresholds , pneumonia_recalls, label='PNEUMONIA recall', marker='o')

plt.xlabel('阈值 threshold')
plt.ylabel('recall')
plt.title('不同阈值下两类recall的变化')
plt.legend()
plt.grid(True)
plt.savefig(r'D:\BME\BME\project\threshold_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("曲线已保存到 threshold_curve.png")

final_threshold = 0.8
final_predicted = (all_probs > final_threshold).astype(int)
# .astype(int),将True/Flase转成1/0

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

cm = confusion_matrix(all_labels, final_predicted)
# 拿真实标签和预测结果做对比，生成一个2*2的表格
print("混淆矩阵：")
print(cm)

# 第三部分：画热力图
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['NORMAL', 'PNEUMONIA'],
            yticklabels=['NORMAL', 'PNEUMONIA'])
# sns.heatmap:上面2*2的数字表画成颜色块，数字越大颜色越深
# annot=True:在格子中显示数字
# fmt='d':数字在格子中按照整数显示(d=decimal),不写的话就是科学记数法
# cmap='Blues':蓝色系配色
# xticklabels=/yticklabels:横纵轴的标签名
plt.xlabel('预测类别')
plt.ylabel('真实类别')
plt.title('Confusion matrix(阈值0.8)')
plt.savefig(r'D:\BME\BME\project\confusion_matrix_0.8.png', dpi=150, bbox_inches='tight')
plt.close()

print(classification_report(all_labels, final_predicted, target_names=['NORMAL', 'PNEUMONIA']))

from PIL import Image

final_predicted = (all_probs>0.8).astype(int)

wrong_indices = np.where(final_predicted != all_labels)[0]
print("wrong_indices 的形状:", wrong_indices.shape)
print("前几个:", wrong_indices[:5])
print(f'一共{len(wrong_indices)}张错误')

show_indices = wrong_indices[:16]
plt.figure(figsize=(16, 16))
for i, idx in enumerate(show_indices):
    idx = int(idx)
    # test_dataset.imgs[idx]是(路径，标签)的元组，取路径
    img_path = test_dataset.imgs[idx][0]
    img = Image.open(img_path)

    true_label = all_labels[idx]
    pred_label = final_predicted[idx]

    class_names = ['NORMAL', 'PNEUMONIA']
    true_name = class_names[true_label]
    pred_name = class_names[pred_label]

    plt.subplot(4, 4, i+1)
    plt.imshow(img, cmap='gray')
    plt.title(f'真实：{true_name}\n预测{pred_name}', fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.savefig(r'D:\BME\BME\project\wrong_samples.png', dpi=120, bbox_inches='tight')
plt.close()
print('错误样本图已保存')