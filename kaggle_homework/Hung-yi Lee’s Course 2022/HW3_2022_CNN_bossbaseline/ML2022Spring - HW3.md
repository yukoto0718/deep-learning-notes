
# HW 3 CNN

HW3 Image Classification (CNN)

**kaggle url:** [ML2022Spring-hw3](https://www.kaggle.com/competitions/ml2022spring-hw3b/overview)
**PDF url:** [HW03.pdf](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2022-course-data/Machine%20Learning%20HW3%20-%20Image%20Classification.pdf)

### Objectives

1. Solve image classification with convolutional neural networks.
2. Improve the performance with data augmentations.
3. Understand popular image model techniques such as residual.


### Grading -- Kaggle and Hints
```
---- simple baseline ----
Score: 0.50099 

---- medium baseline ----
Score: 0.73207
Training Augmentation + Train Longer 

---- strong baseline ----
Score:0.81872
Training Augmentation + Model Design + Train Looonger (+ Cross Validation + Ensemble) 

---- boss baseline ----
Score:0.88446
Training Augmentation + Model Design +Test Time Augmentation + Train Looonger (+ Cross Validation + Ensemble)
```
>这里没有按照要求来做，直接用的预训练模型

### 数据结构
```
数据集文件名格式：
0_0.jpg    → 类别0
0_1.jpg    → 类别0  
1_0.jpg    → 类别1
1_1.jpg    → 类别1
...
10_100.jpg → 类别10
```
代码中提取标签的关键部分：
```
def __getitem__(self, idx):
    fname = self.files[idx]  # 例如: "/path/food11/training/0_123.jpg"
    im = Image.open(fname)
    im = self.transform(im)
    
    # 这里提取标签
    try:
        label = int(fname.split("/")[-1].split("_")[0])
        #     文件名: "0_123.jpg"
        #     .split("/")[-1] → "0_123.jpg" (取最后部分)
        #     .split("_")[0] → "0" (取下划线前的数字)
        #     int("0") → 0 (转为整数)
    except:
        label = -1  # 测试集没有标签
    
    return im, label  # 返回(图片, 标签)
```


### 计划&实践

**优化方向:**
・Training Augmentation
・Train Longer 
・Cross Validation + Ensemble
・Test Time Augmentation
・Model Design

**1.Model Design:**
选择ResNet18
```
import torchvision.models as models

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # 使用预训练ResNet18
        #pretrained=True 使用预训练参数
        self.backbone = models.resnet18(pretrained=True)
        # 替换最后一层为11分类
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 11)
        
    def forward(self, x):
        return self.backbone(x)
```
**2.Training Augmentation:**
数据增强
```
test_tfm = transforms.Compose([
    transforms.Resize((224, 224)),  # 改大尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet标准化！
                        std=[0.229, 0.224, 0.225])
])

train_tfm = transforms.Compose([
    transforms.Resize((224, 224)),  # 改大尺寸
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0),
    transforms.RandomPerspective(distortion_scale=0.6, p=0.6),
    transforms.RandomAffine(degrees=(-30, 30), translate=(0, 0.4), scale=(0.8, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                        std=[0.229, 0.224, 0.225])   
])
```

**3.Train Longer**
```python

#修改训练参数
n_epochs = 3
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)

# 改成
n_epochs = 20  # 增加训练轮数
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)  

#添加学习率调度 - 在optimizer定义后加
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

#在训练循环validation后添加 - 在每个epoch结束时加一行
scheduler.step()  # 更新学习率

```

### 成绩
![image1](./img/hw3_img1.png)