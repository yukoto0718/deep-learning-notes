
# HW 8 	Anomaly Detection

HW8 Anomaly Detection

**kaggle url:** [ML2022Spring-hw8](https://www.kaggle.com/competitions/ml2022spring-hw8/)
**PDF url:** [HW08.pdf](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2022-course-data/Machine%20Learning%20Homework%208%20Anomaly%20Detection.pdf)

### Task
Unsupervised anomaly detection:
Training a model to determine whether the given image is similar with the training data.

### Data
● Training data
    ○ 100000 human faces
● Testing data
    ○ About 10000 from the same distribution with training data (label 0)
    ○ About 10000 from another distribution (anomalies, label 1)
● Format
    ○ data/
    |----- trainingset.npy
    |----- testingset.npy
    ○ Shape: (#images, 64, 64, 3) for each .npy file

### Grading -- Kaggle and Hints
```
---- simple baseline ----
Score: 0.52970
Sample code

---- medium baseline ----
Score: 0.72895
Adjust model structure

---- strong baseline ----
Score:0.77196
Multi-encoder autoencoder

---- boss baseline ----
Score:0.79506
Add random noise and an extra classifier
```


### 关联知识
```
"保守策略": [3, 16, 32, 64, 128],
"平衡策略": [3, 32, 64, 128, 256],  # ← 
"激进策略": [3, 64, 128, 256, 512],
"跳跃策略": [3, 128, 256, 512, 1024]

inplace=False: 创建新张量，安全但耗内存
inplace=True:  直接修改原张量，快速且省内存
```

### 计划&实践
换模型，用resnet，训练久一点，缩小批次大小
不要加噪音效果反而会更好

### 成绩
![image1](./img/hw8_img1.png)