
# HW 2 	Classification

Homework 2 Phoneme Classification

**kaggle url:** [ML2022Spring-hw2](https://www.kaggle.com/competitions/ml2022spring-hw2/overview)
**PDF url:** [HW02.pdf](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2022-course-data/hw2_slides%202022.pdf)

### Task Introduction
・Data Preprocessing: Extract MFCC features from raw waveform (already
done)
・Classification: Perform framewise phoneme classification using
pre-extracted MFCC features


### Grading -- Kaggle and Hints
```
---- simple baseline ----
Score:0.45797
(original)ML2022Spring_HW2.ipynb

---- medium baseline ----
Score: 0.69747 
concat n frames, add layers

---- strong baseline ----
Score:0.75028
concat n, batchnorm, dropout, add layers

---- boss baseline ----
Score: 0.82324
sequence-labeling(using RNN)
```

### Analysis
**1.各个文件的作用**
```
| 文件名             | 内容               | 作用                  |
|-------------------|-------------------|-----------------------|
| train_split.txt   | 训练数据的文件名列表 | 告诉我们哪些录音用来训练  | 
| test_split.txt    | 测试数据的文件名列表 | 告诉我们哪些录音用来测试  |
| train_labels.txt  | 训练数据的标签      | 每一帧对应哪个音素（答案）| 
| feat/train/*.pt   | 训练特征文件        | 每个录音的 MFCC 特征数据|
| feat/test/*.pt    | 测试特征文件        | 测试录音的 MFCC 特征   | 
```

**2.什么是MFCC特征？**

**MFCC = 梅尔频率倒谱系数**

**类比：**

- 原始语音就像一首复杂的交响乐
- MFCC就像把这首交响乐简化成一个"声音指纹"
- 每一帧的MFCC是39个数字，描述了这25毫秒内声音的特征

```
原始语音波形 → [复杂的波浪线]
↓ MFCC提取
每一帧 → [1.2, -0.5, 2.1, 0.8, ...] (39个数字)
```

🎯 **我们的任务具体是什么？**

**输入：** 一帧语音的MFCC特征（39个数字）
**输出：** 这一帧对应的音素类别（41种音素中的一种）

就像给计算机看一个"声音指纹"，让它猜这是什么音素。
```
例如train_labels的19-198-0008：

开始： 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0...
     ↑ 静音段（53个0）- 说话前的安静

然后： 11 11 11 11 11 11 11 11 11 11 11 11
     ↑ 音素11持续了12帧（大约120毫秒）

接着： 39 39 39 39 39 39
     ↑ 音素39持续了6帧（大约60毫秒）

继续： 35 35 35 35 35 35 35
     ↑ 音素35持续了7帧

...依此类推...

结尾： 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     ↑ 又是静音段 - 说话后的安静
```
```
19-198-0008  0   0   0   11  11  39  39  35  35  25  25  ...
             ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓
            帧1 帧2 帧3 帧4 帧5 帧6 帧7 帧8 帧9 帧10帧11 ...
   ____________________________________________________________
   
19-198-0008.pt:
tensor([
    [1.23, -0.45, 2.1, ..., 0.67],  # 帧1: 39个数字 → 标签: 0 (静音)
    [1.19, -0.42, 2.0, ..., 0.71],  # 帧2: 39个数字 → 标签: 0 (静音)  
    [1.15, -0.38, 1.9, ..., 0.63],  # 帧3: 39个数字 → 标签: 0 (静音)
    [2.34, 1.23, -0.8, ..., 1.45],  # 帧4: 39个数字 → 标签: 11 (音素/d/)
    [2.41, 1.31, -0.7, ..., 1.52],  # 帧5: 39个数字 → 标签: 11 (音素/d/)
    [3.12, 0.89, 2.1, ..., 0.88],   # 帧6: 39个数字 → 标签: 39 (音素/zh/)
    [3.08, 0.91, 2.0, ..., 0.92],   # 帧7: 39个数字 → 标签: 39 (音素/zh/)
    ...
])
   ____________________________________________________________
   
时间:    0ms    25ms   50ms   75ms   100ms  125ms  150ms  ...
帧号:    帧1    帧2    帧3    帧4    帧5    帧6    帧7    ...
标签:     0      0      0     11     11     39     39    ...
特征:   39个   39个   39个   39个   39个   39个   39个   ...
```

**3.concat_nframes - 帧拼接数量**

```python
concat_nframes = 1              
# the number of frames to concat with, n must be odd (total 2k+1 = n frames)
```

**详细解释：**
- **作用**: 决定每一帧要"看"多少相邻帧的信息
- **当前值**: 1（只看当前帧，不看前后帧）
- **必须奇数**: 确保有明确的"中心帧"

**不同值的效果：**
```python
concat_nframes = 1: 只看当前帧
                    [当前帧] → 39个特征

concat_nframes = 3: 看前1帧+当前+后1帧  
                    [前帧][当前帧][后帧] → 39×3=117个特征

concat_nframes = 5: 看前2帧+当前+后2帧
                    [前前][前][当前][后][后后] → 39×5=195个特征
```

**调参建议：**
- **值越大**: 模型看到更多上下文，可能更准确，但计算量更大
- **值越小**: 计算快，但可能丢失重要的时序信息
- **经验值**: 通常3-11效果较好




### strongbaseline
**1.Hyper-parameters部分:** 
```
# 基础版本 → strongline版本
concat_nframes = 1    →  concat_nframes = 19
batch_size = 512      →  batch_size = 2048  
num_epoch = 5         →  num_epoch = 50
hidden_layers = 1     →  hidden_layers = 3
hidden_dim = 256      →  hidden_dim = 1024

# strongline新增的参数：
early_stopping = 8    # 新增：早停轮数
```
**2.正则化:** 
```python
#(1)L2正则化（权重衰减）
optimizer = torch.optim.AdamW（.• weight_decay=0.01）
#(2)Dropout正则化
nn.Dropout （0.35）
#(3)BatchNorm正则化
nn.BatchNorm1d（output_dim）
#(4)早停正则化
early_stopping = 8
```
**3.优化器配置&学习率:** 
```python
# strongline新增的优化器配置
optimizer = torch.optim.AdamW(model.parameters(), 
                              lr=learning_rate*5,      # 学习率提高5倍
                              weight_decay=0.01)       # 新增：权重衰减

# strongline新增的学习率调度器
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=8, T_mult=2, eta_min=learning_rate/2)
```

### bossbaseline
**1.MLP改成BiLSTM-CRF**？

**2.Post-processing**？

