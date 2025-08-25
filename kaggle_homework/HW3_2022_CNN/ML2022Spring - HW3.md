
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

### 计划&实践

**优化方向:**
・Training Augmentation
・Train Longer 
・Cross Validation + Ensemble
・Test Time Augmentation
・Model Design

```
阶段0：数据准备和基础建设
数据策略：创建小数据集（训练集20-30%，保持类别平衡）
目标：为快速迭代准备基础设施
为什么：建立可重复的实验流程，确保所有后续实验都在相同条件下比较

阶段1：数据增强策略确定
数据使用：小数据集
模型：保持原始baseline CNN不变
实验内容：

无增强baseline
轻度增强：RandomHorizontalFlip + RandomRotation(10°)
中度增强：上述 + ColorJitter + RandomResizedCrop
重度增强：上述 + RandomErasing/Mixup

设计理由：

为什么小数据集：数据增强效果在小数据集上就能很好体现，且模式一致
为什么先做增强：效果稳定可预测，能快速建立性能提升的信心
为什么固定模型：单变量原则，只测试增强效果，排除架构影响

输出：确定最佳增强策略，建立增强后的性能baseline

阶段2：模型架构优化
数据使用：小数据集 + 阶段1确定的最佳增强
实验内容：

原始baseline（作为对比）
加深版本：增加卷积层
加宽版本：增加通道数
正则化版本：添加Dropout
现代化版本：残差连接、更好的激活函数

设计理由：

为什么小数据集：架构对比需要多次实验，小数据集能快速迭代
为什么用最佳增强：在强化的数据基础上比较架构，结果更有说服力
为什么多种方向：探索不同的提升路径（深度vs宽度vs正则化）

输出：确定最佳模型架构

阶段3：训练策略优化
数据使用：小数据集 + 最佳增强 + 最佳架构
实验内容：

学习率对比：0.001, 0.0003, 0.0001
学习率调度：StepLR, CosineAnnealingLR
优化器对比：Adam vs SGD+momentum
训练时长：确定最佳epoch数

设计理由：

为什么小数据集：超参数效果在小数据集上能快速验证
为什么此时做：在最佳架构基础上优化训练，避免架构和训练参数的交互混淆
单变量测试：一次只改一个超参数，确保知道什么起作用

输出：最佳训练配置

阶段4：完整数据集验证
数据使用：完整数据集
配置：阶段1-3确定的最佳组合
目标：

验证小数据集结论在大数据集上的有效性
建立可靠的完整数据集性能baseline
观察是否需要调整（如epoch数、学习率等）

设计理由：

为什么此时用完整数据集：前面所有组件都已优化，现在验证整体效果
为什么只用最佳配置：避免在大数据集上浪费时间做已经验证过的对比
为什么叫验证：主要目标是确认，而不是重新选择

输出：完整数据集上的稳定性能baseline

阶段5：高级技巧优化
数据使用：完整数据集（因为高级技巧通常需要更多数据才能体现优势）
实验内容：

测试时增强：TTA策略
模型集成：训练3-5个不同初始化的模型
高级正则化：Label smoothing, Cutmix等
架构微调：基于完整数据集的表现进行小幅调整

设计理由：

为什么完整数据集：这些技巧的效果在大数据集上更明显
为什么最后做：这些是锦上添花，需要稳定的基础
为什么分开做：每种技巧分别验证，便于理解贡献度


数据集使用原则总结
小数据集适用场景：

快速原型：架构对比、基础超参数
单变量测试：数据增强策略、学习率等
趋势验证：确认改进方向是否正确
成本敏感：需要多次实验的场景

完整数据集适用场景：

最终验证：确认小数据集结论的有效性
性能上限：追求最佳性能的训练
高级技巧：需要大数据量才能体现优势的方法
最终提交：生成提交结果的训练

科学性保证：

一致的评估标准：同样的验证集分割、评估指标
多次运行平均：重要的结论需要多次实验确认
渐进式改进：每个阶段基于前一阶段的最佳结果
详细记录：每个实验的配置、结果都要记录

这个方案如何？是否符合你的期望？有什么需要调整的地方吗？

```