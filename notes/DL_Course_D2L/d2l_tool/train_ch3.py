# train_ch3.py
import torch
from torch import nn
import matplotlib.pyplot as plt
from IPython import display

# 1. 累加器类
class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n
    
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    
    def reset(self):
        self.data = [0.0] * len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# 2. 准确率计算函数
def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

# 3. 简化版可视化类
class Animator:
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, figsize=(6, 4)):  # ✅ 改小图片尺寸
        self.xlabel = xlabel
        self.ylabel = ylabel  
        self.legend = legend or []
        self.xlim = xlim
        self.ylim = ylim
        self.figsize = figsize
        
        # 初始化图表 - 创建双y轴
        plt.ioff()  # ✅ 关闭交互模式，防止重复显示
        self.fig, self.ax1 = plt.subplots(figsize=figsize)
        self.ax2 = self.ax1.twinx()  # ✅ 创建第二个y轴用于loss
        self.X, self.Y = None, None

    def add(self, x, y):
        """向图表中添加数据点"""
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
            
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
            
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        
        # 清除并重绘
        self.ax1.clear()
        self.ax2.clear()
        
        # ✅ 分别绘制：loss用左轴，准确率用右轴
        if len(self.X) >= 1 and self.X[0]:  # train loss
            self.ax1.plot(self.X[0], self.Y[0], 'b-', label='train loss')
            self.ax1.set_ylabel('Loss', color='b')
            self.ax1.tick_params(axis='y', labelcolor='b')
        
        if len(self.X) >= 2 and self.X[1]:  # train acc
            self.ax2.plot(self.X[1], self.Y[1], 'r--', label='train acc')
        
        if len(self.X) >= 3 and self.X[2]:  # test acc  
            self.ax2.plot(self.X[2], self.Y[2], 'g-.', label='test acc')
        
        # 设置右轴（准确率）
        self.ax2.set_ylabel('Accuracy', color='r')
        self.ax2.tick_params(axis='y', labelcolor='r')
        self.ax2.set_ylim(0, 1)  # 准确率范围0-1
        
        # 设置x轴
        if self.xlabel:
            self.ax1.set_xlabel(self.xlabel)
        if self.xlim:
            self.ax1.set_xlim(self.xlim)
            
        # 添加图例
        lines1, labels1 = self.ax1.get_legend_handles_labels()
        lines2, labels2 = self.ax2.get_legend_handles_labels()
        self.ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
        
        self.ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # ✅ 只用一种显示方式
        display.clear_output(wait=True)
        display.display(self.fig)

# 4. 训练函数
def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一个迭代周期"""
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)  # 训练损失总和、训练准确度总和、样本数
    
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用自定义的SGD更新器
            l.sum().backward()
            updater(X.shape[0])
            
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    
    return metric[0] / metric[2], metric[1] / metric[2]

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['train loss', 'train acc', 'test acc'])
    
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
        
        # 打印进度
        print(f'epoch {epoch + 1}, loss {train_metrics[0]:.3f}, '
              f'train acc {train_metrics[1]:.3f}, test acc {test_acc:.3f}')
    
    train_loss, train_acc = train_metrics
    print(f'\n训练完成！最终结果：')
    print(f'训练损失: {train_loss:.3f}')
    print(f'训练准确率: {train_acc:.3f}')
    print(f'测试准确率: {test_acc:.3f}')

# 5. SGD更新器
def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

print("工具函数加载完成！")