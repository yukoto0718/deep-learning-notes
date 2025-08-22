from transformers import BertModel
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 直接从网络加载，第一次会下载，后续会使用缓存
pretrained = BertModel.from_pretrained("google-bert/bert-base-chinese")
pretrained.to(DEVICE)

print(pretrained)
#三种方法-要看具体是对应哪一种哪一种能用
# print(pretrained[0])
# print(pretrained["embeddings"])
# print(pretrained.embeddings)

#定义下游任务模型
#读上面的代码，我们可以看到模型最后输出的是768个特征向量，并且是linear的
class Model(torch.nn.Module):
    def __init__(self):
        #模型结构设计
        super().__init__()
        self.fc = torch.nn.Linear(768, 2)
    def forward(self,input_ids,attention_mask,token_type_ids):
        #上游任务不参与训练
        with torch.no_grad():
            # 将输入传递给BERT模型
            # input_ids: 词汇ID序列 [batch_size, seq_len]
            # attention_mask: 注意力掩码，标记哪些位置是真实token [batch_size, seq_len]
            # token_type_ids: 句子类型ID，区分句子A和句子B [batch_size, seq_len]
            out = pretrained(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        #下游任务参与训练
        # 提取[CLS]标记的输出作为整个句子的表示
        # out.last_hidden_state的形状是[batch_size, seq_len, hidden_size]
        # [:,0]选择每个样本的第一个位置（[CLS]标记），形状变为[batch_size, hidden_size]

        #原张量形状: torch.Size([2, 4, 3])
        #使用 [:,0] 索引后的形状: torch.Size([2, 3])
        #完整写法 example_tensor[:, 0, :]
        out = self.fc(out.last_hidden_state[:,0])
        # 应用softmax获得概率分布
        # dim=1表示在第1维度（类别维度）上计算softmax
        # dim=0: 沿着行方向（列内归一化）
        # dim=1: 沿着列方向（行内归一化）
        out = out.softmax(dim=1)
        return out

