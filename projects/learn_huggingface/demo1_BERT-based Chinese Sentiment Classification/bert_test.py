import torch
from MyData import MyDataset
from torch.utils.data import DataLoader
from net import Model
from transformers import BertTokenizer #,AdamW
from torch.optim import AdamW

#定义训练设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = 100 #定义训练回合

token = BertTokenizer.from_pretrained("google-bert/bert-base-chinese")

#自定义函数，对数据进行编码处理
def collate_fn(data):
    sentes = [i[0] for i in data]
    label = [i[1] for i in data]
    #编码
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sentes,
        truncation=True,
        padding='max_length',
        max_length=350,
        return_tensors="pt",
        return_length=True
    )
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]
    labels = torch.LongTensor(label)
    return input_ids, attention_mask, token_type_ids, labels

#创建数据集
test_dataset = MyDataset("test")

#创建DataLoader
test_loader = DataLoader(
    dataset = test_dataset,
    batch_size = 16,
    shuffle = True,
    drop_last = True,
    collate_fn=collate_fn
)

if __name__ == '__main__':
    acc = 0
    total = 0
    # 开始测试
    print(DEVICE)
    model = Model().to(DEVICE)
    model.load_state_dict(torch.load("params/0bert.pt"))
    model.eval()
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(test_loader):
        input_ids, attention_mask, token_type_ids, labels = input_ids.to(DEVICE), attention_mask.to(
            DEVICE), token_type_ids.to(DEVICE), labels.to(DEVICE)

        out = model(input_ids, attention_mask, token_type_ids)

        out = out.argmax(dim=1)
        acc += (out == labels).sum().item()
        total += len(labels)
    print(acc / total)