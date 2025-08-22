from torch.utils.data import Dataset
from datasets import load_from_disk

class MyDataset(Dataset):
    #初始化数据
    def __init__(self, split):
        # 从磁盘加载数据
        self.dataset = load_from_disk(r"C:\MyHug_Projects\Hug_Learn\demo1\data\ChnSentiCorp")
        if split == "train":
            self.dataset = self.dataset["train"]
        elif split == "validation":
            self.dataset = self.dataset["validation"]
        elif split == "test":
            self.dataset = self.dataset["test"]
        else:
            print("数据集名称错误！")
    #获取数据集大小
    def __len__(self):
        return len(self.dataset)
    #对数据做定制化处理
    def __getitem__(self, item):
        text = self.dataset[item]["text"]
        label = self.dataset[item]["label"]
        return text, label

if __name__ == "__main__":
    dataset = MyDataset("validation")
    for data in dataset:
        print(data)
