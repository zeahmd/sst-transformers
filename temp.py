from sst_transformers import SSTDataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    train_dataset = SSTDataset(root=False, binary=False, split='train')
    print("len: {}".format(len(train_dataset)))
    train_iterator = DataLoader(dataset=train_dataset,
                                batch_size=32,
                                shuffle=False)

    for text, sentiment in train_iterator:
        #print("{}, {}".format(text, sentiment))
        print(list(text))
        print(sentiment.numpy().tolist())
        break
