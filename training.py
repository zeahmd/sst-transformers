import torch
from sst_transformers.dataset import SSTDataset
from torch.utils.data import DataLoader
from sst_transformers.models import load_transformer
from sst_transformers.utils import transformer_params
from utils import evaluation_metrics
from tqdm import tqdm
from math import ceil
import copy


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_step(model, inputs, labels, optimizer):
    optimizer.zero_grad()

    outputs = model(inputs=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=labels)
    loss, logits = outputs[:2]

    loss.backward()
    optimizer.step()

    return logits, loss

def eval_step(model, inputs, labels):
    outputs = model(inputs=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=labels)
    loss, logits = outputs[:2]

    return logits, loss

def train_epoch(model, tokenizer, train_dataset, optimizer, batch_size):
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    correct_count = 0
    total_loss = 0

    model.train()
    with tqdm(total=ceil(len(train_dataset)/batch_size), desc='train', unit='batch') as pbar:
        for text, sentiment in train_loader:
            text = tokenizer(text, padding=True, return_tensors='pt').to(device)
            sentiment = torch.tensor(sentiment).unsqueeze(0).to(device)

            logits, loss = train_step(model, text, sentiment, optimizer)

            preds = torch.argmax(logits, axis=1)
            correct_count += (preds == sentiment).sum().item()
            total_loss += loss.item()
            pbar.update(1)

    return correct_count / len(train_dataset), total_loss / len(train_dataset)

def eval_epoch(model, tokenizer, eval_dataset, batch_size, split):
    eval_loader = DataLoader(dataset=eval_dataset,
                            batch_size=batch_size,
                            shuffle=True)

    correct_count = 0
    total_loss = 0

    model.eval()
    with torch.no_grad():
        with tqdm(total=ceil(len(eval_dataset)/batch_size), desc=split, unit='batch') as pbar:
            for text, sentiment in eval_loader:
                text = tokenizer(text, padding=True, return_tensors='pt').to(device)
                sentiment = torch.tensor(sentiment).unsqueeze(0).to(device)

                logits, loss = eval_step(model, text, sentiment)

                preds = torch.argmax(logits, axis=1)
                correct_count += (preds == sentiment).sum().item()
                total_loss = loss.item()
                pbar.update(1)


    return correct_count / len(eval_dataset), total_loss / len(eval_dataset)

def train(name, root, binary,  optim, num_epochs=25):
    #best_model_wts = copy.deepcopy(model.state_dict())
    #best_acc = 0.0

    #load model and tokenizer..
    transformer_container = load_transformer(name, binary)
    model = transformer_container['model']
    tokenizer = transformer_container['tokenizer']

    #load batch_size and learning rate..
    params_container = transformer_params(name)
    batch_size = params_container['batch_size']
    learning_rate = params_container['learning_rate']

    #load train, dev and test datasets..
    train_dataset = SSTDataset(root=root, binary=binary, split='train')
    dev_dataset = SSTDataset(root=root, binary=binary, split='dev')
    test_dataset = SSTDataset(root=root, binary=binary, split='test')

    #Intialize optimizer..
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        train_acc, train_loss = train_epoch(model, tokenizer, train_dataset,
                                            optimizer, batch_size)
        print("train_acc: {:.4f}, train_loss: {:.4f}".format(train_acc, train_loss))

        dev_acc, dev_loss = eval_epoch(model, tokenizer, dev_dataset,
                                       batch_size)
        print("dev_acc: {:.4f}, dev_loss: {:.4f}".format(dev_acc, dev_loss))

        if dev_acc > best_acc:
            best_acc = dev_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return model
