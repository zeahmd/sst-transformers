import torch
from torch.utils.data import DataLoader
from sst_transformers.models import load_transformer
from sst_transformers.utils import transformer_params
from utils import evaluation_metrics
from tqdm import tqdm
from math import ceil
import copy


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_step(model, inputs, labels, criterion, optimizer):
    optimizer.zero_grad()

    preds = model(inputs)
    loss = criterion(preds, labels)

    loss.backward()
    optimizer.step()

    return preds, loss

def eval_step(model, inputs, labels, criterion):
    preds = model(inputs)
    loss = criterion(preds, labels)

    return preds, loss

def train_epoch(model, train_dataset, criterion, optimizer, batch_size):
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    correct_count = 0
    total_loss = 0

    model.train()
    with tqdm(total=ceil(len(train_dataset)/batch_size), desc='train', unit='batch') as pbar:
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            preds, loss = train_step(model, imgs, labels, criterion, optimizer)

            preds = torch.argmax(preds, axis=1)
            correct_count += (preds == labels).sum().item()
            total_loss += loss.item()
            pbar.update(1)

    return correct_count / len(train_dataset), total_loss / len(train_dataset)

def eval_epoch(model, eval_dataset, criterion, batch_size, split):
    eval_loader = DataLoader(dataset=eval_dataset,
                            batch_size=batch_size,
                            shuffle=True)

    correct_count = 0
    total_loss = 0

    model.eval()
    with torch.no_grad():
        with tqdm(total=ceil(len(eval_dataset)/batch_size), desc=split, unit='batch') as pbar:
            for imgs, labels in eval_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                preds, loss = eval_step(model, imgs, labels, criterion)

                preds = torch.argmax(preds, axis=1)
                correct_count += (preds == labels).sum().item()
                total_loss = loss.item()
                pbar.update(1)


    return correct_count / len(eval_dataset), total_loss / len(eval_dataset)

def train(name, optim, num_epochs=25):
    #best_model_wts = copy.deepcopy(model.state_dict())
    #best_acc = 0.0

    model = load_transformer(name)['model']
    tokenizer = load_transformer(name)['tokenizer']

    batch_size = transformer_params(name)['batch_size']
    learning_rate = transformer_params(name)['learning_rate']

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        train_acc, train_loss = train_epoch(model, train_dataset, criterion,
                                            optimizer, BATCH_SIZE)
        print("train_acc: {:.4f}, train_loss: {:.4f}".format(train_acc, train_loss))

        dev_acc, dev_loss = eval_epoch(model, dev_dataset, criterion,
                                       BATCH_SIZE)
        print("dev_acc: {:.4f}, dev_loss: {:.4f}".format(dev_acc, dev_loss))

        if dev_acc > best_acc:
            best_acc = dev_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return model
