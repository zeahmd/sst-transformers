import torch
from torch.utils.data import DataLoader
from sst_transformers.models import load_transformer
from sst_transformers.utils import transformer_params
from utils import evaluation_metrics
from tqdm import tqdm
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
    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        preds, loss = train_step(model, imgs, labels, criterion, optimizer)

        preds = torch.argmax(preds, axis=1)
        correct_count += (preds == labels).sum().item()
        total_loss += loss.item()

    return correct_count / len(train_dataset), total_loss / len(train_dataset)

def eval_epoch(model, dev_dataset, criterion, batch_size):
    dev_loader = DataLoader(dataset=dev_dataset,
                            batch_size=batch_size,
                            shuffle=True)

    correct_count = 0
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for imgs, labels in dev_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            preds, loss = eval_step(model, imgs, labels, criterion)

            preds = torch.argmax(preds, axis=1)
            correct_count += (preds == labels).sum().item()
            total_loss = loss.item()

    return correct_count / len(dev_dataset), total_loss / len(dev_dataset)

def train(model, train_dataset, dev_dataset, criterion, optimizer, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

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
