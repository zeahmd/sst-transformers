from transformers import BertTokenizer, BertForSequenceClassification
import torch


def load_transformer(name):
    if name == 'bert':
        return {'model': BertForSequenceClassification.from_pretrained('bert-base-uncased'),
                'tokenizer': BertTokenizer.from_pretrained('bert-base-uncased')}
    else:
        raise ValueError
