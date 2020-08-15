from transformers import BertTokenizer, BertForSequenceClassification, BertConfig


def load_transformer(name, binary):
    if name == 'bert':
        config = BertConfig.from_pretrained('bert-base-uncased')
        if not binary:
            config.num_labels = 5
        return {'model': BertForSequenceClassification.from_pretrained('bert-base-uncased', config=config),
                'tokenizer': BertTokenizer.from_pretrained('bert-base-uncased')}
    else:
        raise ValueError
