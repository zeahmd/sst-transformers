from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertConfig
from transformers import AlbertTokenizer, AlbertForSequenceClassification, AlbertConfig
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, XLMRobertaConfig
from transformers import ElectraTokenizer, ElectraForSequenceClassification, ElectraConfig
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification, MobileBertConfig
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from loguru import logger

def load_transformer(name, binary):
    logger.info(f"Loading model {name}!")
    if name == 'bert-base':
        config = BertConfig.from_pretrained('bert-base-uncased')
        if not binary:
            config.num_labels = 5
        return {'model': BertForSequenceClassification.from_pretrained('bert-base-uncased', config=config),
                'tokenizer': BertTokenizer.from_pretrained('bert-base-uncased')}
    elif name == 'bert-large':
        config = BertConfig.from_pretrained('bert-large-uncased')
        if not binary:
            config.num_labels = 5
        return {'model': BertForSequenceClassification.from_pretrained('bert-large-uncased', config=config),
                'tokenizer': BertTokenizer.from_pretrained('bert-large-uncased')}
    elif name == 'roberta-base':
        config = RobertaConfig.from_pretrained('roberta-base')
        if not binary:
            config.num_labels = 5
        return {'model': RobertaForSequenceClassification.from_pretrained('roberta-base', config=config),
                'tokenizer': RobertaTokenizer.from_pretrained('roberta-base')}
    elif name == 'roberta-large':
        config = RobertaConfig.from_pretrained('roberta-large')
        if not binary:
            config.num_labels = 5
        return {'model': RobertaForSequenceClassification.from_pretrained('roberta-large', config=config),
                'tokenizer': RobertaTokenizer.from_pretrained('roberta-large')}
    elif name == 'distilbert':
        config = DistilBertConfig.from_pretrained('distilbert-base-uncased')
        if not binary:
            config.num_labels = 5
        return {'model': DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', config=config),
                'tokenizer': DistilBertTokenizer.from_pretrained('distilbert-base-uncased')}
    elif name == 'albert-xlarge':
        config = AlbertConfig.from_pretrained('albert-xlarge-v2')
        if not binary:
            config.num_labels = 5
        return {'model': AlbertForSequenceClassification.from_pretrained('albert-xlarge-v2', config=config),
                'tokenizer': AlbertTokenizer.from_pretrained('albert-xlarge-v2')}
    elif name == 'albert-xxlarge':
        config = AlbertConfig.from_pretrained('albert-xxlarge-v2')
        if not binary:
            config.num_labels = 5
        return {'model': AlbertForSequenceClassification.from_pretrained('albert-xxlarge-v2', config=config),
                'tokenizer': AlbertTokenizer.from_pretrained('albert-xxlarge-v2')}
    elif name == 'xlmroberta-base':
        config = XLMRobertaConfig.from_pretrained('xlm-roberta-base')
        if not binary:
            config.num_labels = 5
        return {'model': XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', config=config),
                'tokenizer': XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')}
    elif name == 'xlmroberta-large':
        config = XLMRobertaConfig.from_pretrained('xlm-roberta-large')
        if not binary:
            config.num_labels = 5
        return {'model': XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-large', config=config),
                'tokenizer': XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')}
    elif name == 'electra-small':
        config = ElectraConfig.from_pretrained('google/electra-small-discriminator')
        if not binary:
            config.num_labels = 5
        return {'model': ElectraForSequenceClassification.from_pretrained('google/electra-small-discriminator', config=config),
                'tokenizer': ElectraTokenizer.from_pretrained('google/electra-small-discriminator')}
    elif name == 'electra-large':
        config = ElectraConfig.from_pretrained('google/electra-large-discriminator')
        if not binary:
            config.num_labels = 5
        return {'model': ElectraForSequenceClassification.from_pretrained('google/electra-large-discriminator',
                                                                          config=config),
                'tokenizer': ElectraTokenizer.from_pretrained('google/electra-large-discriminator')}
    elif name == 'mobilebert':
        config = MobileBertConfig.from_pretrained('google/mobilebert-uncased')
        if not binary:
            config.num_labels = 5
        return {'model': MobileBertForSequenceClassification.from_pretrained('google/mobilebert-uncased', config=config),
                'tokenizer': MobileBertTokenizer.from_pretrained('google/mobilebert-uncased')}
    else:
        raise ValueError
