from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    RobertaConfig,
)
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    DistilBertConfig,
)
from transformers import AlbertTokenizer, AlbertForSequenceClassification, AlbertConfig
from transformers import (
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification,
    XLMRobertaConfig,
)
from transformers import (
    ElectraTokenizer,
    ElectraForSequenceClassification,
    ElectraConfig,
)
from transformers import BartTokenizer, BartForSequenceClassification, BartConfig
from loguru import logger


def load_transformer(name, binary):
    logger.info(f"Loading model {name}!")
    if name == "bert-base":
        config = BertConfig.from_pretrained("bert-base-uncased")
        if not binary:
            config.num_labels = 5
        return {
            "model": BertForSequenceClassification.from_pretrained(
                "bert-base-uncased", config=config
            ),
            "tokenizer": BertTokenizer.from_pretrained("bert-base-uncased"),
        }
    elif name == "bert-large":
        config = BertConfig.from_pretrained("bert-large-uncased")
        if not binary:
            config.num_labels = 5
        return {
            "model": BertForSequenceClassification.from_pretrained(
                "bert-large-uncased", config=config
            ),
            "tokenizer": BertTokenizer.from_pretrained("bert-large-uncased"),
        }
    elif name == "roberta-base":
        config = RobertaConfig.from_pretrained("roberta-base")
        if not binary:
            config.num_labels = 5
        return {
            "model": RobertaForSequenceClassification.from_pretrained(
                "roberta-base", config=config
            ),
            "tokenizer": RobertaTokenizer.from_pretrained("roberta-base"),
        }
    elif name == "roberta-large":
        config = RobertaConfig.from_pretrained("roberta-large")
        if not binary:
            config.num_labels = 5
        return {
            "model": RobertaForSequenceClassification.from_pretrained(
                "roberta-large", config=config
            ),
            "tokenizer": RobertaTokenizer.from_pretrained("roberta-large"),
        }
    elif name == "distilbert":
        config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
        if not binary:
            config.num_labels = 5
        return {
            "model": DistilBertForSequenceClassification.from_pretrained(
                "distilbert-base-uncased", config=config
            ),
            "tokenizer": DistilBertTokenizer.from_pretrained("distilbert-base-uncased"),
        }
    elif name == "albert-base-v2":
        config = AlbertConfig.from_pretrained("albert-base-v2")
        if not binary:
            config.num_labels = 5
        return {
            "model": AlbertForSequenceClassification.from_pretrained(
                "albert-base-v2", config=config
            ),
            "tokenizer": AlbertTokenizer.from_pretrained("albert-base-v2"),
        }
    elif name == "xlmroberta-base":
        config = XLMRobertaConfig.from_pretrained("xlm-roberta-base")
        if not binary:
            config.num_labels = 5
        return {
            "model": XLMRobertaForSequenceClassification.from_pretrained(
                "xlm-roberta-base", config=config
            ),
            "tokenizer": XLMRobertaTokenizer.from_pretrained("xlm-roberta-base"),
        }
    elif name == "electra-small":
        config = ElectraConfig.from_pretrained("google/electra-small-discriminator")
        if not binary:
            config.num_labels = 5
        return {
            "model": ElectraForSequenceClassification.from_pretrained(
                "google/electra-small-discriminator", config=config
            ),
            "tokenizer": ElectraTokenizer.from_pretrained(
                "google/electra-small-discriminator"
            ),
        }
    elif name == "bart-large":
        config = BartConfig.from_pretrained("facebook/bart-large")
        if not binary:
            config.num_labels = 5
        return {
            "model": BartForSequenceClassification.from_pretrained(
                "facebook/bart-large", config=config
            ),
            "tokenizer": BartTokenizer.from_pretrained("facebook/bart-large"),
        }
    else:
        raise ValueError
