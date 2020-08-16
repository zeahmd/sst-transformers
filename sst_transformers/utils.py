

def get_binary_label(sentiment):
    if sentiment <= 1:
        return 0
    else:
        return 1


def transformer_params(name):
    '''if name == 'bert-base-uncased' or name == 'bert-large-uncased':
        return {'batch_size': 32,
                'learning_rate': 5e-5}
    elif name == 'roberta-base' or name == 'roberta-large':
        return {'batch_size': 32,
                'learning_rate': 2e-5}
    elif name == 'distilbert':
        return {'batch_size': 32,
                'learning_rate': 1e-5}
    elif name == 'albert-xlarge' or name == 'albert-xxlarge':
        return {'batch_size': 32,
                'learning_rate': 1e-5}
    elif name == 'xlmroberta-base' or name == 'xlmroberta-large':
        return {'batch_size': 32,
                'learning_rate': 1e-5}
    elif name == 'electra-small' or name == 'electra-large':
        return {'batch_size': 32,
                'learning_rate': 1e-5}
    elif name == 'mobilebert':
        return {'batch_size': 32,
                'learning_rate': 1e-5}
    elif name == 'gpt2-medium':
        pass
    '''
    return {'batch_size': 32,
            'learning_rate': 1e-5}
    '''
    else:
        raise ValueError
    '''

