

def get_binary_label(sentiment):
    if sentiment <= 1:
        return 0
    else:
        return 1


def transformer_params(name):
    if name == 'bert':
        return {'batch_size': 32,
                'learning_rate': 5e-5}
    else:
        raise ValueError

