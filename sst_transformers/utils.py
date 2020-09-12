

def get_binary_label(sentiment):
    if sentiment < 2:
        return 0
    if sentiment > 2:
        return 1
    raise ValueError("Invalid sentiment")


def transformer_params(name):
    return {'batch_size': 32,
            'learning_rate': 1e-5}

