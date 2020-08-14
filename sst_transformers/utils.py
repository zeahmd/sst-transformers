

def get_binary_label(sentiment):
    if sentiment <= 1:
        return 0
    else:
        return 1

