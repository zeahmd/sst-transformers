import click
import numpy as np
from training import train

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("-n", "--name", default='bert-base', help='model name')
@click.option("-r", "--root", is_flag=True, help='SST root or all')
@click.option("-b", "--binary", is_flag=True, help='SST binary or fine')
@click.option("-e", "--epochs", default=30, help="no of training iterations/epochs")
@click.option("-p", "--patience", default=np.inf, help='patience for early stopping')
@click.option("-s", "--save", is_flag=True, help="save model")
def run(name, root, binary, epochs, patience, save):
    """
    \b
    SST Transformers
    ----------------\n
    \b
    Transformer Models:
    1- bert-base
    2- bert-large
    3- roberta-base
    4- roberta-large
    5- distilbert
    6- albert-xlarge
    7- albert-xxlarge
    8- xlmroberta-base
    9- xlmroberta-large
    10- electra-small
    11- electra-large
    12- mobilebert
    13- gpt2-medium

    \b
    Dataset Details:
    root: only root sentences
    all: sentences parsed into phrases
    binary: only rows with sentiment negative, positive
    fine: negative, partially negative, neutral, partially positive, positive

    \b
    Note: name parameter can take one of above models only.
    """
    train(name, root, binary, epochs, patience, save)


if __name__ == "__main__":
    run()
