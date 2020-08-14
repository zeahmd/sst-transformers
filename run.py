import click
import numpy as np
from training import train

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("-n", "--name", default='lstm', help='model name')
@click.option("-r", "--root", is_flag=True, help='SST root or all')
@click.option("-b", "--binary", is_flag=True, help='SST binary or fine')
@click.option("-e", "--epochs", default=30, help="no of training iterations/epochs")
@click.option("-p", "--patience", default=np.inf, help='patience for early stopping')
@click.option("-s", "--save", is_flag=True, help="save model")
def run(name, root, binary, epochs, patience, save):
    """
    SST Transformers:\n
    -----------\n
    root: only root sentences\n
    all: sentences parsed into phrases\n
    binary: only rows with sentiment negative, positive\n
    fine: negative, partially negative, neutral, partially positive, positive\n
    """
    train(name, root, binary, epochs, patience, save)



if __name__ == "__main__":
    run()