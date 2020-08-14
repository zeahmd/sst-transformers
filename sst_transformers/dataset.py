import pytreebank
from loguru import logger
from sst_transformers.utils import get_binary_label
from sst_transformers.preprocessing import preprocess_sst


class SSTDataset(object):
    def __init__(self, root, binary, split):
        logger.info("Loading sst dataset!")
        try:
            sst = pytreebank.load_sst()[split]
        except KeyError:
            logger.error("Invalid split key!")

        logger.info(f"Preparing dataset config root: {root}, binary: {binary}, split: {split}!")
        self.text, self.sentiment = list(), list()
        if root:
            if binary:
                for tree in sst:
                    if tree.to_labeled_lines()[0][0] != 2:
                        self.text.append(
                            preprocess_sst(tree.to_labeled_lines()[0][1])
                        )
                        self.sentiment.append(get_binary_label(tree.to_labeled_lines()[0][0]))
            else:
                for tree in sst:
                    self.text.append(
                        preprocess_sst(tree.to_labeled_lines()[0][1])
                    )
                    self.sentiment.append(
                        tree.to_labeled_lines()[0][0]
                    )
        else:
            if binary:
                for tree in sst:
                    for subtree in tree.to_labeled_lines():
                        if subtree[0] != 0:
                            self.text.append(
                                preprocess_sst(subtree[1])
                            )
                            self.sentiment.append(
                                get_binary_label(subtree[0])
                            )

            else:
                for tree in sst:
                    for subtree in tree.to_labeled_lines():
                        self.text.append(
                            preprocess_sst(subtree[1])
                        )
                        self.sentiment.append(
                            subtree[0]
                        )
        logger.info("Done with data preparation!")

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.text[idx]
        sentiment = self.sentiment[idx]

        return text, sentiment
