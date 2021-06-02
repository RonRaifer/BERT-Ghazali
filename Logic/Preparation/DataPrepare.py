import glob
from pathlib import Path


class BERTGhazali_preparation:
    r"""
        BERTGhazali_preparation class is used to prepare the text files and process them for the embeddings production.

        Params:
            - preprocess(:obj:`bool`):
              If True, it will preprocess the given texts. Defaults to False.

            - tokenize(:obj:`bool`):
              If True, it will tokenize the given texts. Defaults to False.


        Returns:
            BERTGhazali_preparation: the preprocessing class

        Examples
        --------
            from bert_ghazali import BERTGhazali_preparation

            attrib = BERTGhazali_preparation(tokenize=True, preprocess=False)

            attrib.tokenize(collections["Source"])

            This will produce tokenized files for the texts inside the given path.

    """
    def __init__(
            self,
            preprocess=False,  # if true, it will preprocess all books.
            tokenize=False,  # if true, it will tokenize all preprocessed files.
    ):
        self.bert_model_for_prep = "aubmindlab/bert-base-arabertv2"
        self.bert_model_for_tokenizer = "aubmindlab/bert-base-arabertv2"
        if tokenize:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_for_tokenizer)
            self.tokenize = True

        if preprocess:
            from Logic.Preparation.preprocess import ArabertPreprocessor
            self.pre_process = ArabertPreprocessor(model_name=self.bert_model_for_prep)
            self.preprocess = True

    def preprocess(self, col):
        original_files = glob.glob(col["Original"] + "*.txt")
        processed_files = col["Processed"]
        for filename in original_files:
            with open(filename, mode="r", encoding="utf8") as f:  # open in readonly mode
                text_original = f.readlines()
            processed_file = open(processed_files + Path(filename).stem + '.txt', mode="w", encoding="utf8")  # + '.txt'
            for line_str in text_original:
                processed_file.write(self.pre_process.preprocess(line_str) + '\n')
            processed_file.close()

    def tokenize(self, col):
        processed_files = glob.glob(col["Processed"] + "*.txt")
        tokenized_files = col["Tokenized"]
        for filename in processed_files:
            with open(filename, mode="r", encoding="utf8") as f:  # open in readonly mode
                text_processed = f.read().replace('\n', '')
            tokenized_file = open(tokenized_files + Path(filename).stem + '.txt', mode="w", encoding="utf8")  # + '.txt'
            tokens_encoded = self.tokenizer.encode(text_processed, add_special_tokens=False)
            tokenized_file.write('\n'.join(str(token) for token in tokens_encoded))
            tokenized_file.close()
