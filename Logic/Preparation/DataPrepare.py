import glob
from pathlib import Path

"""
    Preparing the data!
"""

class BERTGhazali_preparation:
    r"""
        BERTGhazali_preparation class is used to prepare the text files and process them for the embeddings production.

        Params:
            - preprocess(`bool`):
              If True, it will preprocess the given texts. Defaults to False.

            - tokenize(`bool`):
              If True, it will tokenize the given texts. Defaults to False.

            - collection(`dict`):
              Dictionary contains the paths of the desired files.


        Returns:
            BERTGhazali_preparation: the preprocessing class

        Examples
        --------
            from DataPrepare import BERTGhazali_preparation

            attrib = BERTGhazali_preparation(tokenize=True, preprocess=False, collection=collections["Source"])

            attrib.run_preparation()

            This will produce tokenized files for the texts inside the given path.

    """
    def __init__(
            self,
            preprocess=False,  # if true, it will preprocess all books.
            tokenize=False,  # if true, it will tokenize all preprocessed files.
            collection=None,  # the collection to be prepared
    ):
        self.bert_model = "aubmindlab/bert-base-arabertv2"
        if collection is None:
            print("Error, Collection must have a valid path.")
            return
        else:
            self.collection = collection

        if tokenize:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model)
            self.tokenize = True

        if preprocess:
            from Logic.Preparation.preprocess import ArabertPreprocessor
            self.pre_process = ArabertPreprocessor(model_name=self.bert_model)
            self.preprocess = True

    def _preprocess(self):
        """
        Preprocessing texts to fit our task.
        """
        original_files = glob.glob(self.collection["Original"] + "*.txt")
        processed_files = self.collection["Processed"]
        for filename in original_files:
            with open(filename, mode="r", encoding="utf8") as f:  # open in readonly mode
                text_original = f.readlines()
            processed_file = open(processed_files + Path(filename).stem + '.txt', mode="w", encoding="utf8")  # + '.txt'
            for line_str in text_original:
                processed_file.write(self.pre_process.preprocess(line_str) + '\n')
            processed_file.close()

    def _tokenize(self):
        """
        Encodes tokens in given files, and saves them.
        """
        processed_files = glob.glob(self.collection["Processed"] + "*.txt")
        tokenized_files = self.collection["Tokenized"]
        for filename in processed_files:
            with open(filename, mode="r", encoding="utf8") as f:  # open in readonly mode
                text_processed = f.read().replace('\n', '')
            tokenized_file = open(tokenized_files + Path(filename).stem + '.txt', mode="w", encoding="utf8")  # + '.txt'
            tokens_encoded = self.tokenizer.encode(text_processed, add_special_tokens=False)
            tokenized_file.write('\n'.join(str(token) for token in tokens_encoded))
            tokenized_file.close()

    def run_preparation(self):
        """
        Calls ```_preprocess``` and ```_tokenize``` to prepare the data.
        """
        self._preprocess()
        self._tokenize()
