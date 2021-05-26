import logging
import math
import os
import sys
import glob
import utils
import threading
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from utils import params
from utils import progress_bar as pb
from bert_cnn import Bert_KCNN
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

collections = {
    "Source": {
        "Name": "Source",
        "Original": "Data/Original/ts1/",
        "Processed": "Data/Processed/ts1/",
        "Tokenized": "Data/Tokenized/ts1/",
        "Embedding": "Data/Embedding/ts1/"
    },
    "Alternative": {
        "Name": "Alternative",
        "Original": "Data/Original/ts2/",
        "Processed": "Data/Processed/ts2/",
        "Tokenized": "Data/Tokenized/ts2/",
        "Embedding": "Data/Embedding/ts2/"
    },
    "Test": {
        "Name": "Test",
        "Original": "Data/Original/ts3/",
        "Processed": "Data/Processed/ts3/",
        "Tokenized": "Data/Tokenized/ts3/",
        "Embedding": "Data/Embedding/ts3/"
    }
}


class BERTGhazali_preparation:
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
            from preprocess import ArabertPreprocessor
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


def _last_dot_index(tokensList, startIndex, lastIndex):
    for i in reversed(range(startIndex, min(lastIndex, len(tokensList)))):
        if tokensList[i] == '48':
            return i
    raise ValueError


class StdoutRedirector(object):
    def __init__(self, text_widget):
        self.text_space = text_widget
        self.msg = ""
        self.x = 0

    def write(self, string):
        self.text_space.insert('end', f'\r{string}')

    def flush(self):
        pass


def _zipdir(path, ziph):
    # Iterate all the directories and files
    for root, dirs, files in os.walk(path):
        # Create a prefix variable with the folder structure inside the path folder.
        # So if a file is at the path directory will be at the root directory of the zip file
        # so the prefix will be empty. If the file belongs to a containing folder of path folder
        # then the prefix will be that folder.
        if root.replace(path, '') == '':
            prefix = ''
        else:
            # Keep the folder structure after the path folder, append a '/' at the end
            # and remome the first character, if it is a '/' in order to have a path like
            # folder1/folder2/file.txt
            prefix = root.replace(path, '') + '/'
            if prefix[0] == '/':
                prefix = prefix[1:]
        for filename in files:
            actual_file_path = root + '/' + filename
            zipped_file_path = prefix + filename
            ziph.write(actual_file_path, zipped_file_path)


def _balancing_routine(Set0, Set1, F1, F):
    if len(Set0) < len(Set1):
        temp = F
        F = F1
        F1 = temp
    over_sampler = RandomOverSampler(sampling_strategy=F)
    under_sampler = RandomUnderSampler(sampling_strategy=F1)
    x_combined_df = pd.concat([Set0, Set1])  # concat the training set
    y_combined_df = pd.to_numeric(x_combined_df['Label'])
    print(f"Combined Dataframe before sampling: {Counter(y_combined_df)}")
    x_under_sample, y_under_sample = under_sampler.fit_resample(x_combined_df, y_combined_df)
    print(f"Combined Under Sampling: {Counter(y_under_sample)}")
    x_combined_sample, y_combined_sample = over_sampler.fit_resample(x_under_sample, y_under_sample)
    print(f"Combined Dataframe after OVER sampling: {Counter(y_combined_sample)}")
    s0_balanced = pd.DataFrame(x_combined_sample[(x_combined_sample['Label'] == 0)])
    s1_balanced = pd.DataFrame(x_combined_sample[(x_combined_sample['Label'] == 1)])
    return s0_balanced, s1_balanced

    # training loop


class BERTGhazali_Attributer:
    r"""
    A BERTGhazali_Attributer class that uses Bert(google) to produce sentence embeddings, and then feeds Conv layer
    as a classifier. It also generates file needed to do so.

    Params:
        bert_model_name(:obj:`str`):
            The model we use to produce embeddings. Defaults to ``bert-large-arabertv2``.
            (a pretrained model from HuggingFace repository).

        text_division_method(:obj:`str`):
            will be later used to decide the text division method. Defaults to Fixed-Size.

            Can be either:
            - :obj:`Fixed-Size`: will split the text into chunks sized of bert's input len.
            - :obj:`Bottom-Up`: builds the longest block possible of whole sentences without reaching bert's input len.
              If we reach ```Bert input len``` tokens and the current symbol is not "." of a sentence ending,
              then we drop the tokens from the end of the block to the last ".".

        text_console(:obj:`Tkinter.Text`):
            will redirect stdout to the text widget entered.

    Returns::
        BERTGhazali_Attributer: the Attributor class

    Examples:

        from bert_ghazali import BERTGhazali_Attributer

        attrib = BERTGhazali_Attributer("aubmindlab/bert-large-arabertv2", "Bottom-Up", tkinter.Text)

    """

    def __init__(
            self,
            bert_model_name,  # the name of the bert model being used. "bert-large-arabertv2" default.
            text_division_method="Fixed-Size",
            text_console=None,
    ):

        self.text_console = text_console
        self.text_division_method = self._bottom_up_division
        self.embeddings_file = "BU"
        self.bert_model_name = "aubmindlab/bert-base-arabertv2"

        if text_division_method == "Fixed-Size":
            self.text_division_method = self._fixed_size_division
            self.embeddings_file = "FS"

        if bert_model_name != "aubmindlab/bert-large-arabertv2":
            logging.warning(
                "Model provided is not [aubmindlab/bert-base-arabertv2]. Assuming you are using a Fine-Tuned Bert, "
                "you can proceed. "
                "else, errors might be occur"
            )
            self.bert_model_name = bert_model_name

        # from transformers import AutoTokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)

    def _encode_tokens(self, tokensToEncode):
        r"""
        Tokenize and prepare the sequence the model. It uses the tokenizer.encode_plus from library,
        and configures special variables to fit our task.

        Params:
            tokensToEncode (:obj:`str`):
                A string contains the tokens to be encoded.

        Returns::
            tensor contains the encoded tokens.
        """
        # Encode the sentence
        encoded = self.tokenizer.encode_plus(
            text=tokensToEncode,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=params['BERT_INPUT_LENGTH'] + 2,  # maximum length of a sentence
            padding='max_length',  # Add [PAD]s
            return_attention_mask=True,  # Generate the attention mask
            return_tensors='pt',  # ask the function to return PyTorch tensors
        )
        return encoded

    def _fixed_size_division(self, tokenized_file, chunkSize):
        r"""
            Splits the text into chunks sequentially, with length of ```Bert input length```.

            Params:
                tokenized_file(:obj:`TextIO`):
                    A text file, contains the tokenized words. (token ids).

                chunkSize(:obj:`int`):
                    The size of ```Bert input length```.

            Returns::
                List of tensors, while each tensor contains the token encodings of the chunk.
        """
        tokensList = tokenized_file.read().splitlines()
        inputForBERT = []
        for index in range(math.ceil((len(tokensList) / chunkSize))):
            chunk = tokensList[index * chunkSize: min(len(tokensList), (index * chunkSize) + chunkSize)]
            chunk_int = list(map(int, chunk))
            inputForBERT.append(self._encode_tokens(self.tokenizer.decode(chunk_int, skip_special_tokens=True)))
        return inputForBERT

    def _bottom_up_division(self, tokenized_file, chunkSize):
        r"""
            Builds the longest block possible of whole sentences without reaching bert's input len.
            If we reach ```Bert input len``` tokens and the current symbol is not "." of a sentence ending,
            then we drop the tokens from the end of the block to the last ".".

            Params:
                tokenized_file(:obj:`TextIO`):
                    A text file, contains the tokenized words. (token ids).

                chunkSize(:obj:`int`):
                    The size of ```Bert input length```.

            Returns::
                List of tensors, while each tensor contains the token encodings of the chunk.
        """
        tokensList = tokenized_file.read().splitlines()
        inputForBERT = []
        stopPoint = 0
        lastDotIndex = 0
        while stopPoint <= len(tokensList):
            try:
                lastDotIndex = _last_dot_index(tokensList, stopPoint, stopPoint + chunkSize)
            except ValueError:
                # the dot is too far away
                lastDotIndex = stopPoint + chunkSize
            finally:
                tempList = tokensList[stopPoint:lastDotIndex]
                stopPoint = lastDotIndex + 1
                chunk_int = list(map(int, tempList))
                inputForBERT.append(self._encode_tokens(self.tokenizer.decode(chunk_int, skip_special_tokens=True)))
        return inputForBERT

    def _bert_embeddings_general(self):
        r"""
            Checks if user already has generated embeddings for the Bert's parameters,
            If so, it will first check if the embeddings extracted already, else, it will extract them.

            Else, it will call ```_bert_embeddings``` to produce embeddings.
        """
        import zipfile
        self.embeddings_file += str(params['BERT_INPUT_LENGTH'])
        embeddings_zip_location = os.getcwd() + r"\Data\PreviousRuns\Embeddings"
        if not os.path.exists(embeddings_zip_location + "\\" + self.embeddings_file + ".zip"):
            import tempfile
            temp_dir = tempfile.TemporaryDirectory()

            self._bert_embeddings(collections["Source"], temp_dir)
            self._bert_embeddings(collections["Alternative"], temp_dir)
            self._bert_embeddings(collections["Test"], temp_dir)

            # save zip file to previous runs.
            zipf = zipfile.ZipFile(r"Data/PreviousRuns/Embeddings/" + self.embeddings_file + '.zip', 'w',
                                   zipfile.ZIP_DEFLATED)
            _zipdir(temp_dir.name + "/Data/", zipf)
            zipf.close()
            temp_dir.cleanup()

        if os.path.exists(os.getcwd() + r"\Data\Embedding\current.txt"):
            with open(os.getcwd() + r"\Data\Embedding\current.txt", 'r') as file:
                if file.readline() != self.embeddings_file:
                    # unzip the right embeddings file into the general Embedding directory
                    with zipfile.ZipFile(os.path.join(embeddings_zip_location, self.embeddings_file + ".zip"),
                                         'r') as zip_ref:
                        zip_ref.extractall(os.getcwd() + r"\Data")
                        with open(os.getcwd() + r"\Data\Embedding\current.txt", 'w') as f:
                            f.write(self.embeddings_file)
        else:
            with zipfile.ZipFile(os.path.join(embeddings_zip_location, self.embeddings_file + ".zip"), 'r') as zip_ref:
                zip_ref.extractall(os.getcwd() + r"\Data")
                with open(os.getcwd() + r"\Data\Embedding\current.txt", 'w') as f:
                    f.write(self.embeddings_file)

    def _bert_embeddings(self, col, output_path):
        r"""
            Generate embeddings for the files, using the selected Bert model.
            Then it save the generated embeddings in pickle files and zips them, to be used later.

            Params:
                col(:obj:`str`):
                    A string contains the location of the relevant set. (Ghazali, Pseudo-Ghazali, Test-Set).

                output_path(:obj:`str`):
                    A string contains the location where the temporary embeddings files will be saved.

        """
        from transformers import AutoModel
        bert_model = AutoModel.from_pretrained(self.bert_model_name, output_hidden_states=True)
        tokenized_files = glob.glob(col["Tokenized"] + "*.txt")
        df = []
        divided = []
        pb["value"] = 0
        i = 0
        if not os.path.exists(output_path.name + "/" + col["Embedding"]):
            os.makedirs(output_path.name + "/" + col["Embedding"])
        if col["Name"] == "Test":
            print(f"Generating Embeddings For {col['Name']}")
            for filename in tokenized_files:
                pb["value"] = 0
                i = 0
                with open(filename, mode="r", encoding="utf8") as f:  # open in readonly mode
                    divided = self.text_division_method(f, params['BERT_INPUT_LENGTH'])
                    sz = len(divided)
                    print(f"Book: {Path(filename).stem}, Total chunks: {sz}. Please wait...", end="")
                    for bert_input in divided:
                        pb['maximum'] = sz
                        pb["value"] = int(pb["value"]) + 1
                        pb.update()
                        with torch.no_grad():
                            outputs = bert_model(**bert_input)
                            i = i + 1
                            # Save the last hidden state tensor, without the [cls] and [sep] tokens.
                            d = {'Embedding': outputs['last_hidden_state'][0][1:-1]}
                            df.append(d)
                    print(" ~DONE!")
                df = pd.DataFrame(df)
                df.to_pickle(output_path.name + "/" + col["Embedding"] + Path(filename).stem + ".pkl")
                df = []
        else:  # Ghazali OR Pseudo-Ghazali texts
            db_name = 'Pseudo-Ghazali.pkl' if col["Name"] == "Alternative" else 'Ghazali.pkl'
            label = 1 if col["Name"] == "Alternative" else 0
            for filename in tokenized_files:
                with open(filename, mode="r", encoding="utf8") as f:  # open in readonly mode
                    divided.extend(self.text_division_method(f, params['BERT_INPUT_LENGTH']))

            sz = len(divided)
            print(f"\nGenerating Embeddings For {col['Name']}, Total chunks: {sz}. Please wait...", end="")
            for bert_input in divided:
                pb['maximum'] = sz
                pb["value"] = int(pb["value"]) + 1
                pb.update()
                with torch.no_grad():
                    outputs = bert_model(**bert_input)
                    i = i + 1
                    # Save the last hidden state tensor, without the [cls] and [sep] tokens.
                    d = {'Embedding': outputs['last_hidden_state'][0][1:-1], 'Label': label}
                    df.append(d)
            print(" ~DONE!")

            df = pd.DataFrame(df)
            df.to_pickle(output_path.name + "/" + col["Embedding"] + db_name)

    def _train(self, net, train_loader, valid_loader, epochs, train_on_gpu, criterion, optimizer, print_every):
        r"""
            Training the net with the data and parameters specified.

            Params:
                - net(:obj:`object`):
                  The configured neural net object
                - train_loader(:obj:`torch.utils.data.DataLoader`):
                  The training data set, contains the embedding tensors and their labels (Ghazali or not).
                - valid_loader(:obj:`torch.utils.data.DataLoader`):
                  The training validation loader, contains examples to validate the training procedure.
                - epochs(:obj:`int`):
                  The number of epochs for training.
                - train_on_gpu(:obj:`boot`):
                  If True, it will train using GPU, Else, it will use CPU.
                - criterion(:obj:`torch.nn`):
                  Loss functions, Defaults to BCELoss.
                - optimizer(:obj:`torch.optim.Adam`):
                  The optimizer chose for the training process, Defaults to Adam.
                - print_every(:obj:`int`):
                  Define the printing frequency. Defaults to 50.

        """
        # Train on GPU is possible
        if train_on_gpu:
            net.cuda()
        counter = 0

        # Start training
        net.train()
        for e in range(epochs):
            # batch loop
            for inputs, labels in train_loader:
                counter += 1

                if train_on_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()

                # zero accumulated gradients
                net.zero_grad()
                # get the output from the model
                output = net(inputs)

                # calculate the loss and perform backprop
                loss = criterion(output.squeeze(), labels.float())
                loss.backward()
                optimizer.step()

                # loss stats
                if counter % print_every == 0:
                    # Get validation loss
                    val_losses = []
                    net.eval()
                    for inputs, labels in valid_loader:

                        if train_on_gpu:
                            inputs, labels = inputs.cuda(), labels.cuda()

                        output = net(inputs)
                        val_loss = criterion(output.squeeze(), labels.float())
                        val_losses.append(val_loss.item())

                    net.train()
                    print("Epoch: {}/{}...".format(e + 1, epochs),
                          "Step: {}...".format(counter),
                          "Loss: {:.6f}...".format(loss.item()),
                          "Val Loss: {:.6f}".format(np.mean(val_losses)))

    def run(self):
        r"""
            Training the net with the data and parameters specified.

            Params:

        """
        lock = threading.Lock()
        lock.acquire()
        original_stdout = sys.stdout
        try:
            sys.stdout = StdoutRedirector(
                self.text_console)
        finally:
            lock.release()

        self._bert_embeddings_general()

        ghazali_df = pd.read_pickle(collections["Source"]["Embedding"] + "Ghazali.pkl")
        pseudo_df = pd.read_pickle(collections["Alternative"]["Embedding"] + "Pseudo-Ghazali.pkl")

        print(f"Total Ghazali's Samples: {len(ghazali_df)}")
        print(f"Total Pseudo-Ghazali's: {len(pseudo_df)}")

        embedded_files = glob.glob(collections["Test"]["Embedding"] + "*.pkl")
        M = np.zeros((params['Niter'], len(embedded_files)))  # 10 num of the books in test set
        Iter = 0
        pb['maximum'] = params['NB_EPOCHS']
        print("**Starting Training Process**")
        s0, s1 = _balancing_routine(ghazali_df, pseudo_df, params['F1'], params['F'])
        # First checking if GPU is available
        train_on_gpu = torch.cuda.is_available()

        if train_on_gpu:
            print('Training on GPU.')
        else:
            print('No GPU available, training on CPU.')

        net = Bert_KCNN(
            embed_num=params['BERT_INPUT_LENGTH'],  # number of words in seq.,
            embed_dim=768,  # The dimension of BERT embeddings.
            class_num=params['OUTPUT_CLASSES'],  # Number of classes for sigmoid output
            kernel_num=params['KERNELS'],  # Number of kernels (filters)
            kernel_sizes=list(params['1D_CONV_KERNEL'].values()),  # Size each kernel
            dropout=params['DROPOUT_RATE'],
            static=True
        )
        print(net)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=params['LEARNING_RATE'])

        while Iter < params['Niter']:
            # s0, s1 = balancing_routine(ghazali_df, pseudo_df, params['F1'], params['F'])
            s0_sampled = s0.sample(math.ceil(len(s0) / 5)).reset_index(drop=True)
            s1_sampled = s1.sample(math.ceil(len(s1) / 5)).reset_index(drop=True)
            emb_train_df = pd.concat([s0_sampled, s1_sampled])

            labels = torch.FloatTensor(emb_train_df["Label"].values)
            embeddings_tensor = torch.stack(emb_train_df['Embedding'].tolist())

            X_train, X_test, y_train, y_test = train_test_split(embeddings_tensor, labels, test_size=0.33,
                                                                shuffle=True, random_state=1)  # shuffle=True,
            test_idx = int(len(X_test) * 0.5)
            val_x, test_x = X_test[:test_idx], X_test[test_idx:]
            val_y, test_y = y_test[:test_idx], y_test[test_idx:]
            del s0_sampled
            del s1_sampled
            del emb_train_df
            train_data = TensorDataset(X_train, y_train)
            test_data = TensorDataset(test_x, test_y)
            valid_data = TensorDataset(val_x, val_y)

            train_loader = DataLoader(train_data, shuffle=True, batch_size=params['BATCH_SIZE'])
            test_loader = DataLoader(test_data, shuffle=True, batch_size=params['BATCH_SIZE'])
            valid_loader = DataLoader(valid_data, shuffle=True, batch_size=params['BATCH_SIZE'])

            print_every = 16
            # -- Train the net --
            self._train(net, train_loader, valid_loader, params['NB_EPOCHS'], train_on_gpu, criterion, optimizer,
                        print_every)

            # Get test data loss and accuracy
            test_losses = []  # track loss
            num_correct = 0
            # -- Evaluate the model --
            net.eval()
            # iterate over test data
            for inputs, labels in test_loader:

                if train_on_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()

                # get predicted outputs
                output = net(inputs)

                # calculate loss
                test_loss = criterion(output.squeeze(), labels.float())
                test_losses.append(test_loss.item())

                # convert output probabilities to predicted class (0 or 1)
                pred = torch.round(output.squeeze())  # rounds to the nearest integer

                # compare predictions to true label
                correct_tensor = pred.eq(labels.float().view_as(pred))
                correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(
                    correct_tensor.cpu().numpy())
                num_correct += np.sum(correct)

            # -- stats! -- ##
            # avg test loss
            print("Test loss: {:.3f}".format(np.mean(test_losses)))

            # accuracy over all test data
            test_acc = num_correct / len(test_loader.dataset)
            print("Test accuracy: {:.3f}".format(test_acc))

            if test_acc <= params['ACCURACY_THRESHOLD']:
                continue

            i = 0
            for filename in embedded_files:
                emb_file = pd.read_pickle(collections["Test"]["Embedding"] + Path(filename).stem + ".pkl")
                emb_file = pd.DataFrame(emb_file)
                embeddings_test = torch.stack(emb_file['Embedding'].tolist())
                # embeddings_test = DataLoader(emb_file['Embedding'], shuffle=True, batch_size=params['BATCH_SIZE'])
                net.eval()

                # batch_size = embeddings_test.size(0)

                if train_on_gpu:
                    feature_tensor = embeddings_test.cuda()
                else:
                    feature_tensor = embeddings_test

                with torch.no_grad():  # deactivate autograd engine to reduce memory usage and speed up computations
                    # get the output from the model
                    output = net(feature_tensor)
                    M[Iter][i] = round(np.mean(np.array(output.cpu()), axis=0)[0], 4)
                print(f"Iter [{Iter}], File [{i}]: {M[Iter][i]}")
                i += 1
            Iter += 1
            # progress_bar["value"] = int(progress_bar["value"]) + 1
        print("**Finished Training**")
        pb["value"] = 0
        sys.stdout = original_stdout

        utils.heat_map = M
