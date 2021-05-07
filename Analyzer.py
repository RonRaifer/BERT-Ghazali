import glob
import json
import math
import os
import threading
import time
import zipfile

import utils
from GuiFiles import NewGui
from kim_cnn import KimCNN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from collections import Counter
from pathlib import Path
import pandas as pd
import tensorflow as tf
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tkinter as tk
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.keras import optimizers
from transformers import AutoTokenizer, AutoModel
from Model import TEXT_MODEL
from preprocess import ArabertPreprocessor
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from zipfile import ZipFile

model_name = "aubmindlab/bert-base-arabertv2"
tuned_model = "TunedGazaliBert"
pre_process = ArabertPreprocessor(model_name=model_name)
bert_model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

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


def start_preprocess(col):
    original_files = glob.glob(col["Original"] + "*.txt")
    processed_files = col["Processed"]
    for filename in original_files:
        with open(filename, mode="r", encoding="utf8") as f:  # open in readonly mode
            text_original = f.readlines()
        processed_file = open(processed_files + Path(filename).stem + '.txt', mode="w", encoding="utf8")  # + '.txt'
        for line_str in text_original:
            processed_file.write(pre_process.preprocess(line_str) + '\n')
        processed_file.close()


# start_preprocess(collections["Source"])
# start_preprocess(collections["Alternative"])
# start_preprocess(collections["Test"])

def encode_tokens(tokensToEncode):
    # Encode the sentence
    encoded = tokenizer.encode_plus(
        text=tokensToEncode,  # the sentence to be encoded
        add_special_tokens=True,  # Add [CLS] and [SEP]
        max_length=utils.params['BERT_INPUT_LENGTH']+2,  # maximum length of a sentence
        padding='max_length',  # Add [PAD]s
        return_attention_mask=True,  # Generate the attention mask
        return_tensors='pt',  # ask the function to return PyTorch tensors
    )
    return encoded


def tokenize(col):
    processed_files = glob.glob(col["Processed"] + "*.txt")
    tokenized_files = col["Tokenized"]
    for filename in processed_files:
        with open(filename, mode="r", encoding="utf8") as f:  # open in readonly mode
            text_processed = f.read().replace('\n', '')
        tokenized_file = open(tokenized_files + Path(filename).stem + '.txt', mode="w", encoding="utf8")  # + '.txt'
        tokens_encoded = tokenizer.encode(text_processed, add_special_tokens=False)
        tokenized_file.write('\n'.join(str(token) for token in tokens_encoded))
        tokenized_file.close()


def fixed_size_division(tokenized_file, chunkSize):
    tokensList = tokenized_file.read().splitlines()
    inputForBERT = []
    for index in range(math.ceil((len(tokensList) / chunkSize))):
        chunk = tokensList[index * chunkSize: min(len(tokensList), (index * chunkSize) + chunkSize)]
        chunk_int = list(map(int, chunk))
        inputForBERT.append(encode_tokens(tokenizer.decode(chunk_int, skip_special_tokens=True)))
    return inputForBERT


def last_dot_index(tokensList, startIndex, lastIndex):
    for i in reversed(range(startIndex, min(lastIndex, len(tokensList)))):
        if tokensList[i] == '48':
            return i
    raise ValueError


def bottom_up_division(tokenized_file, chunkSize):
    tokensList = tokenized_file.read().splitlines()
    inputForBERT = []
    stopPoint = 0
    lastDotIndex = 0
    while stopPoint <= len(tokensList):
        try:
            lastDotIndex = last_dot_index(tokensList, stopPoint, stopPoint + chunkSize)
        except ValueError:
            # the dot is too far away
            lastDotIndex = stopPoint + chunkSize
        finally:
            tempList = tokensList[stopPoint:lastDotIndex]
            stopPoint = lastDotIndex + 1
            chunk_int = list(map(int, tempList))
            inputForBERT.append(encode_tokens(tokenizer.decode(chunk_int, skip_special_tokens=True)))
    return inputForBERT


# tokenize(collections["Source"])
# tokenize(collections["Alternative"])
# tokenize(collections["Test"])

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def zipdir(path, ziph):
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
            if (prefix[0] == '/'):
                prefix = prefix[1:]
        for filename in files:
            actual_file_path = root + '/' + filename
            zipped_file_path = prefix + filename
            ziph.write(actual_file_path, zipped_file_path)


def bert_embeddings_general():
    from utils import params
    import os
    embeddings_file = "FS" if params['TEXT_DIVISION_METHOD'] == "Fixed-Size" else "BU"
    embeddings_file += str(params['BERT_INPUT_LENGTH'])
    embeddings_zip_location = os.getcwd() + r"\Data\PreviousRuns\Embeddings"
    if not os.path.exists(embeddings_zip_location + "\\" + embeddings_file + ".zip"):
        import tempfile
        tmpdirname = tempfile.TemporaryDirectory()

        division_method = fixed_size_division if params['TEXT_DIVISION_METHOD'] == "Fixed-Size" else bottom_up_division
        bert_embeddings(collections["Source"], division_method, params['BERT_INPUT_LENGTH'], tmpdirname)
        bert_embeddings(collections["Alternative"], division_method, params['BERT_INPUT_LENGTH'], tmpdirname)
        bert_embeddings(collections["Test"], division_method, params['BERT_INPUT_LENGTH'], tmpdirname)
        from zipfile import ZipFile
        import os
        from os.path import basename

        # save zip file to previous runs.
        zipf = zipfile.ZipFile(r"Data/PreviousRuns/Embeddings/" + embeddings_file + '.zip', 'w', zipfile.ZIP_DEFLATED)
        zipdir(tmpdirname.name + "/Data/", zipf)
        zipf.close()
        #tmpdirname.cleanup()

    # unzip the right embeddings file into the general Embedding directory
    with zipfile.ZipFile(os.path.join(embeddings_zip_location, embeddings_file + ".zip"), 'r') as zip_ref:
        zip_ref.extractall(os.getcwd() + r"\Data")


def bert_embeddings(col, division_method, input_len, output_path):
    tokenized_files = glob.glob(col["Tokenized"] + "*.txt")
    df = []
    divided = []
    i = 0
    if not os.path.exists(output_path.name + "/" + col["Embedding"]):
        os.makedirs(output_path.name + "/" + col["Embedding"])
    if col["Name"] == "Test":
        for filename in tokenized_files:
            with open(filename, mode="r", encoding="utf8") as f:  # open in readonly mode
                divided = division_method(f, input_len)
                print(filename)
                i = 0
                for bert_input in divided:
                    sz = len(divided)
                    with torch.no_grad():
                        outputs = bert_model(**bert_input)
                        # sentence_embedding = mean_pooling(outputs, bert_input['attention_mask'])
                        # `outputs['hidden_states'][-2][0]` is a tensor with shape [512 x 768]
                        # Calculate the average of all 510 token vectors. # without CLS!
                        # sentence_embedding = torch.mean(outputs['hidden_states'][-2][0][1:-1], dim=0)
                        i = i + 1
                        print(f'\r{i} chunks of {sz}', end="", flush=True)
                        # d = {'Embedding': outputs[0][0]}
                        # d = {'Embedding': outputs['pooler_output']}
                        d = {'Embedding': outputs['last_hidden_state'][0]}
                        df.append(d)
            df = pd.DataFrame(df)
            df.to_pickle(output_path.name + "/" + col["Embedding"] + Path(filename).stem + ".pkl")
            df = []
    else:  # source\alternative
        db_name = 'Pseudo-Ghazali.pkl' if col["Name"] == "Alternative" else 'Ghazali.pkl'
        label = 1 if col["Name"] == "Alternative" else 0
        for filename in tokenized_files:
            with open(filename, mode="r", encoding="utf8") as f:  # open in readonly mode
                divided.extend(division_method(f, input_len))
                print(filename)
        for bert_input in divided:
            sz = len(divided)
            with torch.no_grad():
                outputs = bert_model(**bert_input)
                # sentence_embedding = mean_pooling(outputs, bert_input['attention_mask'])
                # `outputs['hidden_states'][-2][0]` is a tensor with shape [512 x 768]
                # Calculate the average of all 510 token vectors. # without CLS!
                # sentence_embedding = torch.mean(outputs['hidden_states'][-2][0][1:-1], dim=0)
                i = i + 1
                print(f'\r{i} chunks of {sz}', end="", flush=True)
                # d = {'Embedding': outputs[0][0], 'Label': label}  # label 0 ghazali, 1 if pseudo
                # d = {'Embedding': outputs['pooler_output'], 'Label': label}  # label 0 ghazali, 1 if pseudo
                d = {'Embedding': outputs['last_hidden_state'][0], 'Label': label}
                df.append(d)

        df = pd.DataFrame(df)

        df.to_pickle(output_path.name + "/" + col["Embedding"] + db_name)


def balancing_routine(Set0, Set1, F1, F):
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
    ##NEW
    # s0_balanced = pd.concat([s0_balanced, s0_balanced])
    # s1_balanced = pd.concat([s1_balanced, s1_balanced])
    ##END NEW
    s0_sampled = s0_balanced.sample(math.ceil(len(s0_balanced) / 5)).reset_index(drop=True)
    s1_sampled = s1_balanced.sample(math.ceil(len(s1_balanced) / 5)).reset_index(drop=True)
    # s0_sampled = pd.concat([s0_sampled, s0_sampled])
    # s1_sampled = pd.concat([s1_sampled, s1_sampled])
    return s0_sampled, s1_sampled


''' 
Combined Dataframe before sampling: Counter({0: 3311, 1: 443})
Combined Dataframe after OVER sampling: Counter({0: 3311, 1: 1986})
Combined Over&Under Sampling: Counter({0: 2482, 1: 1986})
'''

import sys


class StdoutRedirector(object):
    def __init__(self, text_widget):
        self.text_space = text_widget
        self.msg = ""
        self.x = 0

    def write(self, string):
        # self.text_space.insert('end', f'\r{string}')
        if not format(string).startswith("\b\b\b"):
            self.text_space.insert("end", string)
        # self.text_space.delete("1.0", tk.END)

    def flush(self):
        # self.text_space.delete("end-50c linestart", "end")
        self.msg = self.text_space.get("end-1l", "end")
        # self.msg = self.text_space.get("end-1l", "end")
        # x = 'end-%dc' % len(msg)
        self.x = len(self.msg)
        # x = f'{en}{len(msg)}{c}'
        # x = 'end-' + str(len(msg)) + 'c'
        # x = "end-85c"
        y = "end-2l"
        # r"end-{}c".format(self.x)
        self.text_space.delete(y, r"end-{}c".format(self.x))
        # self.text_space.insert("end-3l", u"\n{}".format(msg))
        # self.text_space.insert('end-1l', f'\r{string}', end="", flush=True)
        # self.text_space.insert('end-1l', f'\r')


# NewGui.vp_start_gui()
def run(text_console):
    import sys
    lock = threading.Lock()
    lock.acquire()
    try:
        sys.stdout = StdoutRedirector(
            text_console)
    finally:
        lock.release()

    ghazali_df = pd.read_pickle(collections["Source"]["Embedding"] + "Ghazali.pkl")
    pseudo_df = pd.read_pickle(collections["Alternative"]["Embedding"] + "Pseudo-Ghazali.pkl")

    print(f'Samples Class 0 (Ghazali): {len(ghazali_df)}')
    print(f'Samples Class 1 (Pseudo-Ghazali): {len(pseudo_df)}')

    embedded_files = glob.glob(collections["Test"]["Embedding"] + "*.pkl")

    # save_results()
    # 'BERT_INPUT_LENGTH': 510,
    # 'TEXT_DIVISION_METHOD': 'Fixed-Size',
    # '1D_CONV_KERNEL': {1: 3, 2: 6, 3: 12}
    # 'POOLING_SIZE': 500,
    # 'STRIDES': 1,
    # 'ACTIVATION_FUNC': 'Relu',

    def batchOutput(batch, logs):
        pass
        # print("Finished batch: " + str(batch))
        # print(logs)

    batchLogCallback = tf.keras.callbacks.LambdaCallback(on_batch_end=batchOutput)

    from utils import params
    M = np.zeros((params['Niter'], 10))  # 10 num of the books in test set
    text_model = TEXT_MODEL(cnn_filters=params['CNN_FILTERS'],
                            dnn_units=params['DNN_UNITS'],
                            model_output_classes=params['OUTPUT_CLASSES'],
                            dropout_rate=params['DROPOUT_RATE'])
    adam = optimizers.Adam(learning_rate=params['LEARNING_RATE'], decay=params['DECAY'], beta_1=params['MOMENTUM'],
                           beta_2=0.999, amsgrad=False)
    text_model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                       optimizer=adam,
                       metrics=["accuracy"])
    Iter = 0
    while Iter < params['Niter']:
        s0, s1 = balancing_routine(ghazali_df, pseudo_df, params['F1'], params['F'])
        emb_train_df = pd.concat([s0, s1])
        labels = emb_train_df.pop('Label')
        embeddings = emb_train_df.pop('Embedding')
        X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.33,
                                                            shuffle=True)  # shuffle=True,
        training_dataset = tf.data.Dataset.from_tensor_slices(
            ([tf.convert_to_tensor(s) for s in X_train], [tf.convert_to_tensor(s) for s in y_train]))
        validation_dataset = tf.data.Dataset.from_tensor_slices(
            ([tf.convert_to_tensor(s) for s in X_test], [tf.convert_to_tensor(s) for s in y_test]))
        '''
        training_dataset = tf.data.Dataset.from_tensor_slices(
            ([tf.convert_to_tensor(s.reshape(1, 768)) for s in X_train], y_train.values))
        validation_dataset = tf.data.Dataset.from_tensor_slices(
            ([tf.convert_to_tensor(s.reshape(1, 768)) for s in X_test], y_test.values))
        '''
        training_dataset = training_dataset.batch(params['BATCH_SIZE'], drop_remainder=True)
        validation_dataset = validation_dataset.batch(params['BATCH_SIZE'], drop_remainder=True)

        del s0
        del s1
        del emb_train_df

        text_model.fit(training_dataset, epochs=params['NB_EPOCHS'],
                       validation_data=validation_dataset, callbacks=[batchLogCallback])
        loss, acc = text_model.evaluate(validation_dataset, callbacks=[batchLogCallback])
        if acc < params['ACCURACY_THRESHOLD']:
            print(f"Discarded CNN with accuracy {acc}")
            continue
        i = 0
        for filename in embedded_files:
            emb_file = pd.read_pickle(collections["Test"]["Embedding"] + Path(filename).stem + ".pkl")
            emb_df = pd.DataFrame(emb_file)
            emb_pred_df = emb_df.pop('Embedding')
            to_predict = tf.data.Dataset.from_tensor_slices([tf.convert_to_tensor(s) for s in emb_pred_df])
            predict = text_model.predict(to_predict.batch(params['BATCH_SIZE'], drop_remainder=True))
            # predict = text_model.predict(to_predict)
            print(f"File Num: {i}, name: " + Path(filename).stem)
            M[Iter][i] = np.mean(predict, axis=0)[1]
            # M[Iter][i] = np.mean(predict, axis=0)
            print(M[Iter])
            i += 1

        Iter += 1
        import utils
        utils.heat_map = M


def targets_to_tensor(df, target_columns):
    return torch.tensor(df[target_columns].values, dtype=torch.float32)


def run2(text_console):
    import sys
    lock = threading.Lock()
    lock.acquire()
    try:
        sys.stdout = StdoutRedirector(
            text_console)
    finally:
        lock.release()

    bert_embeddings_general()

    import torch.utils.data as data_utils
    ghazali_df = pd.read_pickle(collections["Source"]["Embedding"] + "Ghazali.pkl")
    pseudo_df = pd.read_pickle(collections["Alternative"]["Embedding"] + "Pseudo-Ghazali.pkl")

    print(f'Samples Class 0 (Ghazali): {len(ghazali_df)}')
    print(f'Samples Class 1 (Pseudo-Ghazali): {len(pseudo_df)}')

    embedded_files = glob.glob(collections["Test"]["Embedding"] + "*.pkl")
    import torch.nn as nn
    # save_results()
    # 'BERT_INPUT_LENGTH': 510,
    # 'TEXT_DIVISION_METHOD': 'Fixed-Size',
    # '1D_CONV_KERNEL': {1: 3, 2: 6, 3: 12}
    # 'POOLING_SIZE': 500,
    # 'STRIDES': 1,
    # 'ACTIVATION_FUNC': 'Relu',

    def batchOutput(batch, logs):
        pass
        # print("Finished batch: " + str(batch))
        # print(logs)

    batchLogCallback = tf.keras.callbacks.LambdaCallback(on_batch_end=batchOutput)

    from utils import params
    M = np.zeros((params['Niter'], 10))  # 10 num of the books in test set

    Iter = 0
    while Iter < params['Niter']:
        s0, s1 = balancing_routine(ghazali_df, pseudo_df, params['F1'], params['F'])
        emb_train_df = pd.concat([s0, s1])
        # labels = emb_train_df['Label'].values
        # labels = torch.Tensor([np.asarray(s) for s in labels])
        labels = targets_to_tensor(emb_train_df, 'Label')
        labels = torch.unsqueeze(labels, 1)
        # torch.reshape(labels, ())
        # labels = torch.stack([torch.tensor(s, dtype=torch.float32) for s in emb_train_df['Label'].values])
        # labels = emb_train_df.pop('Label')
        # embeddings = emb_train_df.pop('Embedding')
        # embeddings = emb_train_df.pop('Embedding').values
        embeddings_tensor = torch.stack(emb_train_df['Embedding'].tolist())
        # embeddings_tensor = emb_train_df['Embedding'].values
        # embeddings_tensor = torch.Tensor([np.asarray(s) for s in embeddings_tensor])
        # = torch.Tensor([torch.Tensor(X) for X in emb_train_df['Embedding'].values])
        X_train, X_test, y_train, y_test = train_test_split(embeddings_tensor, labels, test_size=0.33,
                                                            shuffle=True)  # shuffle=True,
        del s0
        del s1
        del emb_train_df
        # training_dataset = tf.data.Dataset.from_tensor_slices(
        #     ([tf.convert_to_tensor(s) for s in X_train], [tf.convert_to_tensor(s) for s in y_train]))
        # validation_dataset = tf.data.Dataset.from_tensor_slices(
        #     ([tf.convert_to_tensor(s) for s in X_test], [tf.convert_to_tensor(s) for s in y_test]))
        # training_dataset = data_utils.TensorDataset(X_train, y_train)
        # validation_dataset = data_utils.TensorDataset(X_test, y_test)

        embed_num = X_train.shape[1]  # number of words in seq
        embed_dim = X_train.shape[2]  # 768
        class_num = 1  # y_train.shape[1]
        kernel_num = 3
        kernel_sizes = [3, 6, 12]
        dropout = 0.3
        static = True

        model = KimCNN(
            embed_num=embed_num,
            embed_dim=embed_dim,
            class_num=class_num,
            kernel_num=kernel_num,
            kernel_sizes=kernel_sizes,
            dropout=dropout,
            static=static,
        )

        n_epochs = params['NB_EPOCHS']
        batch_size = 32
        lr = 0.001
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.BCELoss()

        train_losses, val_losses = [], []

        def generate_batch_data(x, y, batch_size):
            i, batch = 0, 0
            for batch, i in enumerate(range(0, len(x) - batch_size, batch_size), 1):
                x_batch = x[i: i + batch_size]
                y_batch = y[i: i + batch_size]
                yield x_batch, y_batch, batch
            if i + batch_size < len(x):
                yield x[i + batch_size:], y[i + batch_size:], batch + 1
            if batch == 0:
                yield x, y, 1

        total_accuracy = .0
        for epoch in range(n_epochs):
            start_time = time.time()
            train_loss = 0
            model.train(True)
            for x_batch, y_batch, batch in generate_batch_data(X_train, y_train, batch_size):
                y_pred = model(x_batch)
                optimizer.zero_grad()
                loss = loss_fn(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= batch
            train_losses.append(train_loss)
            elapsed = time.time() - start_time
            model.eval()  # disable dropout for deterministic output
            with torch.no_grad():  # deactivate autograd engine to reduce memory usage and speed up computations
                val_loss, batch = 0, 1
                acc = .0
                for x_batch, y_batch, batch in generate_batch_data(X_test, y_test, batch_size):
                    y_pred = model(x_batch)
                    loss = loss_fn(y_pred, y_batch)
                    val_loss += loss.item()
                    # acc += (y_pred.round() == y_batch).sum()/float(y_pred.shape[0])
                    acc += (y_pred.round() == y_batch).sum() / float(y_pred.shape[0])

                # acc_calculated = (acc / (batch + 1)).detach().item()
                acc_calculated = (acc / (batch + 1)).item()
                total_accuracy += acc_calculated
                val_loss /= batch
                val_losses.append(val_loss)

            print(
                "Epoch %d Train loss: %.2f. Validation loss: %.2f. Elapsed time: %.2fs. Accuracy: %.5f."
                % (epoch + 1, train_losses[-1], val_losses[-1], elapsed, acc_calculated)
            )
        total_accuracy = total_accuracy / float(n_epochs)
        print(f'\nTotal acc: {total_accuracy}')

        if total_accuracy <= params['ACCURACY_THRESHOLD']:
            continue

        i = 0
        for filename in embedded_files:
            emb_file = pd.read_pickle(collections["Test"]["Embedding"] + Path(filename).stem + ".pkl")
            emb_file = pd.DataFrame(emb_file)
            embeddings_test = torch.stack(emb_file['Embedding'].tolist())
            model.eval()  # disable dropout for deterministic output
            with torch.no_grad():  # deactivate autograd engine to reduce memory usage and speed up computations
                y_preds = []
                batch = 0
                for x_batch, y_batch, batch in generate_batch_data(embeddings_test, y_test, batch_size):
                    y_pred = model(x_batch)
                    y_preds.extend(y_pred.cpu().numpy().tolist())
                y_preds_np = np.array(y_preds)
                M[Iter][i] = round(np.mean(y_preds_np, axis=0)[0], 4)
            print(f"Iter [{Iter}], File [{i}]: {M[Iter][i]}")
            i += 1
        Iter += 1

    import utils
    utils.heat_map = M


# run2()

def produce_heatmap():
    import seaborn as sns
    # utils.heat_map_plot = plt.figure(figsize=(11, 9), dpi=100)
    utils.heat_map_plot, ax = plt.subplots(figsize=(11, 9), dpi=100)
    # color map
    cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
    sns.heatmap(utils.heat_map, annot=True, cmap=cmap, fmt=".2f",
                linewidth=0.3, cbar_kws={"shrink": .8}, ax=ax)
    # yticks
    # plt.yticks(rotation=0)


def produce_kmeans():
    avgdArr = np.average(utils.heat_map, axis=0)
    kmeans = KMeans(
        init='k-means++',
        n_clusters=2,
        n_init=10,
        max_iter=300,
        random_state=42
    )

    # res = kmeans.fit(transposedMat)
    res2 = kmeans.fit(
        avgdArr.reshape(-1, 1))  # res and res2 are the same, we'll use res2 cuz it has more understandable dimensions.
    res80 = kmeans.predict(avgdArr.reshape(-1, 1))
    centroids = res2.cluster_centers_
    X = res2.labels_
    u_labels = np.unique(res2.labels_)
    centroids = res2.cluster_centers_
    # matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    # kmeans_plot = plt.figure(figsize=(8, 6))
    utils.kmeans_plot = plt.figure(figsize=(6, 5), dpi=100)
    ax = plt.axes()
    x = np.linspace(-1, 11, 100)
    ax.plot(x, x * 0.00000000000001 + centroids[0][0])
    ax.plot(x, x * 0.00000000000001 + centroids[1][0])
    plt.scatter(range(0, 10), avgdArr, c=res80, s=50, cmap='viridis')

    # plt.scatter(centroids[0, :], centroids[1, :], c='r', s=100)
    # plt.scatter(range(0, 10), avgdArr)
    # plt.legend()
    # kmeans_plot.show()
    silVal = silhouette_score(avgdArr.reshape(-1, 1), res2.labels_)

    anchorGhazaliLabel = res2.labels_[0]
    anchorPseudoGhazaliLabel = res2.labels_[8]

    silhouetteDemandSatisfied = silVal > utils.params['SILHOUETTE_THRESHOLD']
    anchorsDemandSatisfied = anchorGhazaliLabel != anchorPseudoGhazaliLabel
    if not silhouetteDemandSatisfied or not anchorsDemandSatisfied:
        print("the given configurations yield unstable classification values.")
        if not silhouetteDemandSatisfied:
            print("\tsilhouette threshold is: " + str(
                utils.params['SILHOUETTE_THRESHOLD']) + ", actual silhouette value: " + str(silVal))
        if not anchorsDemandSatisfied:
            print("\tanchors belong to the same cluster")
    else:
        print("succesfully classified, the labels are: " + str(res2.labels_))


def show_results():
    if utils.heat_map is None:
        utils.heat_map = np.load(os.getcwd() + r"\Data\PreviousRuns\\" + utils.params['Name'] + ".npy")
    produce_kmeans()
    produce_heatmap()

    # heat_map = np.load('C:/Users/Ron/Desktop/BERT-Ghazali/Data/Mat.npy')
    # hm = heat_map

    from yellowbrick.cluster import SilhouetteVisualizer

    # fig, ax = plt.subplots(1, 1, figsize=(6,4))
    # km = KMeans(
    #     init="random",
    #     n_clusters=2,
    #     n_init=10,
    #     max_iter=300,
    #     random_state=42
    # )
    ## km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
    ## q, mod = divmod(i, 2)
    # visualizer = SilhouetteVisualizer(km, colors='yellowbrick')
    # visualizer.fit(avgdArr.reshape(-1,1))
    # visualizer.show()


def read_json():
    data_base = []
    with open('Data/PreviousRuns/PreviousRuns.json', 'r') as json_file:
        try:
            data_base = json.load(json_file)
            print('loaded that: ', data_base)
        except Exception as e:
            print("got %s on json.load()" % e)
    return data_base


def save_results():
    import utils
    data_base = read_json()
    if data_base is None:
        data_base = [utils.params]
    else:
        data_base.append(utils.params)
    with open('Data/PreviousRuns/PreviousRuns.json', 'w') as f:
        json.dump(data_base, f, indent=4)


'''
utils.heat_map = np.load(os.getcwd() + r'\Data\Mat.npy')
import tempfile

# outfile = TemporaryDirectory
# np.save(outfile, utils.heat_map)

tmpdirname = tempfile.TemporaryDirectory()
print('created temporary directory', tmpdirname)
np.save(tmpdirname.name + r'\Mat.npy', utils.heat_map)


with ZipFile('sample2.zip', 'w') as zipObj2:
    for folderName, subfolders, filenames in os.walk(tmpdirname.name):
        for filename in filenames:
            # create complete filepath of file in directory
            filePath = os.path.join(folderName, filename)
            # Add file to zip
            zipObj2.write(filePath, os.path.basename(filePath))

exit()
'''
