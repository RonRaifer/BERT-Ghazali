import glob
import math
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from collections import Counter
from pathlib import Path
import pandas as pd
import tensorflow as tf
import torch
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.keras import optimizers
from transformers import AutoTokenizer, AutoModel
from Model import TEXT_MODEL
from preprocess import ArabertPreprocessor
from sklearn.cluster import KMeans

model_name = "aubmindlab/bert-base-arabertv2"
pre_process = ArabertPreprocessor(model_name=model_name)
bert_model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
Niter = 2

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
        max_length=510,  # maximum length of a sentence
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
        if len(chunk_int) > 510:
            print("error")
            exit()
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

def bert_embeddings(col):
    tokenized_files = glob.glob(col["Tokenized"] + "*.txt")
    df = []
    divided = []
    i = 0
    if col["Name"] == "Test":
        for filename in tokenized_files:
            with open(filename, mode="r", encoding="utf8") as f:  # open in readonly mode
                divided = (fixed_size_division(f, 510))
                print(filename)
                i = 0
                for bert_input in divided:
                    sz = len(divided)
                    with torch.no_grad():
                        outputs = bert_model(**bert_input)
                        i = i + 1
                        print(f'\r{i} chunks of {sz}', end="", flush=True)
                        d = {'Embedding': outputs[0][0]}
                        df.append(d)
            df = pd.DataFrame(df)
            df.to_pickle(col["Embedding"] + Path(filename).stem + ".pkl")
            df = []
    else:
        db_name = 'Pseudo-Ghazali.pkl' if col["Name"] == "Alternative" else 'Ghazali.pkl'
        label = 1 if col["Name"] == "Alternative" else 0
        for filename in tokenized_files:
            with open(filename, mode="r", encoding="utf8") as f:  # open in readonly mode
                divided.extend(fixed_size_division(f, 510))
                print(filename)
        for bert_input in divided:
            sz = len(divided)
            with torch.no_grad():
                outputs = bert_model(**bert_input)
                i = i + 1
                print(f'\r{i} chunks of {sz}', end="", flush=True)
                d = {'Embedding': outputs[0][0], 'Label': label}  # label 0 ghazali, 1 if pseudo
                df.append(d)

        df = pd.DataFrame(df)
        df.to_pickle(col["Embedding"] + db_name)


# bert_embeddings(collections["Source"])
# bert_embeddings(collections["Alternative"])
# bert_embeddings(collections["Test"])


def balancing_routine(Set0, Set1, F1, F):
    over_sampler = RandomOverSampler(sampling_strategy=F)
    under_sampler = RandomUnderSampler(sampling_strategy=F1)
    x_combined_df = pd.concat([Set0, Set1])  # concat the training set
    y_combined_df = pd.to_numeric(x_combined_df['Label'])
    print(f"Combined Dataframe before sampling: {Counter(y_combined_df)}")
    x_over_sample, y_over_sample = over_sampler.fit_resample(x_combined_df, y_combined_df)
    print(f"Combined Dataframe after OVER sampling: {Counter(y_over_sample)}")
    x_combined_sample, y_combined_sample = under_sampler.fit_resample(x_over_sample, y_over_sample)
    print(f"Combined Over&Under Sampling: {Counter(y_combined_sample)}")
    s0_balanced = pd.DataFrame(x_combined_sample[(x_combined_sample['Label'] == 0)])
    s1_balanced = pd.DataFrame(x_combined_sample[(x_combined_sample['Label'] == 1)])
    s0_sampled = s0_balanced.sample(math.floor(len(s0_balanced) / Niter))
    s1_sampled = s1_balanced.sample(math.floor(len(s1_balanced) / Niter))
    return s0_sampled, s1_sampled


def train_test_split_tensors(X, y, **options):
    """
    encapsulation for the sklearn.model_selection.train_test_split function
    in order to split tensors objects and return tensors as output

    :param X: tensorflow.Tensor object
    :param y: tensorflow.Tensor object
    :dict **options: typical sklearn options are available, such as test_size and train_size
    """

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X.numpy(), y.numpy(), **options)

    X_train, X_test = tf.constant(X_train), tf.constant(X_test)
    y_train, y_test = tf.constant(y_train), tf.constant(y_test)

    del (train_test_split)

    return X_train, X_test, y_train, y_test


def cvt_to_tensor(df):
    return torch.tensor(df.values, dtype=torch.float32)


ghazali_df = pd.read_pickle(collections["Source"]["Embedding"] + "Ghazali.pkl")
pseudo_df = pd.read_pickle(collections["Alternative"]["Embedding"] + "Pseudo-Ghazali.pkl")

print(f'Samples Class 0 (Ghazali): {len(ghazali_df)}')
print(f'Samples Class 1 (Pseudo-Ghazali): {len(pseudo_df)}')
Iter = 0
embedded_files = glob.glob(collections["Test"]["Embedding"] + "*.pkl")
M = np.zeros((Niter, 10))   # 10 num of the books in test set
BATCH_SIZE = 30

CNN_FILTERS = 500
DNN_UNITS = 512
OUTPUT_CLASSES = 2
DROPOUT_RATE = 0.3
NB_EPOCHS = 5
Accuracy_threshold = 0.96

text_model = TEXT_MODEL(cnn_filters=CNN_FILTERS,
                        dnn_units=DNN_UNITS,
                        model_output_classes=OUTPUT_CLASSES,
                        dropout_rate=DROPOUT_RATE)
adam = optimizers.Adam(learning_rate=0.01, decay=1, beta_1=0.9, beta_2=0.999, amsgrad=False)
text_model.compile(loss="sparse_categorical_crossentropy",
                   optimizer=adam,
                   metrics=["accuracy"])

while Iter < Niter:
    s0, s1 = balancing_routine(ghazali_df, pseudo_df, 0.9, 0.8)
    emb_train = pd.concat([s0['Embedding'], s1['Embedding']])
    label_train = pd.concat([s0['Label'], s1['Label']])
    emb_train_values = emb_train.tolist()
    x_train = torch.stack(emb_train_values)
    y_train = cvt_to_tensor(label_train)
    X_train, X_test, y_train, y_test = train_test_split_tensors(x_train, y_train)
    training_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    validation_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    del emb_train
    del s0
    del s1
    training_dataset = training_dataset.batch(BATCH_SIZE)
    validation_dataset = validation_dataset.batch(BATCH_SIZE)
    text_model.fit(training_dataset, epochs=NB_EPOCHS, validation_data=validation_dataset)
    loss, acc = text_model.evaluate(validation_dataset)
    if acc < Accuracy_threshold:
        print(f"Discarded CNN with accuracy {acc}")
        continue
    i = 0
    for filename in embedded_files:
        emb_file = pd.read_pickle(collections["Test"]["Embedding"] + Path(filename).stem + ".pkl")
        emb_df = pd.DataFrame(emb_file['Embedding'])
        emb_list = emb_df['Embedding'].tolist()
        emb_torch = torch.stack(emb_list)
        emb_to_predict = tf.constant(emb_torch)
        predict = text_model.predict(emb_to_predict)
        M[Iter][i] = np.mean(predict, axis=0)[1]
        i += 1

    Iter += 1


M_means = np.mean(M, axis=0)
M_means.reshape(-1, 1)
k_means = KMeans(n_clusters=2)
k_means.fit(M_means)
print("a")











exit()
predict = pd.read_pickle(collections["Test"]["Embedding"] + "970.pkl")
predict_df = pd.DataFrame(predict['Embedding'])
predict_val = predict_df['Embedding'].tolist()
pred = torch.stack(predict_val)
pred2 = tf.constant(pred)
predict1 = text_model.predict(pred2)
npMean = np.mean(predict1, axis=0)
kmeans = KMeans(n_clusters=2)
kmeans.fit(npMean)
plt.scatter(npMean[:, 0], npMean[:, 1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(
    npMean[:, 0], npMean[:, 1],
    c='lightblue', marker='o',
    edgecolor='black', s=50
)
plt.show()
# predict_val = tf.data.Dataset.from_tensor_slices(predict_val)
# to_predict = torch.stack(predict_val)
print("hello")
# print("\nPrediction: "+str(predict[0]))

exit()
