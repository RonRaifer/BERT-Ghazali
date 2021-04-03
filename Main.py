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
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

model_name = "aubmindlab/bert-base-arabertv2"
pre_process = ArabertPreprocessor(model_name=model_name)
bert_model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
Niter = 10

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
        max_length=512,  # maximum length of a sentence
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
                        # d = {'Embedding': outputs[0][0]}
                        d = {'Embedding': outputs['pooler_output']}
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
                # d = {'Embedding': outputs[0][0], 'Label': label}  # label 0 ghazali, 1 if pseudo
                d = {'Embedding': outputs['pooler_output'], 'Label': label}  # label 0 ghazali, 1 if pseudo
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
    s0_sampled = s0_balanced.sample(math.ceil(len(s0_balanced) / 10), random_state=1)
    s1_sampled = s1_balanced.sample(math.ceil(len(s1_balanced) / 10), random_state=1)
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
M = np.zeros((Niter, 10))  # 10 num of the books in test set
BATCH_SIZE = 50

CNN_FILTERS = 500
DNN_UNITS = 512
OUTPUT_CLASSES = 2
DROPOUT_RATE = 0.3
NB_EPOCHS = 10
Accuracy_threshold = 0.96

text_model = TEXT_MODEL(cnn_filters=CNN_FILTERS,
                        dnn_units=DNN_UNITS,
                        model_output_classes=OUTPUT_CLASSES,
                        dropout_rate=DROPOUT_RATE)
adam = optimizers.Adam(learning_rate=0.001, decay=1, beta_1=0.9, beta_2=0.999, amsgrad=False)
text_model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                   optimizer=adam,
                   metrics=["accuracy"])

while Iter < Niter:
    s0, s1 = balancing_routine(ghazali_df, pseudo_df, 0.9, 0.8)
    # x_train = tf.stack(s0.Embedding.tolist())
    # s0.Embedding.to_numpy().tolist()
    emb_train_df = pd.concat([s0, s1])
    labels = emb_train_df.pop('Label')
    embeddings = emb_train_df.pop('Embedding')
    # x_train = [*s0.Embedding.to_numpy(), *s1.Embedding.to_numpy()]
    # y_train = [*s0.Label.to_numpy(), *s1.Label.to_numpy()]
    # y_train = [torch.tensor(s, dtype=torch.float32) for s in y_train]
    # embeddings.iloc[0]
    # tf.convert_to_tensor(embeddings)

    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, random_state=1, shuffle=True)

    # nX_train = [np.array(s) for s in X_train]
    # nX_test = [np.array(s) for s in X_test]
    # ny_train = [np.array(s) for s in y_train]
    # ny_test = [np.array(s) for s in y_test]
    # training_dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(nX_train), tf.convert_to_tensor(ny_train)))
    # validation_dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(nX_test), tf.convert_to_tensor(ny_test)))

    training_dataset = tf.data.Dataset.from_tensor_slices(([tf.convert_to_tensor(s) for s in X_train], y_train.values))
    validation_dataset = tf.data.Dataset.from_tensor_slices(([tf.convert_to_tensor(s) for s in X_test], y_test.values))

    training_dataset = training_dataset.batch(1)
    validation_dataset = validation_dataset.batch(1)
    # validating = tf.data.Dataset.from_tensor_slices(([tf.convert_to_tensor(s.reshape(-1, 768, 1)) for s in X_test],
    #                                                  [tf.convert_to_tensor(s) for s in y_test]))
    # label_train = pd.concat([s0['Label'], s1['Label']])
    # emb_train_values = emb_train.tolist()
    # x_train = torch.stack(emb_train_values)
    # y_train = cvt_to_tensor(label_train)
    # X_train, X_test, y_train, y_test = train_test_split_tensors(x_train, y_train, shuffle=True)
    # training_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    # validation_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    # del emb_train
    del s0
    del s1
    del emb_train_df
    # training_dataset = training_dataset.batch(BATCH_SIZE)
    # validation_dataset = validation_dataset.batch(BATCH_SIZE)
    text_model.fit(training_dataset, epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
                   validation_data=validation_dataset)
    # text_model.fit(training_dataset, batch_size=BATCH_SIZE, epochs=NB_EPOCHS, validation_data=validation_dataset)
    loss, acc = text_model.evaluate(validation_dataset, batch_size=BATCH_SIZE)
    # loss, acc = text_model.evaluate(validation_dataset)
    if acc < Accuracy_threshold:
        print(f"Discarded CNN with accuracy {acc}")
        continue
    i = 0
    for filename in embedded_files:
        emb_file = pd.read_pickle(collections["Test"]["Embedding"] + Path(filename).stem + ".pkl")
        emb_df = pd.DataFrame(emb_file)
        emb_pred_df = emb_df.pop('Embedding')
        # [tf.convert_to_tensor(s) for s in emb_pred_df]
        # for i in range(len(emb_pred_df)):
        #     emb_pred_df.iloc[i] = tf.convert_to_tensor(emb_pred_df.iloc[i])
        to_predict = tf.data.Dataset.from_tensor_slices([tf.convert_to_tensor(s) for s in emb_pred_df])
        predict = text_model.predict(to_predict.batch(BATCH_SIZE))
        print(f"File Num: {i}, name: " + Path(filename).stem)
        M[Iter][i] = np.mean(predict, axis=0)[1]
        print(M[Iter])
        i += 1

    Iter += 1

#np.save('Data/Mat12.npy', M)    # .npy extension is added if not given
d = np.load('Data/Mat.npy')
transposedMat = d.transpose()
avgdArr = np.average(d,axis = 0)
kmeans = KMeans(
    init = "random",
    n_clusters = 2,
    n_init = 10,
    max_iter = 300,
    random_state = 42
    )
exit()
#res = kmeans.fit(transposedMat)
res2 = kmeans.fit(avgdArr.reshape(-1,1)) #res and res2 are the same, we'll use res2 cuz it has more understandable dimensions.
silVal = sklearn.metrics.silhouette_score(avgdArr.reshape(-1,1), res2.labels_)

anchorGhazaliLabel = res2.labels_[0]
anchorPseudoGhazaliLabel = res2.labels_[8]

silhouetteDemandSatisfied = silVal > silhouetteTreshold
anchorsDemandSatisfied = anchorGhazaliLabel != anchorPseudoGhazaliLabel
if(not silhouetteDemandSatisfied or not anchorsDemandSatisfied):
    print("the given configurations yield unstable classification values.")
    if(not silhouetteDemandSatisfied):
        print("\tsilhouette threshold is: "+ str(silhouetteTreshold) + ", actual silhouette value: "+ str(silVal))
    if(not anchorsDemandSatisfied):
        print("\tanchors belong to the same cluster")
else:
    print("succesfully classified, the labels are: " + str(res2.labels_))

from yellowbrick.cluster import SilhouetteVisualizer

#fig, ax = plt.subplots(1, 1, figsize=(6,4))
#km = KMeans(
#     init="random",
#     n_clusters=2,
#     n_init=10,
#     max_iter=300,
#     random_state=42
# )
## km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
## q, mod = divmod(i, 2)
#visualizer = SilhouetteVisualizer(km, colors='yellowbrick')
#visualizer.fit(avgdArr.reshape(-1,1))
#visualizer.show()

plt.scatter(range(0, 10), avgdArr)
plt.show()

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
