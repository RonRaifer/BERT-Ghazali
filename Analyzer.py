import glob
import math
import os

from GuiFiles import NewGui

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
from sklearn.metrics import silhouette_score

model_name = "aubmindlab/bert-base-arabertv2"
tuned_model = "TunedGazaliBert"
pre_process = ArabertPreprocessor(model_name=model_name)
bert_model = AutoModel.from_pretrained(tuned_model, output_hidden_states=True)
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

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def bert_embeddings(col):
    tokenized_files = glob.glob(col["Tokenized"] + "*.txt")
    df = []
    divided = []
    i = 0
    if col["Name"] == "Test":
        for filename in tokenized_files:
            with open(filename, mode="r", encoding="utf8") as f:  # open in readonly mode
                # divided = fixed_size_division(f, 510)
                divided = fixed_size_division(f, 510)
                # divided = bottom_up_division(f, 510)
                print(filename)
                i = 0
                for bert_input in divided:
                    sz = len(divided)
                    with torch.no_grad():
                        outputs = bert_model(**bert_input)
                        sentence_embedding = mean_pooling(outputs, bert_input['attention_mask'])
                        # `outputs['hidden_states'][-2][0]` is a tensor with shape [512 x 768]
                        # Calculate the average of all 510 token vectors. # without CLS!
                        # sentence_embedding = torch.mean(outputs['hidden_states'][-2][0][1:-1], dim=0)
                        i = i + 1
                        print(f'\r{i} chunks of {sz}', end="", flush=True)
                        # d = {'Embedding': outputs[0][0]}
                        # d = {'Embedding': outputs['pooler_output']}
                        d = {'Embedding': sentence_embedding}
                        df.append(d)
            df = pd.DataFrame(df)
            df.to_pickle(col["Embedding"] + Path(filename).stem + ".pkl")
            df = []
    else:
        db_name = 'Pseudo-Ghazali.pkl' if col["Name"] == "Alternative" else 'Ghazali.pkl'
        label = 1 if col["Name"] == "Alternative" else 0
        for filename in tokenized_files:
            with open(filename, mode="r", encoding="utf8") as f:  # open in readonly mode
                # divided.extend(fixed_size_division(f, 510))
                divided.extend(fixed_size_division(f, 510))
                # divided.extend(bottom_up_division(f, 510))
                print(filename)
        for bert_input in divided:
            sz = len(divided)
            with torch.no_grad():
                outputs = bert_model(**bert_input)
                sentence_embedding = mean_pooling(outputs, bert_input['attention_mask'])
                # `outputs['hidden_states'][-2][0]` is a tensor with shape [512 x 768]
                # Calculate the average of all 510 token vectors. # without CLS!
                # sentence_embedding = torch.mean(outputs['hidden_states'][-2][0][1:-1], dim=0)
                i = i + 1
                print(f'\r{i} chunks of {sz}', end="", flush=True)
                # d = {'Embedding': outputs[0][0], 'Label': label}  # label 0 ghazali, 1 if pseudo
                # d = {'Embedding': outputs['pooler_output'], 'Label': label}  # label 0 ghazali, 1 if pseudo
                d = {'Embedding': sentence_embedding, 'Label': label}
                df.append(d)

        df = pd.DataFrame(df)
        df.to_pickle(col["Embedding"] + db_name)


# bert_embeddings(collections["Source"])
# bert_embeddings(collections["Alternative"])
# bert_embeddings(collections["Test"])

'''
######### TEST
def fixed_size_division_TEST(tokenized_file, chunkSize):
    tokensList = tokenized_file.read().splitlines()
    inputForBERT = []
    for index in range(math.ceil((len(tokensList) / chunkSize))):
        chunk = tokensList[index * chunkSize: min(len(tokensList), (index * chunkSize) + chunkSize)]
        chunk_int = list(map(int, chunk))
        inputForBERT.append(tokenizer.decode(chunk_int, skip_special_tokens=True))
    return inputForBERT


def bert_train(col):
    tokenized_files = glob.glob(col["Tokenized"] + "*.txt")
    df = []
    divided = []
    i = 0
    db_name = 'Pseudo-Ghazali.pkl' if col["Name"] == "Alternative" else 'Ghazali.pkl'
    label = 1 if col["Name"] == "Alternative" else 0
    for filename in tokenized_files:
        with open(filename, mode="r", encoding="utf8") as f:  # open in readonly mode
            divided.extend(fixed_size_division_TEST(f, 510))
            # divided.extend(bottom_up_division(f, 510))
            print(filename)
    for bert_input in divided:
        sz = len(divided)
        i = i + 1
        print(f'\r{i} chunks of {sz}', end="", flush=True)
        d = {'Embedding': bert_input, 'Label': label}
        df.append(d)

    df = pd.DataFrame(df)
    df.to_pickle("Data/" + db_name)


def balancing_routine_NEW(Set0, Set1, F1, F):
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
    # s0_sampled = pd.concat([s0_sampled, s0_sampled])
    # s1_sampled = pd.concat([s1_sampled, s1_sampled])
    return s0_balanced, s1_balanced


bert_train(collections["Source"])
bert_train(collections["Alternative"])
se0, se1 = balancing_routine_NEW(pd.read_pickle("Data/Ghazali.pkl")
                                 , pd.read_pickle("Data/Pseudo-Ghazali.pkl")
                                 , 0.3, 'minority')
emb_train_df = pd.concat([se0, se1]).sample(frac=1)
emb_train_df.to_pickle("Data/DB.pkl")
exit()
######### TEST
'''


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
    s0_balanced = pd.concat([s0_balanced, s0_balanced])
    s1_balanced = pd.concat([s1_balanced, s1_balanced])
    ##END NEW
    s0_sampled = s0_balanced.sample(math.ceil(len(s0_balanced) / 2)).reset_index(drop=True)
    s1_sampled = s1_balanced.sample(math.ceil(len(s1_balanced) / 2)).reset_index(drop=True)
    # s0_sampled = pd.concat([s0_sampled, s0_sampled])
    # s1_sampled = pd.concat([s1_sampled, s1_sampled])
    return s0_sampled, s1_sampled


''' 
Combined Dataframe before sampling: Counter({0: 3311, 1: 443})
Combined Dataframe after OVER sampling: Counter({0: 3311, 1: 1986})
Combined Over&Under Sampling: Counter({0: 2482, 1: 1986})
'''


# NewGui.vp_start_gui()
def run():
    ghazali_df = pd.read_pickle(collections["Source"]["Embedding"] + "Ghazali.pkl")
    pseudo_df = pd.read_pickle(collections["Alternative"]["Embedding"] + "Pseudo-Ghazali.pkl")

    print(f'Samples Class 0 (Ghazali): {len(ghazali_df)}')
    print(f'Samples Class 1 (Pseudo-Ghazali): {len(pseudo_df)}')

    embedded_files = glob.glob(collections["Test"]["Embedding"] + "*.pkl")


    # 'BERT_INPUT_LENGTH': 510,
    # 'TEXT_DIVISION_METHOD': 'Fixed-Size',
    # '1D_CONV_KERNEL': {1: 3, 2: 6, 3: 12}
    # 'POOLING_SIZE': 500,
    # 'STRIDES': 1,
    # 'ACTIVATION_FUNC': 'Relu',

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
                       validation_data=validation_dataset)
        loss, acc = text_model.evaluate(validation_dataset)
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

    np.save('Data/MatPooledNew4.npy', M)  # .npy extension is added if not given
    d = np.load('Data/MatPooledNew4.npy')
    transposedMat = d.transpose()
    avgdArr = np.average(d, axis=0)
    kmeans = KMeans(
        init="random",
        n_clusters=2,
        n_init=10,
        max_iter=300,
        random_state=42
    )

    # res = kmeans.fit(transposedMat)
    res2 = kmeans.fit(
        avgdArr.reshape(-1, 1))  # res and res2 are the same, we'll use res2 cuz it has more understandable dimensions.
    silVal = silhouette_score(avgdArr.reshape(-1, 1), res2.labels_)

    anchorGhazaliLabel = res2.labels_[0]
    anchorPseudoGhazaliLabel = res2.labels_[8]

    silhouetteDemandSatisfied = silVal > params['SILHOUETTE_THRESHOLD']
    anchorsDemandSatisfied = anchorGhazaliLabel != anchorPseudoGhazaliLabel
    if (not silhouetteDemandSatisfied or not anchorsDemandSatisfied):
        print("the given configurations yield unstable classification values.")
        if (not silhouetteDemandSatisfied):
            print("\tsilhouette threshold is: " + str(
                params['SILHOUETTE_THRESHOLD']) + ", actual silhouette value: " + str(silVal))
        if (not anchorsDemandSatisfied):
            print("\tanchors belong to the same cluster")
    else:
        print("succesfully classified, the labels are: " + str(res2.labels_))

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

    plt.scatter(range(0, 10), avgdArr)
    plt.show()

