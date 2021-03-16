import glob
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import tensorflow as tf
from Model import TEXT_MODEL
from preprocess import ArabertPreprocessor
from pathlib import Path
import math

model_name = "aubmindlab/bert-base-arabertv2"
pre_process = ArabertPreprocessor(model_name=model_name)
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def start_preprocess(ts):
    original_files = glob.glob('Data/Original/' + ts + '/*.txt')
    processed_files = "Data/Processed/" + ts + "/"
    for filename in original_files:
        with open(filename, mode="r", encoding="utf8") as f:  # open in readonly mode
            text_original = f.readlines()
        processed_file = open(processed_files + Path(filename).stem + '.txt', mode="w", encoding="utf8")  # + '.txt'
        for line_str in text_original:
            processed_file.write(pre_process.preprocess(line_str) + '\n')
        processed_file.close()


# start_preprocess('ts1')
# start_preprocess('ts2')
# start_preprocess('ts3')
def encode_tokens(tokensToEncode):
    # Encode the sentence
    encoded = tokenizer.encode_plus(
        text=tokensToEncode,  # the sentence to be encoded
        add_special_tokens=True,  # Add [CLS] and [SEP]
        # is_split_into_words=True,
        max_length=510,  # maximum length of a sentence
        padding='max_length',  # Add [PAD]s
        return_attention_mask=True,  # Generate the attention mask
        return_tensors='pt',  # ask the function to return PyTorch tensors
    )
    return encoded


def tokenize(ts):
    processed_files = glob.glob('Data/Processed/' + ts + '/*.txt')
    tokenized_files = "Data/Tokenized/" + ts + "/"
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


def last_dot_index(list, startIndex, lastIndex):
    for i in reversed(range(startIndex, min(lastIndex, len(list)))):
        if list[i] == '48':
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
            inputForBERT.append(encode_tokens(tokenizer.decode(chunk_int)))
    return inputForBERT


# tokenize('ts1')
# tokenize('ts2')
# tokenize('ts3')

def bert_embeddings(set_path, label):
    tokenized_files = glob.glob(set_path + "/*.txt")
    df = []
    divided = []
    i = 0
    for filename in tokenized_files:
        with open(filename, mode="r", encoding="utf8") as f:  # open in readonly mode
            divided.extend(fixed_size_division(f, 510))
            print(filename)
    for bert_input in divided:
        sz = len(divided)
        outputs = model(**bert_input)
        i = i + 1
        print(f'\r{i} chunks of {sz}', end="", flush=True)
        pooled_vec = outputs['pooler_output']
        d = {'Embedding': pooled_vec.detach().numpy(), 'Label': label}  # label 0 ghazali, 1 if pseudo
        df.append(d)

    df = pd.DataFrame(df)
    db_name = 'Pseudo-Ghazali.pkl' if label == 1 else 'Ghazali.pkl'
    df.to_pickle('Data/Embedding/' + db_name)
    # df.to_feather('Data/Embedding/' + db_name)


# bert_embeddings("Data/Tokenized/ts2", 1)
# bert_embeddings("Data/Tokenized/ts1", 0)

def balancing_routine(Set0, Set1, F1, F):
    over_sampler = RandomOverSampler(sampling_strategy=F)
    under_sampler = RandomUnderSampler(sampling_strategy=F1)
    x_combined_df = pd.concat([Set0, Set1])  # concate the training set
    y_combined_df = pd.to_numeric(x_combined_df['Label'])
    print(f"Combined Dataframe before sampling: {Counter(y_combined_df)}")
    x_over_sample, y_over_sample = over_sampler.fit_resample(x_combined_df, y_combined_df)
    print(f"Combined Dataframe after OVER sampling: {Counter(y_over_sample)}")
    x_combined_sample, y_combined_sample = under_sampler.fit_resample(x_over_sample, y_over_sample)
    print(f"Combined Over&Under Sampling: {Counter(y_combined_sample)}")
    print(x_combined_sample)


ghazali_df = pd.read_pickle('Data/Embedding/Ghazali.pkl')
pseudo_df = pd.read_pickle('Data/Embedding/Pseudo-Ghazali.pkl')

print(f'Samples Class 0 (Ghazali): {len(ghazali_df)}')
print(f'Samples Class 1 (Pseudo-Ghazali): {len(pseudo_df)}')
balancing_routine(ghazali_df, pseudo_df, 0.9, 0.8)

# print(f'Samples Class 0 (Ghazali): {len(s1)}')
# print(f'Samples Class 1 (Pseudo-Ghazali): {len(s2)}')
# outputs = model(**bert_input)
# pooled_vec = outputs['pooler_output']
# print(pooled_vec.size())
# print(pooled_vec)

exit()
# Embedding without [CLS] and [SEP]
# emb_no_tags = outputs['last_hidden_state'][0][1:-1]
# emb_no_tags.shape  # (seq_len - 2) x emb_dim
# print("Embeddings without TAGS:")
# print(emb_no_tags.size())


# x_train = emb_no_tags
x_train = pooled_vec
x_train = x_train.detach().numpy()

x_train = x_train.reshape(-1, 768, 1)
y_train = [1]

x_val = [0]
y_val = [0]
model1 = Sequential()

model1.add(Conv1D(128, 3, activation='relu', input_shape=(768, 1)))  # input_shape = (768,1)
model1.add(Conv1D(256, 3, activation='relu', input_shape=(768, 1)))
# flat
model1.add(Flatten())

model1.add(Dense(2, activation='softmax'))
# model1.summary()

adam = optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
model1.compile(loss='sparse_categorical_crossentropy',
               optimizer=adam,
               metrics=['accuracy'])

history = model1.fit(np.array(x_train), np.array(y_train),
                     epochs=20,
                     batch_size=200,
                     # validation_data=(np.array(x_val), np.array(y_val)), callbacks=[reduce_lr, early]
                     )
#######################
# processed_dataset = tf.data.Dataset.from_generator(lambda: y, output_types=tf.int32)
# BATCH_SIZE = 32
# batched_dataset = processed_dataset.padded_batch(BATCH_SIZE, padded_shapes=((None, ), ()))
# next(iter(batched_dataset))

# https://stackabuse.com/python-for-nlp-word-embeddings-for-deep-learning-in-keras/
