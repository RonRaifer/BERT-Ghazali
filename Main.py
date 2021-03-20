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
from kcnn import KimCNN
from preprocess import ArabertPreprocessor
from pathlib import Path
import math
import torch
import time
import torch.nn as nn

model_name = "aubmindlab/bert-base-arabertv2"
pre_process = ArabertPreprocessor(model_name=model_name)
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
Niter = 20


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
            inputForBERT.append(encode_tokens(tokenizer.decode(chunk_int, skip_special_tokens=True)))
    return inputForBERT


# tokenize('ts1')
# tokenize('ts2')
# tokenize('ts3')

def bert_embeddings(set_path, label):
    tokenized_files = glob.glob(set_path + "/*.txt")
    db_name = 'Pseudo-Ghazali.pkl' if label == 1 else 'Ghazali.pkl'
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
        # outputs2 = model(**bert_input)[0]
        i = i + 1
        print(f'\r{i} chunks of {sz}', end="", flush=True)
        pooled_vec = outputs['pooler_output']
        d = {'Embedding': pooled_vec.detach().numpy, 'Label': label}  # label 0 ghazali, 1 if pseudo
        df.append(d)

    df = pd.DataFrame(df)
    df.to_pickle('Data/Embedding/' + db_name)
    # df.to_feather('Data/Embedding/' + db_name)


# bert_embeddings("Data/Tokenized/ts2", 1)
# bert_embeddings("Data/Tokenized/ts1", 0)


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


ghazali_df = pd.read_pickle('Data/Embedding/Ghazali.pkl')
pseudo_df = pd.read_pickle('Data/Embedding/Pseudo-Ghazali.pkl')

print(f'Samples Class 0 (Ghazali): {len(ghazali_df)}')
print(f'Samples Class 1 (Pseudo-Ghazali): {len(pseudo_df)}')
s0, s1 = balancing_routine(ghazali_df, pseudo_df, 0.9, 0.8)

emb_train = pd.concat([s0['Embedding'], s1['Embedding']])
label_train = pd.concat([s0['Label'], s1['Label']])


# print(emb_train.values.dtype)
# torch_tensor = tf.convert_to_tensor(emb_train.tolist())
# print(torch_tensor.shape)

# emblst = emb_train.tolist()
# emblst = np.array(emb_train.tolist())
# emblst = emblst.reshape(-1, 768, 1)
# emblst = tf.convert_to_tensor(emb_train.tolist())


def targets_to_tensor(df):
    lst = []
    for ar in df:
        lst.append(torch.from_numpy(ar))
    return lst


emblst = targets_to_tensor(emb_train.tolist())
# emblst = tf.convert_to_tensor(emb_train.tolist())

lbl = label_train.tolist()
dataset = tf.data.Dataset.from_tensor_slices((emblst, lbl))
for feat, targ in dataset.take(1):
    print('Features: {}, Target: {}'.format(feat, targ))

train_dataset = dataset.shuffle(len(emb_train) + len(label_train)).batch(1)
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

history = model1.fit(train_dataset,
                     epochs=20,
                     batch_size=200,
                     # validation_data=(np.array(x_val), np.array(y_val)), callbacks=[reduce_lr, early]
                     )

exit()
embed_num = 510
embed_dim = 768
class_num = 2
kernel_num = 3
kernel_sizes = [3, 6, 12]
dropout = 0.5
static = True


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


model = KimCNN(
    embed_num=embed_num,
    embed_dim=embed_dim,
    class_num=class_num,
    kernel_num=kernel_num,
    kernel_sizes=kernel_sizes,
    dropout=dropout,
    static=static,
)

n_epochs = 20
batch_size = 1
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss

train_losses, val_losses = [], []

for epoch in range(n_epochs):
    start_time = time.time()
    train_loss = 0

    model.train(True)
    for x_batch, y_batch, batch in generate_batch_data(x_train, y_train, batch_size):
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
        for x_batch, y_batch, batch in generate_batch_data(x_val, y_val, batch_size):
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            val_loss += loss.item()
        val_loss /= batch
        val_losses.append(val_loss)

    print(
        "Epoch %d Train loss: %.2f. Validation loss: %.2f. Elapsed time: %.2fs."
        % (epoch + 1, train_losses[-1], val_losses[-1], elapsed)
    )

'''
emblst = emb_train.tolist()
print(emblst)

emblst = np.array(emblst)
emblst = emblst.reshape(-1, 768, 1)
print(emblst)
'''

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

history = model1.fit(train_dataset,
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
