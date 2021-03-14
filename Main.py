import glob
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
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


def tokenize(ts):
    processed_files = glob.glob('Data/Processed/' + ts + '/*.txt')
    tokenized_files = "Data/Tokenized/" + ts + "/"
    for filename in processed_files:
        with open(filename, mode="r", encoding="utf8") as f:  # open in readonly mode
            text_processed = f.read().replace('\n', '')
        tokenized_file = open(tokenized_files + Path(filename).stem + '.txt', mode="w", encoding="utf8")  # + '.txt'
        tokenized_file.write("\n".join(tokenizer.tokenize(text_processed)))
        tokenized_file.close()


def encode_tokens(tokensToEncode):
    # Encode the sentence
    encoded = tokenizer.encode_plus(
        text=tokensToEncode,  # the sentence to be encoded
        add_special_tokens=True,  # Add [CLS] and [SEP]
        is_split_into_words=True,
        max_length=510,  # maximum length of a sentence
        padding='max_length',  # Add [PAD]s
        return_attention_mask=True,  # Generate the attention mask
        return_tensors='pt',  # ask the function to return PyTorch tensors
    )
    return encoded


def fixed_size_division(tokenized_file, chunkSize):
    tokensList = tokenized_file.read().splitlines()
    inputForBERT = []
    for index in range(math.ceil((len(tokensList) / chunkSize))):
        chunk = tokensList[index * chunkSize: min(len(tokensList), (index * chunkSize) + chunkSize)]
        inputForBERT.append(encode_tokens(chunk))
    return inputForBERT


def last_dot_index(list, startIndex, lastIndex):
    for i in reversed(range(startIndex, min(lastIndex, len(list)))):
        if list[i] == '.':
            return i
    raise ValueError


def buttom_up_division(tokenized_file, chunkSize):
    tokensList = tokenized_file.read().splitlines()
    inputForBERT = []
    stopPoint = 0
    lastDotIndex = 0

    while stopPoint <= len(tokensList):
        try:
            #lastDotIndex = len(tempList) - tempList[::-1].index(".")
            lastDotIndex = last_dot_index(tokensList, stopPoint, stopPoint+chunkSize)
            #lastDotIndex = tokensList[stopPoint:stopPoint+chunkSize].rindex(".")
        except ValueError:
            # the dot is too far away
            lastDotIndex = stopPoint+chunkSize
        finally:
            tempList = tokensList[stopPoint:lastDotIndex]
            stopPoint = lastDotIndex+1
            inputForBERT.append(encode_tokens(tempList))
    return inputForBERT


# tokenize('ts1')
# tokenize('ts2')
# tokenize('ts3')
f = open(r"C:\Users\Ron\Desktop\BERT-Ghazali\Data\Tokenized\ts2\3b.txt", mode="r", encoding="utf8")
test = buttom_up_division(f, 510)

txt_tmp = "لا يطلع علي +ها إلا رب ال+ أرباب جل جلال +ه فيحسن ال+ شك في +ه ف+ هذه وجوه"
# text_processed_str = ' '.join(map(str, text_processed))
# create tensor id's and tokenize the input
# The senetence to be encoded
t = tokenizer.tokenize(txt_tmp)
print(t)





'''
# Get the input IDs and attention mask in tensor format
input_ids = encoded['input_ids']
attn_mask = encoded['attention_mask']
print(input_ids[0])
'''

'''
inputs = tokenizer.encode_plus(txt_tmp, return_tensors='pt', add_special_tokens=True)
# inputs = tokenizer.encode(txt_tmp, padding=True, max_length=50, add_special_tokens=True, return_tensors='pt')
print("Input ID's:")

print(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))
'''
print(encoded['input_ids'][0])
outputs = model(**encoded)

# Embedding without [CLS] and [SEP]
emb_no_tags = outputs['last_hidden_state'][0][1:-1]
emb_no_tags.shape  # (seq_len - 2) x emb_dim
pooled_vec = outputs['pooler_output']
print("Embeddings without TAGS:")
# print(emb_no_tags.size())
print(pooled_vec.size())
print(pooled_vec)

# x_train = emb_no_tags
x_train = pooled_vec
x_train = x_train.detach().numpy()

x_train = x_train.reshape(-1, 768, 1)
y_train = [1]
print(x_train)
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
