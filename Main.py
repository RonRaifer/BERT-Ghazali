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
    # text_processed = ''
    for filename in processed_files:
        with open(filename, mode="r", encoding="utf8") as f:  # open in readonly mode
            text_processed = f.read().replace('\n', '')
        tokenized_file = open(tokenized_files + Path(filename).stem + '.txt', mode="w", encoding="utf8")  # + '.txt'
        inputs = tokenizer.encode_plus(text_processed, return_tensors='pt')  # tokenize whole file
        # tokens_str = ' '.join(map(str, tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])))
        st = inputs['input_ids'][0][1:-1]
        st = st.numpy()
        tokens_str = ' '.join(map(str, st))
        tokenized_file.write(tokens_str)
        tokenized_file.close()


tokenize('ts1')
tokenize('ts2')
tokenize('ts3')

'''
ardb = pd.read_csv("db.csv")
ardb.isnull().values.any()
ardb.shape

prep = []
seq = list(ardb['text'])
for sen in seq:
    prep.append(sen)

y = ardb['isghazali']
y = np.array(list(map(lambda x: 1 if x=="1" else 0, y)))

print(y[0])
print(prep[0])
'''
txt_tmp = "كان يتضاءل دون حق جلال +ه حمد ال+ حامد +ين ."
# text_processed_str = ' '.join(map(str, text_processed))
# create tensor id's and tokenize the input
inputs = tokenizer.encode_plus(txt_tmp, return_tensors='pt')
print("Input ID's:")
print(inputs['input_ids'][0])
print(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))

outputs = model(**inputs)

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
