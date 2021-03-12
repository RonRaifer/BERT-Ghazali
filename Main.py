import glob
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from tensorflow.keras import layers, Sequential
import torch
from Model import TEXT_MODEL
from preprocess import ArabertPreprocessor
from pathlib import Path

model_name = "aubmindlab/bert-base-arabertv2"
pre_process = ArabertPreprocessor(model_name=model_name)
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

'''
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


start_preprocess('ts1')
'''

'''
def tokenize(ts):
    processed_files = glob.glob('Data/Processed/' + ts + '/*.txt')
    tokenized_files = "Data/Tokenized/" + ts + "/"
    # text_processed = ''
    for filename in processed_files:
        with open(filename, mode="r", encoding="utf8") as f:  # open in readonly mode
            text_processed = f.read().replace('\n', '')
        tokenized_file = open(tokenized_files + Path(filename).stem + '.txt', mode="w", encoding="utf8")  # + '.txt'
        inputs = tokenizer.encode_plus(text_processed, return_tensors='pt')  # tokenize whole file
        tokens_str = ' '.join(map(str, tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])))
        tokenized_file.write(tokens_str)
        tokenized_file.close()


tokenize('ts1')
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
print("Embeddings without TAGS:")
print(emb_no_tags.size())
#######################

Y_train = [0]
X_train = emb_no_tags
print(X_train)
txt2_tmp = "و+ هو ال+ كتاب ال+ أول من ربع ال+ عباد +ات"
inputs = tokenizer.encode_plus(txt2_tmp, return_tensors='pt')
outputs2 = model(**inputs)
emb2 = outputs2['last_hidden_state'][0][1:-1]
emb2.shape

Y_test = [0]
X_test = emb2

plt.style.use('ggplot')


def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()


def create_model(num_filters, kernel_size, vocab_size, embedding_dim, maxlen):
    model1 = Sequential()
    model1.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
    model1.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
    model1.add(layers.GlobalMaxPooling1D())
    model1.add(layers.Dense(10, activation='relu'))
    model1.add(layers.Dense(1, activation='sigmoid'))
    model1.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])
    return model1


kernel_s = 3
nn_model = create_model(500, kernel_s, 5000, 768, 30)
history = nn_model.fit(X_train, Y_train,
                       epochs=10,
                       verbose=False,
                       validation_data=(X_test, Y_test),
                       batch_size=10)
loss, accuracy = nn_model.evaluate(X_train, Y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = nn_model.evaluate(X_test, Y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
plot_history(history)
