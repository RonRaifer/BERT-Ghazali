import glob
import math
from collections import Counter
from pathlib import Path
import pandas as pd
import tensorflow as tf
import torch
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.keras import optimizers
from transformers import AutoTokenizer, AutoModel
from Model import TEXT_MODEL
from preprocess import ArabertPreprocessor

model_name = "aubmindlab/bert-base-arabertv2"
pre_process = ArabertPreprocessor(model_name=model_name)
bert_model = AutoModel.from_pretrained(model_name)
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
        with torch.no_grad():
            outputs = bert_model(**bert_input)
            i = i + 1
            print(f'\r{i} chunks of {sz}', end="", flush=True)
            d = {'Embedding': outputs[0][0], 'Label': label}  # label 0 ghazali, 1 if pseudo
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


def targets_to_tensor(df):
    return torch.tensor(df.values, dtype=torch.float32)


emb_train_values = emb_train.tolist()
x_train = torch.stack(emb_train_values)
y_train = targets_to_tensor(label_train)
BATCH_SIZE = 20
TOTAL_SAMPLES = math.ceil(len(x_train) / BATCH_SIZE)
TEST_SAMPLES = TOTAL_SAMPLES
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
shuffled_dataset = dataset.shuffle(len(x_train)).batch(TOTAL_SAMPLES)
test_data = shuffled_dataset.take(TEST_SAMPLES)
train_data = shuffled_dataset.skip(TEST_SAMPLES)

CNN_FILTERS = 500
DNN_UNITS = 512
OUTPUT_CLASSES = 2
DROPOUT_RATE = 0.5
NB_EPOCHS = 5

text_model = TEXT_MODEL(cnn_filters=CNN_FILTERS,
                        dnn_units=DNN_UNITS,
                        model_output_classes=OUTPUT_CLASSES,
                        dropout_rate=DROPOUT_RATE)
adam = optimizers.Adam(learning_rate=0.01, decay=1, beta_1=0.9, beta_2=0.999, amsgrad=False)
text_model.compile(loss="sparse_categorical_crossentropy",
                   optimizer=adam,
                   metrics=["accuracy"])
text_model.fit(train_data, epochs=NB_EPOCHS)
results = text_model.evaluate(test_data)
print(results)

# predict = text_model.predict(test_data.)
# print("\nPrediction: "+str(predict[0]))

exit()

