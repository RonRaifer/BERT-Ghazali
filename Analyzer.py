import glob
import json
import math
import os
import threading
import zipfile
import utils
from kim_cnn import KimCNN
from collections import Counter
from pathlib import Path
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
# from transformers import AutoTokenizer, AutoModel
from preprocess import ArabertPreprocessor
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

model_name = "aubmindlab/bert-base-arabertv2"
model_name = "asafaya/bert-base-arabic"
tuned_model = "TunedGazaliBert"
# bert_model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

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
    pre_process = ArabertPreprocessor(model_name=model_name)
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
        max_length=utils.params['BERT_INPUT_LENGTH'] + 2,  # maximum length of a sentence
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
            if prefix[0] == '/':
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
        # import os
        from os.path import basename

        # save zip file to previous runs.
        zipf = zipfile.ZipFile(r"Data/PreviousRuns/Embeddings/" + embeddings_file + '.zip', 'w', zipfile.ZIP_DEFLATED)
        zipdir(tmpdirname.name + "/Data/", zipf)
        zipf.close()
        # tmpdirname.cleanup()

    if os.path.exists(os.getcwd() + r"\Data\Embedding\current.txt"):
        with open(os.getcwd() + r"\Data\Embedding\current.txt", 'r') as file:
            if file.readline() != embeddings_file:
                # unzip the right embeddings file into the general Embedding directory
                with zipfile.ZipFile(os.path.join(embeddings_zip_location, embeddings_file + ".zip"), 'r') as zip_ref:
                    zip_ref.extractall(os.getcwd() + r"\Data")
                    with open(os.getcwd() + r"\Data\Embedding\current.txt", 'w') as f:
                        f.write(embeddings_file)
    else:
        with zipfile.ZipFile(os.path.join(embeddings_zip_location, embeddings_file + ".zip"), 'r') as zip_ref:
            zip_ref.extractall(os.getcwd() + r"\Data")
            with open(os.getcwd() + r"\Data\Embedding\current.txt", 'w') as f:
                f.write(embeddings_file)


def bert_embeddings(col, division_method, input_len, output_path):
    tokenized_files = glob.glob(col["Tokenized"] + "*.txt")
    df = []
    divided = []
    utils.progress_bar["value"] = 0
    i = 0
    if not os.path.exists(output_path.name + "/" + col["Embedding"]):
        os.makedirs(output_path.name + "/" + col["Embedding"])
    if col["Name"] == "Test":
        print(f"Generating Embeddings For {col['Name']}")
        for filename in tokenized_files:
            utils.progress_bar["value"] = 0
            i = 0
            with open(filename, mode="r", encoding="utf8") as f:  # open in readonly mode
                divided = division_method(f, input_len)
                sz = len(divided)
                print(f"Book: {Path(filename).stem}, Total chunks: {sz}. Please wait...", end="")
                for bert_input in divided:
                    utils.progress_bar['maximum'] = sz
                    utils.progress_bar["value"] = int(utils.progress_bar["value"]) + 1
                    utils.progress_bar.update()
                    with torch.no_grad():
                        outputs = bert_model(**bert_input)
                        i = i + 1
                        # d = {'Embedding': outputs[0][0]}
                        # d = {'Embedding': outputs['pooler_output']}
                        # d = {'Embedding': outputs['last_hidden_state'][0]}
                        d = {'Embedding': outputs['last_hidden_state'][0][1:-1]}
                        df.append(d)
                print(" ~DONE!")
            df = pd.DataFrame(df)
            df.to_pickle(output_path.name + "/" + col["Embedding"] + Path(filename).stem + ".pkl")
            df = []
    else:  # source\alternative
        db_name = 'Pseudo-Ghazali.pkl' if col["Name"] == "Alternative" else 'Ghazali.pkl'
        label = 1 if col["Name"] == "Alternative" else 0
        # print("Loaded files:", end="")
        for filename in tokenized_files:
            with open(filename, mode="r", encoding="utf8") as f:  # open in readonly mode
                divided.extend(division_method(f, input_len))
                # print(f"{Path(filename).stem}", end=", ")

        sz = len(divided)
        print(f"\nGenerating Embeddings For {col['Name']}, Total chunks: {sz}. Please wait...", end="")
        for bert_input in divided:
            utils.progress_bar['maximum'] = sz
            utils.progress_bar["value"] = int(utils.progress_bar["value"]) + 1
            utils.progress_bar.update()
            with torch.no_grad():
                outputs = bert_model(**bert_input)
                # sentence_embedding = mean_pooling(outputs, bert_input['attention_mask'])
                # `outputs['hidden_states'][-2][0]` is a tensor with shape [512 x 768]
                # Calculate the average of all 510 token vectors. # without CLS!
                # sentence_embedding = torch.mean(outputs['hidden_states'][-2][0][1:-1], dim=0)
                i = i + 1
                # print(f'\r{i} chunks of {sz}', end="", flush=True)
                # d = {'Embedding': outputs[0][0], 'Label': label}  # label 0 ghazali, 1 if pseudo
                # d = {'Embedding': outputs['pooler_output'], 'Label': label}  # label 0 ghazali, 1 if pseudo
                d = {'Embedding': outputs['last_hidden_state'][0][1:-1], 'Label': label}

                df.append(d)
        print(" ~DONE!")

        df = pd.DataFrame(df)
        df.to_pickle(output_path.name + "/" + col["Embedding"] + db_name)


def balancing_routine(Set0, Set1, F1, F):
    if len(Set0) < len(Set1):
        temp = F
        F = F1
        F1 = temp
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
    # s0_sampled = s0_balanced.sample(math.ceil(len(s0_balanced) / 5)).reset_index(drop=True)
    # s1_sampled = s1_balanced.sample(math.ceil(len(s1_balanced) / 5)).reset_index(drop=True)
    s0_sampled = s0_balanced
    s1_sampled = s1_balanced
    return s0_sampled, s1_sampled


''' 
Combined Dataframe before sampling: Counter({0: 3311, 1: 443})
Combined Dataframe after OVER sampling: Counter({0: 3311, 1: 1986})
Combined Over&Under Sampling: Counter({0: 2482, 1: 1986})
'''


class StdoutRedirector(object):
    def __init__(self, text_widget):
        self.text_space = text_widget
        self.msg = ""
        self.x = 0

    def write(self, string):
        self.text_space.insert('end', f'\r{string}')

    def flush(self):
        pass


def targets_to_tensor(df, target_columns):
    return torch.tensor(df[target_columns].values, dtype=torch.float32)


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


def run2(text_console):
    import sys
    lock = threading.Lock()
    lock.acquire()
    original_stdout = sys.stdout
    try:
        sys.stdout = StdoutRedirector(
            text_console)
    finally:
        lock.release()

    bert_embeddings_general()

    import torch.utils.data as data_utils
    ghazali_df = pd.read_pickle(collections["Source"]["Embedding"] + "Ghazali.pkl")
    pseudo_df = pd.read_pickle(collections["Alternative"]["Embedding"] + "Pseudo-Ghazali.pkl")

    print(f"Total Ghazali's Samples: {len(ghazali_df)}")
    print(f"Total Pseudo-Ghazali's: {len(pseudo_df)}")

    embedded_files = glob.glob(collections["Test"]["Embedding"] + "*.pkl")
    import torch.nn as nn

    from utils import params, progress_bar
    M = np.zeros((params['Niter'], len(embedded_files)))  # 10 num of the books in test set

    Iter = 0
    progress_bar['maximum'] = params['NB_EPOCHS']
    print("**Starting Training Process**")
    # NEW
    s0, s1 = balancing_routine(ghazali_df, pseudo_df, params['F1'], params['F'])
    # First checking if GPU is available
    train_on_gpu = torch.cuda.is_available()

    if train_on_gpu:
        print('Training on GPU.')
    else:
        print('No GPU available, training on CPU.')

    embed_num = params['BERT_INPUT_LENGTH']  # number of words in seq
    embed_dim = 768  # 768
    class_num = params['OUTPUT_CLASSES']  # y_train.shape[1]
    kernel_num = params['KERNELS']
    kernel_sizes = list(params['1D_CONV_KERNEL'].values())
    dropout = params['DROPOUT_RATE']
    num_filters = 100
    static = True

    net = KimCNN(
        embed_num=embed_num,
        embed_dim=embed_dim,
        class_num=class_num,
        kernel_num=kernel_num,
        kernel_sizes=kernel_sizes,
        dropout=dropout,
        static=static
    )
    print(net)
    nepochs = params['NB_EPOCHS']
    batch_size = params['BATCH_SIZE']
    lr = params['LEARNING_RATE']

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    from torch.utils.data import TensorDataset, DataLoader

    # training loop
    def train(net, train_loader, valid_loader, epochs, print_every=100):
        # move model to GPU, if available
        if (train_on_gpu):
            net.cuda()

        counter = 0  # for printing

        # train for some number of epochs
        net.train()
        for e in range(epochs):

            # batch loop
            for inputs, labels in train_loader:
                counter += 1

                if (train_on_gpu):
                    inputs, labels = inputs.cuda(), labels.cuda()

                # b_input_ids = torch.tensor(inputs)
                # inputs = inputs.type(torch.LongTensor)
                # zero accumulated gradients
                net.zero_grad()
                # get the output from the model
                output = net(inputs)

                # calculate the loss and perform backprop
                loss = criterion(output.squeeze(), labels.float())
                loss.backward()
                optimizer.step()

                # loss stats
                if counter % print_every == 0:
                    # Get validation loss
                    val_losses = []
                    net.eval()
                    for inputs, labels in valid_loader:

                        if (train_on_gpu):
                            inputs, labels = inputs.cuda(), labels.cuda()

                        output = net(inputs)
                        val_loss = criterion(output.squeeze(), labels.float())

                        val_losses.append(val_loss.item())

                    net.train()
                    print("Epoch: {}/{}...".format(e + 1, epochs),
                          "Step: {}...".format(counter),
                          "Loss: {:.6f}...".format(loss.item()),
                          "Val Loss: {:.6f}".format(np.mean(val_losses)))

    while Iter < params['Niter']:
        # s0, s1 = balancing_routine(ghazali_df, pseudo_df, params['F1'], params['F'])
        s0_sampled = s0.sample(math.ceil(len(s0) / 5)).reset_index(drop=True)
        s1_sampled = s1.sample(math.ceil(len(s1) / 5)).reset_index(drop=True)
        emb_train_df = pd.concat([s0_sampled, s1_sampled])
        # labels = emb_train_df['Label'].values
        # labels = torch.Tensor([np.asarray(s) for s in labels])

        # labels = targets_to_tensor(emb_train_df, 'Label')
        # labels = torch.unsqueeze(labels, 1)
        # labels = torch.unsqueeze(labels, 1)

        # torch.reshape(labels, ())
        # labels = torch.stack([torch.tensor(s, dtype=torch.float32) for s in emb_train_df['Label'].values])
        labels = torch.FloatTensor(emb_train_df["Label"].values)\
            # .unsqueeze(1)
        # labels = labels.unsqueeze(1)
        # labels = emb_train_df.pop('Label')
        # embeddings = emb_train_df.pop('Embedding')
        # embeddings = emb_train_df.pop('Embedding').values

        embeddings_tensor = torch.stack(emb_train_df['Embedding'].tolist())

        # embeddings_tensor = emb_train_df['Embedding'].values
        # embeddings_tensor = torch.Tensor([np.asarray(s) for s in embeddings_tensor])
        # = torch.Tensor([torch.Tensor(X) for X in emb_train_df['Embedding'].values])
        X_train, X_test, y_train, y_test = train_test_split(embeddings_tensor, labels, test_size=0.33,
                                                            shuffle=True, random_state=1)  # shuffle=True,
        test_idx = int(len(X_test) * 0.5)
        val_x, test_x = X_test[:test_idx], X_test[test_idx:]
        val_y, test_y = y_test[:test_idx], y_test[test_idx:]
        del s0_sampled
        del s1_sampled
        del emb_train_df
        train_data = TensorDataset(X_train, y_train)
        test_data = TensorDataset(test_x, test_y)
        valid_data = TensorDataset(val_x, val_y)
        train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
        test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
        valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
        # training_dataset = data_utils.TensorDataset(X_train, y_train)
        # validation_dataset = data_utils.TensorDataset(X_test, y_test)
        epochs = nepochs
        print_every = 16
        train(net, train_loader, valid_loader, epochs, print_every=print_every)

        # Get test data loss and accuracy

        test_losses = []  # track loss
        num_correct = 0

        net.eval()
        # iterate over test data
        for inputs, labels in test_loader:

            if (train_on_gpu):
                inputs, labels = inputs.cuda(), labels.cuda()

            # get predicted outputs
            output = net(inputs)

            # calculate loss
            test_loss = criterion(output.squeeze(), labels.float())
            test_losses.append(test_loss.item())

            # convert output probabilities to predicted class (0 or 1)
            pred = torch.round(output.squeeze())  # rounds to the nearest integer

            # compare predictions to true label
            correct_tensor = pred.eq(labels.float().view_as(pred))
            correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(
                correct_tensor.cpu().numpy())
            num_correct += np.sum(correct)

        # -- stats! -- ##
        # avg test loss
        print("Test loss: {:.3f}".format(np.mean(test_losses)))

        # accuracy over all test data
        test_acc = num_correct / len(test_loader.dataset)
        print("Test accuracy: {:.3f}".format(test_acc))

        # print(f'\nTotal acc: {total_accuracy}')

        if test_acc <= params['ACCURACY_THRESHOLD']:
            continue

        i = 0
        for filename in embedded_files:
            emb_file = pd.read_pickle(collections["Test"]["Embedding"] + Path(filename).stem + ".pkl")
            emb_file = pd.DataFrame(emb_file)
            embeddings_test = torch.stack(emb_file['Embedding'].tolist())

            net.eval()

            batch_size = embeddings_test.size(0)

            if train_on_gpu:
                feature_tensor = embeddings_test.cuda()
            else:
                feature_tensor = embeddings_test
            with torch.no_grad():  # deactivate autograd engine to reduce memory usage and speed up computations
                # get the output from the model
                output = net(feature_tensor)
                M[Iter][i] = round(np.mean(np.array(output.cpu()), axis=0)[0], 4)
            print(f"Iter [{Iter}], File [{i}]: {M[Iter][i]}")
            i += 1
        Iter += 1
        # progress_bar["value"] = int(progress_bar["value"]) + 1
    print("**Finished Training**")
    progress_bar["value"] = 0
    sys.stdout = original_stdout

    import utils
    utils.heat_map = M


def produce_heatmap():
    import seaborn as sns
    utils.heat_map_plot, ax = plt.subplots(figsize=(11, 9), dpi=100)
    # color map
    cmap = sns.light_palette("seagreen", as_cmap=True)
    sns.heatmap(utils.heat_map, annot=True, cmap=cmap, fmt=".2f",
                linewidth=0.3, cbar_kws={"shrink": .8}, ax=ax)


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
    # res80 = kmeans.predict(avgdArr.reshape(-1, 1))
    # centroids = res2.cluster_centers_
    utils.labels = np.zeros((len(res2.labels_),), dtype=int)

    # u_labels = np.unique(res2.labels_)
    centroids = res2.cluster_centers_
    anchorGhazaliLabel = res2.labels_[0]
    anchorPseudoGhazaliLabel = res2.labels_[8]
    for i, lbl in enumerate(res2.labels_):
        if lbl == anchorPseudoGhazaliLabel:
            utils.labels[i] = 1


    # matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    # kmeans_plot = plt.figure(figsize=(8, 6))
    utils.kmeans_plot = plt.figure(figsize=(6, 5), dpi=100)
    ax = plt.axes()
    x = np.linspace(-1, 11, 100)
    ax.plot(x, x * 0.00000000000001 + centroids[0][0])
    ax.plot(x, x * 0.00000000000001 + centroids[1][0])
    plt.scatter(range(0, len(avgdArr)), avgdArr, c=utils.labels, s=50, cmap='viridis')

    # plt.scatter(centroids[0, :], centroids[1, :], c='r', s=100)
    # plt.scatter(range(0, 10), avgdArr)
    # plt.legend()
    # kmeans_plot.show()

    silVal = silhouette_score(avgdArr.reshape(-1, 1), utils.labels)
    utils.silhouette_calc = silVal
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
        print("succesfully classified, the labels are: " + str(utils.labels))


def show_results():
    if utils.heat_map is None:
        utils.heat_map = np.load(os.getcwd() + r"\Data\PreviousRuns\\" + utils.params['Name'] + ".npy")
    produce_kmeans()
    produce_heatmap()


def read_json():
    data_base = []
    with open('Data/PreviousRuns/PreviousRuns.json', 'r') as json_file:
        try:
            data_base = json.load(json_file)
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
    with open(os.getcwd() + r"\Data\PreviousRuns\\" + utils.params['Name'] + ".npy", 'wb') as m:
        np.save(m, utils.heat_map)
