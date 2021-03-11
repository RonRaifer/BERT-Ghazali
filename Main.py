import os
import glob
from transformers import AutoTokenizer, AutoModel
from preprocess import ArabertPreprocessor
from pathlib import Path

model_name = "aubmindlab/bert-base-arabertv2"
pre_process = ArabertPreprocessor(model_name=model_name)
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def start_preprocess(ts):
    original_files = glob.glob('Data/Original/' + ts + '/*.txt')
    processed_path = "Data/Processed/" + ts + "/"
    for filename in original_files:
        with open(filename, mode="r", encoding="utf8") as f:  # open in readonly mode
            text_original = f.readlines()
        processed_file = open(processed_path + Path(filename).stem + '.txt', mode="w", encoding="utf8")  # + '.txt'
        for line_str in text_original:
            processed_file.write(pre_process.preprocess(line_str) + '\n')
        processed_file.close()


start_preprocess('ts1')

# myfile.close()

# text_original = "ومن ذلك التلحين في القرآن والآذان. وكذلك الاشتغال بدقائق الجدل والمناظرة من. "
# text_processed = []
# print("Original Text:\n")
# for line in text_original:
#    text_processed.append(pre_process.preprocess(line))
#    print(line)


# text_processed_str = ' '.join(map(str, text_processed))
# create tensor id's and tokenize the input
# inputs = tokenizer.encode_plus(text_processed_str, return_tensors='pt')
# print("Input ID's:")
# print(inputs['input_ids'][0])
# print(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))

'''
outputs = model(**inputs)
# Embedding with [CLS] and [SEP]
emb = outputs['last_hidden_state']
emb.shape   # batch_size x seq_len x emb_dim
print("Embeddings with TAGS:")
print(emb)

# Embedding without [CLS] and [SEP]
emb_no_tags = outputs['last_hidden_state'][0][1:-1]
emb_no_tags.shape   # (seq_len - 2) x emb_dim
print("Embeddings without TAGS:")
print(emb_no_tags)

pooled_vec = outputs['pooler_output']
pooled_vec.shape    # batch_size x emb_dim
print("Pooled vector output size:")
print(pooled_vec.size())
print("Pooled vector:")
print(pooled_vec)
'''
