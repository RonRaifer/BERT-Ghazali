from transformers import AutoTokenizer, AutoModel
from preprocess import ArabertPreprocessor

model_name = "aubmindlab/bert-base-arabertv2"
pre_process = ArabertPreprocessor(model_name=model_name)
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

t1_original_path = 'Data/Original/arab/t1/'
t1_processed_path = 'Data/Processed/arab/t1/'

myfile = open(t1_original_path + 'j1-k0.txt', mode="r", encoding="utf8")
text_original = myfile.readlines()
myfile.close()

# text_original = "ومن ذلك التلحين في القرآن والآذان. وكذلك الاشتغال بدقائق الجدل والمناظرة من. "
text_processed = []
print("Original Text:\n")
for line in text_original:
    text_processed.append(pre_process.preprocess(line))
    print(line)

myfile = open(t1_processed_path + 'j1-k0.txt', mode="w", encoding="utf8")
print("Processed Text:\n")
for line in text_processed:
    print(line)
    myfile.write(line + '\n')

myfile.close()

text_processed_str = ' '.join(map(str, text_processed))
# create tensor id's and tokenize the input
inputs = tokenizer.encode_plus(text_processed_str, return_tensors='pt')
print("Input ID's:")
print(inputs['input_ids'][0])
print(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))

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