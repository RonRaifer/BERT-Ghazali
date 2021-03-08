from transformers import AutoTokenizer, AutoModel
from preprocess import ArabertPreprocessor

model_name = "aubmindlab/bert-base-arabertv2"
pre_process = ArabertPreprocessor(model_name=model_name)
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

text_original = "ومن ذلك التلحين في القرآن والآذان. وكذلك الاشتغال بدقائق الجدل والمناظرة من. "
text_processed = pre_process.preprocess(text_original)
print("Original Text:\n" + text_original)
print("Processed Text:\n" + text_processed)

# create tensor id's and tokenize the input
inputs = tokenizer.encode_plus(text_processed, return_tensors='pt')
print("Input ID's:")
print(inputs['input_ids'][0])
print(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))

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
