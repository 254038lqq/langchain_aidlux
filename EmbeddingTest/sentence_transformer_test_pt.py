from sentence_transformers import SentenceTransformer
# over the wall
import os
if True:
    os.environ['https_proxy'] = 'http://192.168.110.222:7890'
    os.environ['http_proxy'] = 'http://192.168.110.222:7890'
    os.environ['all_proxy'] = 'socks5://192.168.110.222:7890'

model = SentenceTransformer("all-MiniLM-L6-v2")

# Our sentences to encode
sentences = [
    "This framework generates embeddings for each input sentence",
    "Sentences are passed as a list of string.",
    "The quick brown fox jumps over the lazy dog."
]

# Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)

# Print the embeddings
for sentence, embedding in zip(sentences, embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")