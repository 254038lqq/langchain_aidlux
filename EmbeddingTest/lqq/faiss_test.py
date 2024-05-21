# from FlagEmbedding import FlagModel
# sentences_1 = ["样例数据-1", "样例数据-2"]
# sentences_2 = ["样例数据-3", "样例数据-4"]
# model = FlagModel('BAAI/bge-large-zh-v1.5', 
#                   query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
#                   use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
# embeddings_1 = model.encode(sentences_1)
# embeddings_2 = model.encode(sentences_2)
# similarity = embeddings_1 @ embeddings_2.T
# print(similarity)

# from langchain.embeddings import HuggingFaceBgeEmbeddings
# model_name = "BAAI/bge-large-en-v1.5"
# model_kwargs = {'device': 'cpu'}
# encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
# model = HuggingFaceBgeEmbeddings(
#     model_name=model_name,
#     model_kwargs=model_kwargs,
#     encode_kwargs=encode_kwargs,
#     query_instruction="为这个句子生成表示以用于检索相关文章："
# )
# model.query_instruction = "为这个句子生成表示以用于检索相关文章："

# from sentence_transformers import SentenceTransformer
# sentences_1 = ["样例数据-1", "样例数据-2"]
# sentences_2 = ["样例数据-3", "样例数据-4"]
# model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
# embeddings_1 = model.encode(sentences_1, normalize_embeddings=True)
# embeddings_2 = model.encode(sentences_2, normalize_embeddings=True)
# similarity = embeddings_1 @ embeddings_2.T
# print(similarity)

# pip install chromadb==0.3.29

from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader
import time

class MyEmbeddings:
    def __init__(self):
        # self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('BAAI/bge-large-zh-v1.5')

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.model.encode(t).tolist() for t in texts]
        
    def embed_query(self, text: str) -> Optional[List[float]]:  
        return self.model.encode(text).tolist()

# from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# loader = TextLoader("/home/aidlux/langchain/tesila.txt")
# docs = loader.load()
loader_start_time =time.time()
loader = PyPDFLoader("/home/aidlux/langchain/EmbeddingTest/Owners_Manual.pdf")
# pages = loader.load_and_split()
pages = loader.load()
loader_end_time =time.time()
# print(pages)
print(f"loader_time:{(loader_end_time-loader_start_time)*1000}ms")



text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
texts = text_splitter.split_documents(pages)

# print(texts[0].page_content)

# from langchain.vectorstores import Chroma

from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
embeddings = MyEmbeddings()
# db = Chroma.from_documents(texts, embeddings)
db = FAISS.from_documents(texts, embeddings)

query = "我应该如何从前部控制后屏幕"

import time


embed_search_start_time = time.time()
query_vector = embeddings.embed_query (query)
docs1 = db.similarity_search_by_vector(query_vector, k=1)
embed_search_end_time = time.time()
# docs.page_content
print(f"docs1:{docs1}")
print(f"embed_query_searchtime:{(embed_search_end_time-embed_search_start_time)*1000}ms")


# str_search_start_time = time.time()
# docs0 = db.similarity_search(query)
# str_search_end_time = time.time()
# # print(f"docs0[0].page_content:{docs0[0].page_content}")
# print(f"docs0:{docs0}")
# print(f"str_query_searchtime:{(str_search_end_time-str_search_start_time)*1000}ms")




