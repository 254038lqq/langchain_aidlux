# 参考：https://blog.csdn.net/qiaotl/article/details/134378276
from langchain_community.document_loaders import PyPDFLoader
from Crypto.Cipher import AES
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_text_splitters.base import Language, TextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from typing import List, Optional
import openai
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

import os
os.environ['https_proxy'] = 'http://192.168.110.222:7890'
os.environ['http_proxy'] = 'http://192.168.110.222:7890'
os.environ['all_proxy'] = 'socks5://192.168.110.222:7890'

class MyEmbeddings:
    def __init__(self):
        # self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        from sentence_transformers import SentenceTransformer
        # self.model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.model.encode(t).tolist() for t in texts]
        
    def embed_query(self, text: str) -> Optional[List[float]]:  
        return self.model.encode(text).tolist()
# from test import MyEmbeddings
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings

# 1.DocumentLoader
loader = PyPDFLoader("/home/aidlux/langchain_pt/EmbeddingTest/Owners_Manual.pdf")
# pages = loader.load_and_split()
pages = loader.load()
# print(pages)


#2. TextSplitter
# 2.1
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(pages)
# print(texts)
#2.2
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# texts = text_splitter.split_documents(pages)
# print(texts)
#2.3
# python_splitter = RecursiveCharacterTextSplitter.from_language(
#     language = Language.PYTHON, chunk_size=1000, chunk_overlap=100
# )
# texts = python_splitter.split_documents(pages)
# print(texts)


# 3.Embedding model
# embeddings = MyEmbeddings()
embeddings = OpenAIEmbeddings()
store = LocalFileStore("/home/aidlux/langchain_pt/faisscache")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    embeddings, 
    store, 
    namespace=embeddings.model
)

# 保存
import time
vector_start_time = time.time()
vector = FAISS.from_documents(texts, cached_embedder)
vector_end_time = time.time()
print(f"all_time:{(vector_end_time-vector_start_time)*1000}ms")


# 加载
query = "我应该怎么擦除个人数据"
embed_search_start_time = time.time()
query_vector = embeddings.embed_query (query)
docs1 = vector.similarity_search_by_vector(query_vector, k=1)
embed_search_end_time = time.time()
# docs.page_content
print(f"docs1:{docs1}")
print(f"embed_query_searchtime:{(embed_search_end_time-embed_search_start_time)*1000}ms")


# str_search_start_time = time.time()
# docs0 = vector.similarity_search(query)
# str_search_end_time = time.time()
# # print(f"docs0[0].page_content:{docs0[0].page_content}")
# print(f"docs0:{docs0}")
# print(f"str_query_searchtime:{(str_search_end_time-str_search_start_time)*1000}ms")


# db = Chroma.from_documents(
#     texts, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
# db = Chroma.from_documents(texts,OpenAIEmbeddings)