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
from langchain_openai import OpenAIEmbeddings
from typing import List, Optional

# 
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


# embeddings = MyEmbeddings()
embeddings = OpenAIEmbeddings()
# db = Chroma.from_documents(texts, embeddings)
# db = FAISS.from_documents(texts, embeddings)

if False:
    from langchain.vectorstores import DocArrayInMemorySearch
    #创建向量数据库
    vectordb = DocArrayInMemorySearch.from_texts(
        ["青蛙是食草动物",
        "人是由恐龙进化而来的。",
        "熊猫喜欢吃天鹅肉。",
        "1+1=5",
        "2+2=8",
        "3+3=9",
        "Gemini Pro is a Large Language Model was made by GoogleDeepMind",
        "A Language model is trained by predicting the next token"
        ],
        embedding=embeddings 
    )
    
    #创建检索器,让它每次只返回1条最相关的文档：search_kwargs={"k": 1}
    openai_retriever = vectordb.as_retriever(search_kwargs={"k": 1})

    openai_retriever.get_relevant_documents("恐龙")



# from langchain.vectorstores import Chroma

from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
# from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from Crypto.Cipher import AES

# loader = TextLoader("/home/aidlux/langchain/EmbeddingTest/Owners_Manual.pdf")
# docs = loader.load()
loader = PyPDFLoader("/home/aidlux/langchain_pt/EmbeddingTest/Owners_Manual.pdf")
# pages = loader.load_and_split()
pages = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(pages)

# print(texts[0].page_content)

persist_directory = '/home/aidlux/langchain_pt/chroma'

# 加载数据库
import time
t0 = time.time()
vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
)
# 将加载的向量数据库持久化到磁盘上
vectordb.persist()
t1 = time.time()
# vdb = FAISS.from_documents(texts, embeddings)
print(f"向量数据库构建完成,耗时{(t1-t0)*1000}ms")


# # 加载数据库
# vdb = Chroma(
#     persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
#     embedding_function=embeddings
# )

# query = "我应该怎么擦除个人数据"
# docs = vdb.similarity_search(query)



# query = "我应该如何从前部控制后屏幕"

# import time


# embed_search_start_time = time.time()
# query_vector = embeddings.embed_query (query)
# docs1 = db.similarity_search_by_vector(query_vector, k=1)
# embed_search_end_time = time.time()
# # docs.page_content
# print(f"docs1:{docs1}")
# print(f"embed_query_searchtime:{(embed_search_end_time-embed_search_start_time)*1000}ms")


# str_search_start_time = time.time()
# docs0 = db.similarity_search(query)
# str_search_end_time = time.time()
# # print(f"docs0[0].page_content:{docs0[0].page_content}")
# print(f"docs0:{docs0}")
# print(f"str_query_searchtime:{(str_search_end_time-str_search_start_time)*1000}ms")




