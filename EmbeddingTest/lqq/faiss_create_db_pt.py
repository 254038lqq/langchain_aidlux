# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import DocArrayInMemorySearch
# from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.tools import Tool
# from pydantic.v1 import BaseModel, Field # <-- Uses v1 namespace
# from pydantic import Field, field_validator # pydantic v2  
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

import os
import platform
os_name = platform.system()
windows_flag =False
if os_name == 'Windows':
    windows_flag = True
    print("当前操作系统是 Windows")
else:
    print("当前操作系统不是 Windows")

if False:
    import openai
    from dotenv import load_dotenv, find_dotenv
    _ = load_dotenv(find_dotenv()) # read local .env file
    openai.api_key = os.environ['OPENAI_API_KEY']

    # os.environ['https_proxy'] = 'http://192.168.110.222:7890'
    # os.environ['http_proxy'] = 'http://192.168.110.222:7890'
    # os.environ['all_proxy'] = 'socks5://192.168.110.222:7890'
#/home/aidlux/huggingface_model/baai-bge-large-zh-v1.5
# model_name = "BAAI/bge-large-zh-v1.5"
model_name=r"D:\algorithm\ai_agent\models_huggingface\models--BAAI--bge-large-zh-v1.5\snapshots\c11661ba3f9407eeb473765838eb4437e0f015c0"
if windows_flag:
    # model_kwargs = {"device": "cpu"} 
    model_kwargs = {"device": "cuda"} 
else:
    model_kwargs = {"device": "cpu"} 
#cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, 
#ve, fpga, ort, xla, lazy, vulkan, mps, meta, hpu, mti
encode_kwargs = {"normalize_embeddings": True}
bge_embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

# bge_embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-zh-v1.5")  # 中文 v1.5
    
# bge_embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-en-v1.5")   # 英文v1.5
# bge_embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-m3")              # 多语言，大模型
# bge_embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-reranker-large")   # 中英文大模型

# bge_embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-zh")  # 嵌入式中文大模型
# bge_embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-zh")    # 嵌入式中文 小模型

# from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
# bge_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# result = bge_embeddings.embed_query("hi this is harrison")
# print(len(result))
# doc_result = bge_embeddings.embed_documents(["hi this is harrison"])

if False:
    import sys
    sys.path.append(r"C:\Users\pt\miniconda3\envs\langchain_0\Lib\site-packages")
    from langchain_community.vectorstores import DocArrayInMemorySearch
    vectordb = DocArrayInMemorySearch.from_texts(
        ["青蛙是食草动物",
        "人是由恐龙进化而来的",
        "熊猫喜欢吃天鹅肉",
        "1+1=5",
        "2+2=8",
        "3+3=9",
        "Gemini Pro is a Large Language Model was made by GoogleDeepMind",
        "A Language model is trained by predicting the next token"
        ],
        embedding=bge_embeddings 
    )
    
    # #创建检索器
    bge_retriever = vectordb.as_retriever(search_kwargs={"k": 1})

    result =bge_retriever.get_relevant_documents("恐龙")
    print(result)
    print("first ok!")


from langchain_community.document_loaders import PyPDFLoader
# 1.DocumentLoader
pdf_path = "./EmbeddingTest/Owners_Manual.pdf"
# pdf_path = "./EmbeddingTest/llm_kv.pdf"
loader = PyPDFLoader(pdf_path,headers=None,extract_images=False)
# pages = loader.load_and_split()
pages = loader.load()
from langchain.text_splitter import CharacterTextSplitter  #,RecursiveCharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
texts = text_splitter.split_documents(pages)


from langchain_community.vectorstores import FAISS
# db = FAISS.from_documents(texts, bge_embeddings)  
# print(db.index.ntotal)

# retriever = db.as_retriever()

# from langchain.chains import RetrievalQA
# from langchain.llms import OpenAI
# qa = RetrievalQA.from_chain_type(OpenAI(), chain_type="stuff", retriever=retriever)
# query = "What did the president say about Ketanji Brown Jackson"
# qa.run(query)

# db.save_local("./vector_fast/tmp_faiss/faiss_index")
# new_db = FAISS.load_local("./vector_fast/tmp_faiss/faiss_index/index.faiss", bge_embeddings)
# query = "清洗数据"
# docs = new_db.similarity_search(query)

# 保存
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
save_path ="./vector_fast/tmp_faiss/faisscache"
store = LocalFileStore(save_path)
cached_embedder = CacheBackedEmbeddings.from_bytes_store(bge_embeddings, store)

import time

vector_start_time = time.time()
vector = FAISS.from_documents(texts, cached_embedder)
vector_end_time = time.time()
print(f"all_time:{(vector_end_time-vector_start_time)*1000}ms")



# 加载
query = "个人数据"
embed_search_start_time = time.time()
query_vector = bge_embeddings.embed_query(query)
docs1 = vector.similarity_search_by_vector(query_vector, k=1)
embed_search_end_time = time.time()
# docs.page_content
print(f"docs1:{docs1}")
print(f"embed_query_searchtime:{(embed_search_end_time-embed_search_start_time)*1000}ms")
