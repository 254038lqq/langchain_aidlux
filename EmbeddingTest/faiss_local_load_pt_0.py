# from langchain_core.tools import Tool
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import os
import time
import platform
os_name = platform.system()
windows_flag =False
if os_name == 'Windows':
    windows_flag = True
    print("当前操作系统是 Windows")
else:
    print("当前操作系统不是 Windows")
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



from langchain_community.vectorstores import FAISS
faiss_save_p = "./vector_fast/tmp_faiss/faisscache_0"

vector_start_time = time.time()
if False: # pdf
    from langchain.storage import LocalFileStore
    from langchain.embeddings import CacheBackedEmbeddings
    save_path ="./vector_fast/tmp_faiss/faisscache"
    store = LocalFileStore(save_path)
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(bge_embeddings, store)

    from langchain_community.document_loaders import PyPDFLoader
    # 1.DocumentLoader
    pdf_path = "./EmbeddingTest/Owners_Manual.pdf"
    # pdf_path = "./EmbeddingTest/llm_kv.pdf"
    loader_0 = PyPDFLoader(pdf_path,headers=None,extract_images=False)
    pages = loader_0.load()


    from langchain.text_splitter import CharacterTextSplitter  #,RecursiveCharacterTextSplitter
    text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=100) #separator ="\n\n"
    texts = text_splitter.split_documents(pages)

    # from langchain_text_splitters import RecursiveCharacterTextSplitter

    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    # texts = text_splitter.split_documents(pages)



    # vector_start_time = time.time()
    vector = FAISS.from_documents(texts, cached_embedder)
    vector.save_local(faiss_save_p)
    vector_end_time = time.time()
    print(f"all_cache:{(vector_end_time-vector_start_time)*1000}ms")
elif False:  # txt
    from langchain.storage import LocalFileStore
    from langchain.embeddings import CacheBackedEmbeddings
    save_path ="./vector_fast/tmp_faiss/faisscache_txt_03"
    store = LocalFileStore(save_path)
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(bge_embeddings, store)
    from langchain.document_loaders import TextLoader

    # 1.DocumentLoader
    # txt_path = r"D:\algorithm\ai_agent\langchain_pt\vector_fast\Owners_Manual.txt"
    # txt_path = r"D:\algorithm\ai_agent\langchain_pt\vector_fast\Owners_Manual315.txt"
    txt_path=r"D:\algorithm\ai_agent\langchain_pt\vector_fast\0319\Owners_Manual.txt"
    loader_0 = TextLoader(txt_path, encoding="utf-8")

    pages = loader_0.load()


    from langchain.text_splitter import CharacterTextSplitter  #,RecursiveCharacterTextSplitter
    text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=0) #separator ="\n\n"
    texts = text_splitter.split_documents(pages)

    # from langchain_text_splitters import RecursiveCharacterTextSplitter
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    # texts = text_splitter.split_documents(pages)

    # vector_start_time = time.time()
    vector = FAISS.from_documents(texts, cached_embedder)
    vector.save_local(faiss_save_p)
    vector_end_time = time.time()
    print(f"all_cache:{(vector_end_time-vector_start_time)*1000}ms")




vector_start_time = time.time()
# deserialize_from_bytes
test_vector = FAISS.load_local(faiss_save_p,bge_embeddings,allow_dangerous_deserialization=True) 
vector_end_time = time.time()
print(f"load:{(vector_end_time-vector_start_time)*1000}ms")



# 加载
flag =True
while(flag):

    query = input("请输入agent:")
    embed_search_start_time = time.time()
    # query_vector = bge_embeddings.embed_query(query)
    # docs1 = vector.similarity_search_by_vector(query_vector,k=1)
    # docs2 = test_vector.similarity_search_by_vector(query_vector) 
    docs2 = test_vector.similarity_search(query,k = 2)     # similarity_search = embed_query+ similarity_search_by_vector
    embed_search_end_time = time.time()
    print(f"fdocs2:{docs2}")
    print("##################")
    # docs.page_content
    # print(f"docs1:{docs1}")
    print(f"embed_query_searchtime:{(embed_search_end_time-embed_search_start_time)*1000}ms")
    