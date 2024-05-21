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
if False:
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

vector_start_time = time.time()
test_vector = FAISS.load_local(faiss_save_p,bge_embeddings,allow_dangerous_deserialization=True)
vector_end_time = time.time()
print(f"all_cache:{(vector_end_time-vector_start_time)*1000}ms")





if False:
    import json
    import requests
    from typing import Any, List, Mapping, Optional
    from langchain.llms.base import LLM
    from langchain.callbacks.manager import CallbackManagerForLLMRun

    class CustomLLM(LLM):
        n: int
        # endpoint: str = "http://localhost:5000"
        # endpoint: str = "http://127.0.0.1:4004"
        # endpoint: str = "http://192.168.110.31:4004"
        endpoint: str = "http://192.168.111.1:4004"
        model: str = "chatglm2-6b"
        def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            callbacks: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
        ) -> str:
            headers = {"Content-Type": "application/json"}
            data = {"model": self.model, "messages": [{"role": "user", "content": prompt}]}
            # print(f"data:{data}")
            # response = requests.post(f"{self.endpoint}/api/sumHundredTime", headers=headers, json=data)
            response = requests.post(f"{self.endpoint}/api/sumHundredTime?question={prompt}")
            result = response.content.decode()
            json_data = json.loads(result)
            text = json_data["data"]
            return text
        
        @property
        def _llm_type(self) -> str:
            return "chatglm2-6b"
        
        @property
        def _identifying_params(self) -> Mapping[str, Any]:
            """Get the identifying parameters."""
            return {"n": self.n}

    llm = CustomLLM(n=10)
elif False:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI()
elif True:

    from langchain_community.llms.chatglm3 import ChatGLM3
    from langchain.schema.messages import AIMessage
    # endpoint_url = "http://127.0.0.1:8000/v1/chat/completions"
    # endpoint_url = "http://0.0.0.0:8000/v1/chat/completions"
    endpoint_url = "http://127.0.0.1:8080/v1/chat/completions"
    # messages = [
    #     AIMessage(content="我将从美国到中国来旅游，出行前希望了解中国的城市"),
    #     AIMessage(content="欢迎问我任何问题。"),
    # ]
    llm = ChatGLM3(
        endpoint_url=endpoint_url,
        max_tokens=80000,
        #prefix_messages=messages,
        top_p=0.9,
    )
    
else: 
    from langchain_community.llms import Ollama
    llm = Ollama(model="llama2")



if True:
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你现在是一个车载智能助手，用中文回答"),
        ("user", "{input1}"),
        ("user", "{input2}")
        
    ])

else:
    from langchain import PromptTemplate
    template = """
    你现在是一个车载智能助手,
    根据:
    {input2} 
    回答'{input1}'的相关问题。
    """
    prompt = PromptTemplate.from_template(template)
chain = prompt | llm 

# 加载
flag =True
while(flag):

    query = input("请输入agent:")
    embed_search_start_time = time.time()
    # query_vector = bge_embeddings.embed_query(query)
    # docs1 = vector.similarity_search_by_vector(query_vector,k=1)
    docs2 = test_vector.similarity_search(query,k = 2)     # similarity_search = embed_query+ similarity_search_by_vector
    embed_search_end_time = time.time()
    print(f"fdocs2:{docs2}")
    print("##################")

    print(f"embed_query_searchtime:{(embed_search_end_time-embed_search_start_time)*1000}ms")
    response_start_time = time.time()
    response =chain.invoke({"input1": query, "input2": docs2[0].page_content})
    response_end_time = time.time()
    print(f"response:{response}")
    print(f"response_time:{(response_end_time-response_start_time)*1000}ms")