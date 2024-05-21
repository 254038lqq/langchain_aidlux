import os
# from langchain import OpenAI
# from langchain.llms import OpenAI
# from langchain_community.llms import OpenAI
# from langchain_openai import OpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain_openai import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough
# from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

if True:
    import os
    os.environ['https_proxy'] = 'http://192.168.110.222:7890'
    os.environ['http_proxy'] = 'http://192.168.110.222:7890'
    os.environ['all_proxy'] = 'socks5://192.168.110.222:7890'
    os.environ["OPENAI_API_KEY"] = ""


# from langchain_community.embeddings import OllamaEmbeddings
    
# embeddings = OllamaEmbeddings()
    
embeddings = OpenAIEmbeddings()
persist_directory = '/home/aidlux/langchain_pt/chroma'
# 加载数据库
vdb = Chroma(
    persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
    embedding_function=embeddings
)
query = "我应该怎么擦除个人数据"
docs = vdb.similarity_search(query)
print(docs)

