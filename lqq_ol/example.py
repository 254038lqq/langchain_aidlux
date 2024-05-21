# 使用 LangChain 来集成自定义的 LLM 以及其中的实现原理。

import requests
from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
import json

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


# exampl select

# from langchain.prompts import (
#     ChatPromptTemplate,
#     FewShotChatMessagePromptTemplate,
# )
# examples = [
#     {"input": "2+2", "output": "4"},
#     {"input": "2+3", "output": "5"},
# ]
# example_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("human", "{input}"),
#         ("ai", "{output}"),
#     ]
# )
# few_shot_prompt = FewShotChatMessagePromptTemplate(
#     example_prompt=example_prompt,
#     examples=examples,
# )

# print(few_shot_prompt.format())
# final_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a wondrous wizard of math."),
#         few_shot_prompt,
#         ("human", "{input}"),
#     ]
# )
# # from langchain_community.chat_models import ChatAnthropic
# # from langchain_openai import ChatOpenAI
# # chain = final_prompt | ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
# chain = final_prompt | CustomLLM(n=10)

# # chain.invoke({"input": "What's the square of a triangle?"})
# chain.invoke({"input": "5+6"})
# # chain.invoke({"5+6"})


# # 1.set OPENAI_API_KEY
# import os
# import openai
# from dotenv import load_dotenv, find_dotenv
# _ = load_dotenv(find_dotenv()) # read local .env file
# openai.api_key = os.environ['OPENAI_API_KEY']


# # 2.运行 VPN 设置代码
# os.environ['https_proxy'] = 'http://192.168.110.222:7890'
# os.environ['http_proxy'] = 'http://192.168.110.222:7890'
# os.environ['all_proxy'] = 'socks5://192.168.110.222:7890'

# from langchain.chat_models import ChatOpenAI
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
from langchain import PromptTemplate


template = """/
判断顾客的就餐意图：'堂食'还是'外带'
'堂食'表示顾客在店内就餐；'外带'表示顾客想要带走菜品，不在店内就餐。
请从顾客的输入{input}中判断顾客的就餐地点意图是“堂食”还是“外带”，如果可以解析到意图，仅返回“堂食”或者“外带”。如果没有解析到意图，请输出“您选择堂食还是外带？”
"""
prompt = PromptTemplate.from_template(template)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "eat in means eating in a restaurant. take away means not to eat in the restaurant, to take the dishes to take away to eat elsewhere.Please determine from the customer's input {input} whether the customer's intention is 'eat-in' or 'take-away'. If the intention can be resolved, only return 'eat-in' or 'take-away'. If the intention is not resolved, output 'Do you prefer to eat-in or take-away?'"),
#         ("user", "{input}"),
#         # MessagesPlaceholder(variable_name="agent_scratchpad")
        
#     ]
# )
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "Judge my intentions:eat in  or take away(eat in means eating in a restaurant,take away  means not to eat in the restaurant, Take the food and eat it somewhere else).If you cannot determine, please output:'Please let me know if you would like to 'eat in' or 'take away', So that I can determine whether I need to arrange a table for you.'example:eat in"),
#         ("user", "{input}"),
#         # MessagesPlaceholder(variable_name="agent_scratchpad")
        
#     ]
# )
# 
from langchain.schema.output_parser import StrOutputParser
output_parser = StrOutputParser()

#创建一个简单链
# chain = prompt | llm | output_parser
chain = prompt | llm 

while True:
    input_text = input("input:")
    intent =chain.invoke({"input": input_text})
    print("解析出的意图为：", intent)

