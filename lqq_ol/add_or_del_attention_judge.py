# 1.set OPENAI_API_KEY
import os
import openai
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']


# 2.运行 VPN 设置代码
import os
os.environ['https_proxy'] = 'http://192.168.110.222:7890'
os.environ['http_proxy'] = 'http://192.168.110.222:7890'
os.environ['all_proxy'] = 'socks5://192.168.110.222:7890'

from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
from langchain import PromptTemplate


from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector.ngram_overlap import NGramOverlapExampleSelector

# template = """
# You are a standing-sitting intention judge helper, please determine from {input} whether I want to sit or stand, if you can parse the intention, please output 'sit' or 'stand'. If the intent is not resolved, output 'Sorry I can not resolve your intent, please re-enter.'
# """
# prompt = PromptTemplate.from_template(template)
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)
add_del_examples = [
    {"input": "Please give me *", "output": "add"},
    {"input": "Please give me another *", "output": "add"},
    {"input": "Please give me * more  *", "output": "add"},
    {"input": "Please reduce one *", "output": "del"},
    {"input": "I don't want any more *", "output": "del"},
    {"input": "I don't want any more *", "output": "del"},
]

from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector

from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

add_del_example_selector = NGramOverlapExampleSelector(
    examples=add_del_examples,
    example_prompt=example_prompt,
    threshold=-1.0,
)
add_del_dynamic_prompt = FewShotPromptTemplate(
    # We provide an ExampleSelector instead of examples.
    example_selector=add_del_example_selector,
    example_prompt=example_prompt,
    prefix="Judge my intentions: add or del.",
    suffix="Input: {sentence}\nOutput:",
    input_variables=["sentence"],
)
print(f"add_del_dynamic_prompt\n{add_del_dynamic_prompt}")
add_del_intentation_chain = add_del_dynamic_prompt | llm

# 写一个chain判断是 order 还是  recommend
while True:
    diner_numbers_input = input("(order)：")
    diner_numbers = add_del_intentation_chain.invoke({"sentence":diner_numbers_input })
    print(add_del_dynamic_prompt.format(sentence = diner_numbers_input))
    print(f"intentions:{diner_numbers}")