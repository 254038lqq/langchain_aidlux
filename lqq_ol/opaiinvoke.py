import os
os.environ["OPENAI_API_KEY"] = ""
os.environ["TAVILY_API_KEY"] = "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"

import os
os.environ['https_proxy'] = 'http://192.168.110.222:7890'
os.environ['http_proxy'] = 'http://192.168.110.222:7890'
os.environ['all_proxy'] = 'socks5://192.168.110.222:7890'


from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

response = llm.invoke("你好")


print(response)
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector.ngram_overlap import NGramOverlapExampleSelector
from langchain import PromptTemplate
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)
add_del_examples = [
    {"input": "please give me *", "output": "add"},
    {"input": "Please give me another *", "output": "add"},
    {"input": "Please give me * more  *", "output": "add"},
    {"input": "Please reduce one *", "output": "del"},
    {"input": "I don't want any more *", "output": "del"},
    {"input": "I don't want any more *", "output": "del"},

]
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

add_del_intentation_chain = add_del_dynamic_prompt | llm

input = "give me an apple"
response = add_del_intentation_chain.invoke( {"sentence": input},)
print(response)