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
print(llm.invoke("你好"))
# llm.invoke("nihao 。。。。。。。")
# print()
# print(llm("你好"))
# print(llm._llm_type)


from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector.ngram_overlap import NGramOverlapExampleSelector
# from langchain import PromptTemplate
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

]
add_del_example_selector = NGramOverlapExampleSelector(
    examples=add_del_examples,
    example_prompt=example_prompt,
    threshold=-1.0 +  1e-9,
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

# input = "Please reduce one apple"
# response = add_del_intentation_chain.invoke( {"sentence": input},)
# # 输出add
# print(response)

# print(add_del_dynamic_prompt)
while True:
    diner_numbers_input = input("(order)：")
    diner_numbers = add_del_intentation_chain.invoke({"sentence":diner_numbers_input })
    print(add_del_dynamic_prompt.format(sentence = diner_numbers_input))
    print(f"intentions:{diner_numbers}")