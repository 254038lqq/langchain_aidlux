import os
import openai
# import sys
# sys.ad
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

import os
os.environ['https_proxy'] = 'http://192.168.110.222:7890'
os.environ['http_proxy'] = 'http://192.168.110.222:7890'
os.environ['all_proxy'] = 'socks5://192.168.110.222:7890'

from langchain.chains import SimpleSequentialChain
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector

from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

# llm = ChatOpenAI(temperature=0.9)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=llm, prompt=default_prompt)

# 1.welcome 
welcome_prompt = ChatPromptTemplate.from_template(
    # "帮我生成一句有关欢迎来到{Restaurant name}的句子。在你的{Restaurant name}中有各种口味的汉堡，小吃与饮品。"
    
    "Help me generate a sentence about welcoming to {Restaurant name}. In your {Restaurant name}, there are various flavors of burgers, snacks, and beverages."
)
#  欢迎您光临蜜雪冰城！请随意挑选您喜欢的口味，我们将竭诚为您服务。
# Restaurant_name = "蜜雪冰城"
Restaurant_name = "Honey Snow City"

welcome_chain = LLMChain(llm=llm, prompt=welcome_prompt)



# 2.eat in or take away
dine_or_go_examples = [
    {"input": "I want to take", "output": "take away"},
    {"input": "I'll eat here", "output": "eat in"},
    # {"input": "I'll eat at your place", "output": "eat in"},
    {"input": "Take it to work", "output": "take away"},
    {"input": "I'd like a coke, please", "output": "Please let me know if you would like to eat in or take out so that I can arrange a table for you"},   
]
dine_or_go_example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)

dine_or_go_example_selector = LengthBasedExampleSelector(
    examples=dine_or_go_examples,
    example_prompt=dine_or_go_example_prompt,
    max_length=25,
)
dine_or_go_dynamic_prompt = FewShotPromptTemplate(
    example_selector=dine_or_go_example_selector,
    example_prompt=dine_or_go_example_prompt,
    prefix="Judge my intentions:eat in  or take away(eat in means eating in a restaurant,take away  means not to eat in the restaurant, to take away food).If you cannot determine, please output:'Please let me know if you would like to 'eat in' or 'take away', So that I can determine whether I need to arrange a table for you.'",
    suffix="Input: {DineOrGo}\nOutput:",
    input_variables=["DineOrGo"],
)

dine_or_go_chain = dine_or_go_dynamic_prompt | llm 
# response = welcome_chain.invoke({"Restaurant name": Restaurant_name})
# print(f"response:{response}")



# 3. 用餐人数
diner_numbers_examples = [
    {"input": "There are three of us", "output": "three"},
    {"input": "Dining for five", "output": "five"},
    {"input": "we have * people", "output": "*"},
    {"input": "3", "output": "three"},
    {"input": "I'd like a coke, please", "output": "Please let me know the number of diners so that we can arrange a table for you."},   
]
diner_numbers_example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)

diner_numbers_example_selector = SemanticSimilarityExampleSelector.from_examples(
    diner_numbers_examples,
    OpenAIEmbeddings(),
    Chroma,
    k=2,
)
diner_numbers_dynamic_prompt = FewShotPromptTemplate(
    example_selector=diner_numbers_example_selector,
    example_prompt=diner_numbers_example_prompt,
    prefix="Obtain the number of diners.If you cannot obtain, please output:'Please let me know the number of diners so that we can arrange a table for you.'",
    suffix="Input: {diner_numbers}\nOutput:",
    input_variables=["diner_numbers"],
)
diner_numbers_chain = diner_numbers_dynamic_prompt | llm


# 4.添加菜单相关的知识向量库
# 4.1
embeddings = OpenAIEmbeddings()

# print(embeddings)
# print(fhgh)
store = LocalFileStore("/home/aidlux/langchain/englishcache")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    embeddings, 
    store, 
    namespace=embeddings.model
)
file_path = "/home/aidlux/langchain/englishcaidanjson.txt"
try:
    loader = TextLoader(file_path,encoding='utf-8')
    raw_documents = loader.load()
except Exception as e:
    print(f"Error loading file: {e}")
    raise 
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)  
documents = text_splitter.split_documents(raw_documents)

import time
vector_start_time = time.time()
vector = FAISS.from_documents(documents, cached_embedder)
vector_end_time = time.time()
print(f"all_time:{(vector_end_time-vector_start_time)*1000}ms")
retriever = vector.as_retriever(
    search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5}
)

# query = "what information of the Iced Lemon Tea"
# docs = vector.similarity_search(query)
# print(docs)
# # docs = retriever.get_relevant_documents("what information of the Iced Lemon Tea")
# # print(docs)
# print(xgssdbz)
recommend_retriever_tool = create_retriever_tool(
    retriever,
    "recommend_assistant",
    "When receiving inquiries related to burger recommendations, snack recommendations, beverage recommendations, or any queries related to food recommendations or placing orders, you must use this tool. Please refrain from creating your own answers."
)
# order_retriever_tool = create_retriever_tool(
#     retriever,
#     "Ordering_assistant",
#     "You must use this tool when you receive input related to ordering food.Extract the following information from the input: dish_name: the name of the dish, if the dish name is not found, output -1.dish_quantity: the number of dishes, if not, output -2.dish_price: the price of dishes, if not, output -3. The output contains the following three keys in json format:Dish_name:{dish_name},Dish_quantity:{dish_quantity}{dish_unit},Dish_price:{dish_price}$"
# )

# order_retriever_tool = create_retriever_tool(
#     retriever,
#     "Ordering_assistant",
#     "You must use this tool when you receive input related to ordering food.Extract the following information from the input: dish_name: the name of the dish, if the dish name is not found, output:'Sorry,we don't have{dish_name},but we have {similarproduct}'.dish_quantity: the number of dishes, if not, take the default value 1.dish_price: the price of dishes, if not, output -1. dish_type: the type of dishes, if not, output -2. The output contains the following three keys in json format:dish_name:{dish_name},dish_quantity:{dish_quantity}{dish_unit},dish_price:{dish_price}$,dish_type:{dish_type}"
# )
# The output contains the following three keys in json format:dish_name:{dish_name},dish_quantity:{dish_quantity}{dish_unit},dish_price:{dish_price}$,dish_type:{dish_type}
# order_retriever_tool = create_retriever_tool(
#     retriever,
#     "Ordering_assistant",
#     "Extract the following information from the input: {dish_name} and {quantity}.  You must use this tool to retrieve {dish_name} and recommend a similar available Dish Name in the database. Then use the {dish_name} to find out the price and menu type of an {dish_name} from {retriever}. If no Dish Name is matched, output: 'Sorry, we don't have {dish_name},' Summarize the menu information ordered by the user. The menu information needs to include dish name, dish quantity, dish price, menu type. Example:the customer wants 2 Strawberry milkshakes at a price of 3.99 each, and it belongs to the Drinks menu type"
# )
# Extract the following information from the input: {dish_name} and {quantity}.  You must use this tool to retrieve {dish_name} and recommend a similar available Dish Name in the database. Then use the {dish_name} to find out the price and menu type of an {dish_name} from {retriever}. If no dish name is matched, output: 'Sorry, we don't have {dish_name},' If matched a similar dish name, summarize the menu {information} ordered by the user and convert it into a fixed format. The format is: {dish_name}, {quantity}, {price}. Unobtained information defaults to -1. Example: Classic Cola, 1, 0.5$
order_retriever_tool = create_retriever_tool(
    retriever,
    "Ordering_assistant",
    "Extract the following information from the input: {dish_name} and {quantity}. If not extract the quantity information need Use the default value 1. You must use this tool to retrieve {dish_name} and recommend a similar available Dish Name in the database. Then use the {dish_name} to find out the price and menu type of an {dish_name} from {retriever}. If no Dish Name is matched, output: 'Sorry, we don't have {dish_name},' Summarize the menu information ordered by the user. Convert it {information} into a fixed format, The format is: dish_name:{dish_name},dish_quantity:{dish_quantity},dish_price:{dish_price},dish_type:{dish_type}"
)
# tools = [recommend_retriever_tool,order_retriever_tool]
tools = [order_retriever_tool]
tools2 = [recommend_retriever_tool]

from langchain.prompts.prompt import PromptTemplate

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
order_prompt = ChatPromptTemplate.from_messages(
    [   
        # 您是一个食品推荐助手。在做推荐时，请从三种菜单类型中随机选择一种项目，包括汉堡、小吃和饮料，组成一份套餐。在做出推荐后，请使用以下格式进行输出：“我们这里有各种美味的汉堡、饮料和小吃。我们已经为您推荐了 、 和 *。您还需要其他什么吗？您可以选择下单或告诉我您的口味偏好，这样我就可以继续为您推荐更多美味的食物。”
        # ("system", "You are a food recommendation assistant. When making recommendations, please randomly select one item from each of the three Menu Types: Burgers, Snacks, and Beverages, to form a combo meal. After making a recommendation, please use the following format for output: 'We have a variety of delicious burgers, beverages, and snacks here. We have already recommended * and * and * for you. Do you need anything else? You can choose to place an order or tell me your taste preferences, so I can continue to recommend more delicious food for you.'"),
        # 你是一个食品推荐助手。
        #  After making a recommendation, please use the following format for output: 'We have a variety of delicious burgers, beverages, and snacks here." 
        #     +"Do you need anything else? You can choose to place an order or tell me your taste preferences, so I can continue to recommend more delicious food for you.'"),

        ("system", "You are a ordering assistant.Please extract dish_name,dish_quantity,dish_priece,dish_type from the input of the customer."),
        # MessagesPlaceholder(variable_name='chat_history', optional=True),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
        
    ]
)

# print(f"react_prompt::{react_prompt}")
# react_agent = create_react_agent(llm, tools, react_prompt1)
# agent_executor = AgentExecutor(agent=react_agent, tools=tools, verbose=True,handle_parsing_errors=True,max_iterations=10)

order_agent = create_openai_functions_agent(llm, tools, order_prompt)
agent_executor = AgentExecutor(agent=order_agent, tools=tools, verbose=True,handle_parsing_errors=True,)

recommend_agent = create_openai_functions_agent(llm, tools, order_prompt)
recommend_agent_executor = AgentExecutor(agent=recommend_agent, tools=tools2, verbose=True,handle_parsing_errors=True,)

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
message_history = ChatMessageHistory()
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    # This is needed because in most real world scenarios, a session id is needed
    # It isn't really used here because we are using a simple in memory ChatMessageHistory
    lambda session_id: message_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)


# 03
# retriever_tool = create_retriever_tool(
#     retriever,
#     "Ordering_assistant",
#     "When receiving inquiries related to burger recommendations, snack recommendations, beverage recommendations, or any queries related to food recommendations or placing orders, you must use this tool. Please refrain from creating your own answers."
# )
# tools = [retriever_tool]

# dish_recommendation_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a food recommendation assistant.Each recommendation should have some randomness to it, aiming to recommend different items each time."),
#         # MessagesPlaceholder(variable_name='chat_history', optional=True),
#         ("user", "{input}"),
#         MessagesPlaceholder(variable_name="agent_scratchpad")
        
#     ]
# )
# dish_recommendation_agent = create_openai_functions_agent(llm, tools, dish_recommendation_prompt)
# dish_recommendation_agent_executor = AgentExecutor(agent=dish_recommendation_agent, tools=tools, verbose=True)

# message_history = ChatMessageHistory()
# dish_recommendation_agent_with_chat_history = RunnableWithMessageHistory(
#     dish_recommendation_agent_executor,
#     # This is needed because in most real world scenarios, a session id is needed
#     # It isn't really used here because we are using a simple in memory ChatMessageHistory
#     lambda session_id: message_history,
#     input_messages_key="input",
#     history_messages_key="chat_history",
# )
from langchain.prompts.example_selector.ngram_overlap import NGramOverlapExampleSelector
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
    # diner_numbers_input = input("(order)：")
    # diner_numbers = add_del_intentation_chain.invoke({"sentence":diner_numbers_input })


del_examples = [
    {"input": "please give me one apple", "output": "dish_name:apple,dish_quantity:1"},
    {"input": "I don't want any more watermelon", "output": "dish_name:watermelon,dish_quantity:1"},
]

del_example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)
# print(f"example_prompt:{example_prompt}")

del_example_selector = SemanticSimilarityExampleSelector.from_examples(
    # The list of examples available to select from.
    del_examples,
    # The embedding class used to produce embeddings which are used to measure semantic similarity.
    OpenAIEmbeddings(),
    # The VectorStore class that is used to store the embeddings and do a similarity search over.
    Chroma,
    # The number of examples to produce.
    k=1,
)
similar_prompt = FewShotPromptTemplate(
    # We provide an ExampleSelector instead of examples.
    example_selector=del_example_selector,
    example_prompt=del_example_prompt,
    prefix="Extract the following information from the input: dish_name and quantity. If not extract the quantity information need Use the default value 1.",
    suffix="Input: {adjective}\nOutput:",
    input_variables=["adjective"],
)

del_chain = similar_prompt | llm 

# # 1.欢迎 + 2.堂食/外带  +  3.用餐人数  +  4.菜品推荐与点餐使用一个执行器
# 客户点到没有的菜品的时候，给推荐相近的产品
# 没有数量的话默认为一份
# 对输出做解析  添加到菜单中
while True:
    welcome_response = welcome_chain.invoke({"Restaurant name": Restaurant_name})
    print(welcome_response)
    dine_or_go_sign = 1
    while dine_or_go_sign:
        dine_or_go_input = input("(Would you like to eat in or take away?)：")
        intent = dine_or_go_chain.invoke({"DineOrGo": dine_or_go_input})
        print("Parsed intent:", intent)
        if intent.content in ['eat in', 'take away']:
            dine_or_go_sign = 0
    if intent.content ==  'eat in':
        diner_numbers_sign = 1
        while diner_numbers_sign:
            diner_numbers_input = input("(Please tell me the number of diners)：")
            diner_numbers = diner_numbers_chain.invoke({"diner_numbers":diner_numbers_input })
            print(f"numbersp:{diner_numbers}")   
            if diner_numbers.content == 'Please let me know the number of diners so that we can arrange a table for you.':
                continue
            else:
                diner_numbers_sign = 0
        print("I have arranged a table for you at table number six.")
    
    print( "We have a variety of delicious burgers, beverages, and snacks here. You can choose to place an order or tell me your taste preferences, so I can continue to recommend more delicious food for you.")
    order_completed_sign = 1
    order_details = []
    while order_completed_sign:
        extracted_info = {}
        query =input("(Enter 'END',means order completed):")
        if query.lower() == "end" or query.upper() == "end":
            # 
            print("Thank you for coming. Enjoy your meal")
            break
        else:
            response  = agent_executor.invoke({"input": query},
                                  config={"configurable": {"session_id": "<foo>"},
                                          }
                                  )
            print(f"response:{response}")
            # print(f"response:{response['output']}")

            input_text = response['input']
            output_lines = response['output'].split(',')
            # ['dish_name:Iced Coffee', 'dish_quantity:1', 'dish_price:2.50', 'dish_type:Drinks']
            for line in output_lines:
                key, value = line.split(':')
                key = key.strip()
                value = value.strip()
                if key == 'dish_quantity':
                    now_quantity = int(value)
                 # 检查是否是新的订单中已存在的dish_name
                if key == 'dish_name' and value in [order['dish_name'] for order in order_details]:
                    # 如果是相同的dish_name，则在已存在的订单中更新数量
                    for order in order_details:
                        if order['dish_name'] == value:
                            # print(order['dish_quantity'])
                            # print(type(order['dish_quantity']))
                            order['dish_quantity'] = int(order['dish_quantity']) + now_quantity  # 如果没有提取到数量，默认加1
                else:
                    # 如果是新的dish_name，将提取的信息添加到字典中
                    extracted_info[key] = value
                    # print(extracted_info)
            if 'dish_name' in extracted_info:
                order_details.append(extracted_info)
        print(order_details)

    
    pay_or_continue_sign =1
    while pay_or_continue_sign:
        print(f"You have now ordered the {order_details},Enter to confirm payment Or edit the shopping cart")
        pay_or_continue_input = input("(Enter to confirm payment):")
        if pay_or_continue_input == "confirm":
            total_price = 0
            for item in order_details:
                quantity = float(item['dish_quantity'])
                price = float(item['dish_price'])   

                single_item_total = quantity * price

                total_price += single_item_total

            # print("Total price: ", total_price)
            print(f"You have now ordered the {order_details}.You need to pay the total {total_price}$")
            pay_or_continue_sign = 0
        else:
            # continue_input =input("(Please enter your operation):")
            add_del_intention_response = add_del_intentation_chain.invoke({"sentence":pay_or_continue_input })
            print(f"add_del_intention_response:{add_del_intention_response}")
            add_del_intention_response_str = str(add_del_intention_response)
            start_index = add_del_intention_response_str.index("'") + 1
            end_index = add_del_intention_response_str.index("'", start_index)
            add_del_intention = add_del_intention_response_str[start_index:end_index]
            print(f"add_del_intention{add_del_intention}")

            if add_del_intention == "add":
                response  = agent_executor.invoke({"input": pay_or_continue_input},
                                    config={"configurable": {"session_id": "<foo>"},
                                            }
                                    )
                print(f"response:{response}")
                # print(f"response:{response['output']}")

                input_text = response['input']
                output_lines = response['output'].split(',')
                # ['dish_name:Iced Coffee', 'dish_quantity:1', 'dish_price:2.50', 'dish_type:Drinks']
                for line in output_lines:
                    key, value = line.split(':')
                    key = key.strip()
                    value = value.strip()
                    if key == 'dish_quantity':
                        now_quantity = int(value)
                    # 检查是否是新的订单中已存在的dish_name
                    if key == 'dish_name' and value in [order['dish_name'] for order in order_details]:
                        # 如果是相同的dish_name，则在已存在的订单中更新数量
                        for order in order_details:
                            if order['dish_name'] == value:
                                # print(order['dish_quantity'])
                                # print(type(order['dish_quantity']))
                                order['dish_quantity'] = int(order['dish_quantity']) + now_quantity  # 如果没有提取到数量，默认加1
                    else:
                        # 如果是新的dish_name，将提取的信息添加到字典中
                        extracted_info[key] = value
                        # print(extracted_info)
                if 'dish_name' in extracted_info:
                    order_details.append(extracted_info)
            elif add_del_intention == "del":
                # intent = del_chain.invoke({"adjective": pay_or_continue_input})
                del_chain_response = del_chain.invoke({"adjective": pay_or_continue_input})
                output_lines = del_chain_response.split(',')
                print(output_lines)

            else:
                continue

    
    

            # input
            # 判断 input的意图，加餐还是减餐
            # if +  不知道加几个  还得写一个



    # 判断是不是有小吃
    snacks_exist = any(order['dish_type'] == 'snacks' for order in order_details)
    drink_exist = any(order['dish_type'] == 'Drinks' for order in order_details)
    
    if not snacks_exist and not drink_exist:
        recommend_drink_and_snacks = "Please recommend a snack and a beverage for me."
    elif not snacks_exist:
        recommend_drink_and_snacks = "Please recommend a snack for me."
    elif not drink_exist:
        recommend_drink_and_snacks = "Please recommend a drink for me."
    
    response_recommend_drink_and_snacks  = recommend_agent_executor.invoke({"input": recommend_drink_and_snacks},
                                  config={"configurable": {"session_id": "<foo>"},
                                          }
                                  )
    print(response_recommend_drink_and_snacks)




# take away
# please give me 5 Iced Coffee
# 
    
