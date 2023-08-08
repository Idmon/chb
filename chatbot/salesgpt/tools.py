from langchain.agents import Tool
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma
import requests
import tiktoken
from sympy import N, sympify

import os
import base64
from PIL import Image
import io
import json

tokenizer = tiktoken.get_encoding('cl100k_base')

def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

def get_product_catalog(url):
    url = url
    response = requests.get(url)

    # Ensure the request was successful
    if response.status_code == 200:
        text = response.text
        return text
    else:
        print(f'Request failed with status {response.status_code}')

def setup_knowledge_base(product_catalog: str = None):
    """
    We assume that the product catalog is simply a text string.
    """
    if product_catalog is None:
        product_catalog = get_product_catalog('https://pastebin.com/raw/h32LNF7w')
    elif 'http' in product_catalog:
        product_catalog = get_product_catalog(product_catalog)
    else:
        # load product catalog
        with open(product_catalog, "r") as f:
            product_catalog = f.read()

    text_splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0)
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=10,
    #     chunk_overlap=0,
    #     length_function=tiktoken_len,
    #     separators=['\n\n', '\n', ' ','']
    # )
    texts = text_splitter.split_text(product_catalog)

    llm = OpenAI(temperature=0)
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings)

    # docsearch = Chroma.from_texts(
    #     texts, embeddings, collection_name="product-knowledge-base"
    # )

    knowledge_base = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=docsearch.as_retriever()
    )
    return knowledge_base


def calculator(expression):
    # Convert the string expression into a sympy expression
    expr = sympify(expression)
    # Calculate its numerical value
    result = N(expr)
    return result
     
         
    
def get_tools(knowledge_base, tool_names):
        
    available_tools  = [
        Tool(
            name="SearchLaws",
            func=knowledge_base.run,
            description="Handig wanneer je een vraag moet beantwoorden of moeten zoeken naar wetten uit het 'Auteursrechtboek'",
        ),
        Tool(
            name="SearchMenu",
            func=knowledge_base.run,
            description="Handig wanneer de klant vraagt om de menukaart, gerechten, prijzen, ingredienten, toppings.",
        ),
        Tool(
            name="SearchProducts",
            func=knowledge_base.run,
            description="useful for when you need to answer questions about product information",
        ),
        Tool(
            name="Calculator",
            func=calculator,
            description="(Convert first to Sympy expression) useful for when you need to calculate things. Make sure to convert the calculation query to a Sympy expression. (EXAMPLE) Action Input: (2 + 2*3/(4^2))",
        )
    ]

    return [tool for tool in available_tools if tool.name in tool_names]

