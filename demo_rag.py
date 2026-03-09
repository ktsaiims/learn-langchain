from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from pprint import pprint

load_dotenv()

embeddings = OpenAIEmbeddings(model='text-embedding-3-large')

texts = [
    'Apple makes good computers',
    'I love apples',
    'I think Apple is innovative',
    'I enjoy oranges',
    'I like Lenovo Thinkpads',
    'Linux is better than Apple',
    'Grapes are ok'
]

vector_store = FAISS.from_texts(texts, embedding=embeddings)

# pprint(vector_store.similarity_search('Apples are my favorite food', k=7))
pprint(vector_store.similarity_search('Windows is a medicore operating system', k=7))
