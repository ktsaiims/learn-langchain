import requests
from dotenv import load_dotenv
from os import getenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

model = init_chat_model(
    model='gpt-4.1-mini',
    temperature=0.1
)

for chunk in model.stream('Hi what is python?'):
    print(chunk.text, end='', flush=True)
