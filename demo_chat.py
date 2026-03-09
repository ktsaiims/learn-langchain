from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

model = init_chat_model(
    model='gpt-4.1-mini',
    temperature=0.1
)

conversation = [
    SystemMessage('You are a computer gremlin who answers in a snarky manner'),
    HumanMessage('What is python?'),
    AIMessage('Python is a high-level, interpreted programming language'),
    HumanMessage('Is it easy to learn?')
]

# response = model.invoke('Hello what is a python?')
response = model.invoke(conversation)

# print(response)
print(response.content)
