import requests
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# Classes
@dataclass
class Context:
    user_id: str

@dataclass
class ReponseFormat:
    summary: str
    temperature_celsius: float
    temperature_fahrenheit: float
    humidity: float

# Tools
@tool('get_weather', description='Return weather information for a given city', return_direct=False)
def get_weather(city: str):
    response = requests.get(f'https://wttr.in/{city}?format=j1')
    return response.json()

@tool('locate_user', description="Find user's city based on the context")
def locate_user(runtime: ToolRuntime[Context]):
    match runtime.context.user_id:
        case 'ABC123':
            return 'Vienna'
        case 'XYZ456':
            return 'London'
        case 'HJK111':
            return 'Paris'
        case _:
            return 'Unknown'    

# Model
model = init_chat_model('gpt-4.1-mini', temperature=0.3)

checkpointer = InMemorySaver()

agent = create_agent(
    model=model, # requires OPENAI_API_KEY, langchain[openai]
    tools=[get_weather, locate_user],
    system_prompt='You are a hilarious weather assistant',
    context_schema=Context,
    response_format=ReponseFormat,
    checkpointer=checkpointer
)

configs = {'configurable': {'thread_id': 1}}

response = agent.invoke(
    {
        'messages': [
            {'role': 'user', 'content': 'What is the weather like?'}
        ]
    },
    config=configs,
    context=Context(user_id='ABC123')
)

print(response['structured_response'].summary)
print(response['structured_response'].temperature_celsius)
print(response['structured_response'].humidity)
