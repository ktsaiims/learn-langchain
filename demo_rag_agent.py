from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool
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
    'Grapes are good',
    'I hate mangos',
    'I hate Windows',
    'Pears are bad',
    'Watermelons are yucky'
]

vector_store = FAISS.from_texts(texts, embedding=embeddings)

# pprint(vector_store.similarity_search('What fruits does user like?', k=3))
# pprint(vector_store.similarity_search('Which OS does user despise?', k=3))

retriever = vector_store.as_retriever(search_kwargs={'k': 3})

retriever_tool = create_retriever_tool(
    retriever=retriever,
    name='knowledgebase_search',
    description='Search the knowledgebase for information'
)

agent = create_agent(
    model='gpt-4.1-mini',
    tools=[retriever_tool],
    system_prompt=('You are assistant. For questions regarding fruits or computers, first call the knowledgebase_search tool to retrieve context. You might need to use it multiple times before answering.')
)

result = agent.invoke(
    {
        'messages': [
            {'role': 'user', 'content': 'What 3 fruits does the person like and dislike?'}
        ]
    }
)

pprint(result)
