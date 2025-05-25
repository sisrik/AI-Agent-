from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama
from tools import search_tool

from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

llm = ChatOllama(model="mistral")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """

            You are a research assistant that will help generate a research paper.
            Answer the user query and use necessary tools.
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ] 
).partial(format_instructions=parser.get_format_instructions())

tools= [search_tool]
agent= create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor=AgentExecutor(agent=agent, tools=[], verbose=True)
query= input("What can i help you search?")
raw_response= agent_executor.invoke({"query": query})
print(raw_response)



