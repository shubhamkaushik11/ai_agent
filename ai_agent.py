from dotenv import dotenv_values,load_dotenv
import os
#config = dotenv_values(".env")
load_dotenv()
GROQ_API_KEY=os.environ['GROQ_API_KEY']
TAVILY_API_KEY=os.environ['TAVILY_API_KEY']
import os
# os.environ['GROQ_API_KEY'] = GROQ_API_KEY
# os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
groq_llm=ChatGroq(model="llama-3.3-70b-versatile")

search_tool=TavilySearchResults(max_results=2)

#Step3: Setup AI Agent with Search tool functionality
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage


def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    if provider=="Groq":
        llm=ChatGroq(model=llm_id)
    elif provider=="OpenAI":
        ai_messages ="OpenAI is Not Configured Yet"
        return ai_messages

    tools=[TavilySearchResults(max_results=2)] if allow_search else []
    agent=create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=system_prompt
    )
    state={"messages": query}
    response=agent.invoke(state)
    messages=response.get("messages")
    ai_messages=[message.content for message in messages if isinstance(message, AIMessage)]
    return ai_messages[-1]





