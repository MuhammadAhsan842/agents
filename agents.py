
!pip install llama-index-core
!pip install llama-index-readers-file
!pip install llama-index-embeddings-openai
!pip install llama-index-llms-llama-api
!pip install 'crewai[tools]'

import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import LlamaIndexTool
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI


reader = SimpleDirectoryReader(input_files=["finance.csv"])
docs = reader.load_data()

docs[1].get_content()

from google.colab import userdata
os.environ['OPENAI_API_KEY']=userdata.get('OPENAI_API_KEY')

llm = OpenAI(model="gpt-4o")
index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine(similarity_top_k=5, llm=llm)

query_tool = LlamaIndexTool.from_query_engine(
    query_engine,
    name="Finance Query Tool",
    description="Use this tool to lookup the financial data of products and their sales",
)

query_tool.args_schema.schema()



researcher = Agent(
    role="Senior Market Analyst",
    goal="Uncover insights about product sales trends",
    backstory="""You work at a market research firm.
  Your goal is to understand sales patterns across different product categories.""",
    verbose=True,
    allow_delegation=False,
    tools=[query_tool],
)
writer = Agent(
    role="Product Content Specialist",
    goal="Craft compelling content on product trends",
    backstory="""You are a renowned Content Specialist, known for your insightful and engaging articles.
  You transform complex sales data into compelling narratives.""",
    verbose=True,
    allow_delegation=False,
)

# Create tasks for your agents
task1 = Task(
    description="""Analyze the sales data of top 5 products in the last quarter.""",
    expected_output="Detailed sales report with trends and insights",
    agent=researcher,
)

task2 = Task(
    description="""Using the insights provided, develop an engaging blog
  post that highlights the top-selling products and their market trends.
   Your post should be informative yet accessible, catering to a casual audience.
  Make it sound cool, avoid complex words.""",
    expected_output="Full blog post of at least 4 paragraphs",
    agent=writer,
)

# Instantiate your crew with a sequential process
crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    verbose=2,  # You can set it to 1 or 2 to different logging levels
)

result = crew.kickoff()

print("######################")
print(result)

