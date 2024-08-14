import os
import requests
from crewai import Agent, Task, Crew, Process
from crewai_tools import tool
from datetime import datetime, timedelta
from IPython.display import Markdown

# Set up API keys
os.environ["OPENAI_API_KEY"] = "YOUR-API-KEY"

# ArxivTool function to fetch papers from arXiv
@tool
def arxiv_tool(query="machine learning", max_results=5, days_back=7):
    """
    Tool to search for recent machine learning and AI research papers on arXiv and return their details, including PDF links.
    """
    base_url = "http://export.arxiv.org/api/query"
    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    query_params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }
    response = requests.get(base_url, params=query_params)
    response.raise_for_status()

    entries = response.text.split('<entry>')
    papers = []
    for entry in entries[1:]:
        title = entry.split('<title>')[1].split('</title>')[0].strip()
        pdf_url = entry.split('<link title="pdf" href="')[1].split('"')[0].strip()
        published_date = entry.split('<published>')[1].split('</published>')[0][:10]
        if published_date >= start_date:
            papers.append({'title': title, 'pdf_url': pdf_url, 'published_date': published_date})

    if len(papers) < 5:
        raise ValueError("Not enough recent papers found. Adjust the query or the time range.")

    return papers[:max_results]

# 1. Researcher Agent
researcher = Agent(
    role='Internet Researcher',
    goal='Find recent machine learning and AI research papers published within the last week.',
    verbose=True,
    memory=True,
    backstory=(
        "You are a diligent researcher who scours the internet for the latest advancements in machine learning and AI."
    ),
    tools=[arxiv_tool],  # Use the ArxivTool to fetch papers
)

# 2. Reader Agent
reader = Agent(
    role='Paper Analyst',
    goal='Thoroughly read each research paper and extract key insights.',
    verbose=True,
    memory=True,
    backstory=(
        "With a keen eye for detail, you excel at breaking down complex research papers and identifying the most important information."
    ),
    tools=[],  # No external tools, relying on agent's capabilities
)

# 3. Writer Agent
writer = Agent(
    role='Content Writer',
    goal='Craft engaging summaries of research papers for LinkedIn.',
    verbose=True,
    memory=True,
    backstory=(
        "A master of words, you create compelling narratives that make research accessible and interesting."
    ),
    tools=[],  # No external tools, relying on agent's capabilities
)

# Task 1: Research Task - Fetch papers from arXiv
research_task = Task(
    description=(
        "Search arXiv for new research papers on machine learning and AI released in the past week. "
        "Download the PDFs of these papers for further analysis."
    ),
    expected_output='A list of downloaded PDFs of the most recent research papers.',
    tools=[arxiv_tool],
    agent=researcher,
)

# Task 2: Reading Task - Summarize the papers
reading_task = Task(
    description=(
        "Read each paper thoroughly and extract key points, methodologies, and findings. "
        "Generate a long, detailed summary for each paper."
    ),
    expected_output='Detailed summaries of each paper.',
    tools=[],  # No external tools, relying on agent's capabilities
    agent=reader,
)

# Task 3: Writing Task - Create LinkedIn summaries
writing_task = Task(
    description=(
        "Write a long, engaging summary of each paper suitable for LinkedIn, highlighting its importance and impact in the field."
    ),
    expected_output='A LinkedIn post draft for each research paper.',
    tools=[],  # No external tools, relying on agent's capabilities
    agent=writer,
    async_execution=False,  
)

# Forming the crew
crew = Crew(
    agents=[researcher, reader, writer],
    tasks=[research_task, reading_task, writing_task],
    process=Process.sequential
)

# Kicking off the crew
result = crew.kickoff(inputs={})
print("Markdown content has been printed to the screen.")
