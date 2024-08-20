import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from crewai import Agent, Task, Crew, Process
from crewai_tools.tools.base_tool import BaseTool

os.environ["OPENAI_API_KEY"] = "YOUR-API-KEY"

# Define the ArxivTool class with required fields
class ArxivTool(BaseTool):
    name: str = "ArxivTool"
    description: str = "A tool to search for recent machine learning and AI research papers on arXiv."

    def _run(self, query: str = "machine learning", max_results: int = 5, days_back: int = 7):
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

        return papers[:max_results]

# Define a WebScrapingTool class
class WebScrapingTool(BaseTool):
    name: str = "WebScrapingTool"
    description: str = "A tool to scrape recent machine learning and AI research papers from web pages."

    def _run(self, query: str = "machine learning research papers", max_results: int = 5, days_back: int = 7):
        search_url = f"https://www.google.com/search?q={query}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

        response = requests.get(search_url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        search_results = soup.find_all('div', class_='BNeawe vvjwJb AP7Wnd')

        papers = []
        for i, result in enumerate(search_results[:max_results]):
            title = result.get_text()
            link = result.find_parent('a')['href']
            published_date = datetime.now().strftime('%Y-%m-%d')  # Using today's date for simplicity
            papers.append({
                'title': title,
                'url': link,
                'published_date': published_date
            })

        return papers

# Define a function to execute the writing task for a specific topic
def execute_writing_task(task_output):
    """
    Custom function to format the summaries in Markdown and print them to the screen.
    """
    markdown_content = "# Research Paper Summaries\n\n"
    for summary in task_output:
        markdown_content += f"## {summary['title']}\n\n"
        markdown_content += f"**Published Date:** {summary.get('published_date', 'N/A')}\n\n"
        markdown_content += f"**Link:** [{summary.get('url', 'Link to paper')}]({summary.get('url', '')})\n\n"
        markdown_content += f"{summary.get('summary', '')}\n\n---\n\n"
    
    # Print the markdown content
    print(markdown_content)

    return markdown_content

# Forming the crew and defining tasks
def create_crew_for_topic(topic):
    # 1. Researcher Agent
    researcher = Agent(
        role='Internet Researcher',
        goal=f'Find recent research papers published within the last week on {topic}.',
        verbose=True,
        memory=True,
        backstory=(
            "You are a diligent researcher who scours the internet for the latest advancements in AI, "
            "including machine learning, large language models, and other related fields."
        ),
        tools=[ArxivTool(), WebScrapingTool()],
    )

    # 2. Reader Agent
    reader = Agent(
        role='Paper Analyst',
        goal=f'Thoroughly read each research paper on {topic} and extract key insights.',
        verbose=True,
        memory=True,
        backstory=(
            "With a keen eye for detail, you excel at breaking down complex research papers and identifying "
            "the most important information."
        ),
        tools=[],
    )

    # 3. Writer Agent
    writer = Agent(
        role='Content Writer',
        goal=f'Craft engaging summaries of research papers on {topic} in Markdown format.',
        verbose=True,
        memory=True,
        backstory=(
            "A master of words, you create compelling narratives that make research accessible and interesting."
        ),
        tools=[],
    )

    # Task 1: Research Task
    research_task = Task(
        description=(
            f"Search arXiv and the web (Medium, GitHub, etc.) for new research papers on {topic} "
            "released in the past week. Download the PDFs or URLs of these papers for further analysis."
        ),
        expected_output='A list of downloaded PDFs or URLs of the most recent research papers.',
        tools=[ArxivTool(), WebScrapingTool()],
        agent=researcher,
    )

    # Task 2: Reading Task
    reading_task = Task(
        description=(
            f"Read each paper on {topic} thoroughly and extract key points, methodologies, and findings. "
            "Generate a long, detailed summary for each paper."
        ),
        expected_output='Detailed summaries of each paper.',
        tools=[],
        agent=reader,
    )

    # Task 3: Writing Task
    writing_task = Task(
        description=(
            f"Write a long, engaging summary of each paper on {topic} suitable for LinkedIn, highlighting its "
            "importance and impact in the field."
        ),
        expected_output='A Markdown file containing summaries for each research paper.',
        tools=[],
        agent=writer,
        async_execution=False,
        execute=execute_writing_task,
    )

    # Forming the crew
    crew = Crew(
        agents=[researcher, reader, writer],
        tasks=[research_task, reading_task, writing_task],
        process=Process.sequential
    )
    return crew

# Handling multiple topics
topics = ["machine learning", "large language model", "small language model", "RAG", "AI ethics"]

for topic in topics:
    print(f"Processing topic: {topic}")
    crew = create_crew_for_topic(topic)
    crew.kickoff(inputs={})

print("Markdown content for all topics has been printed to the screen.")
