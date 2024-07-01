import os
import asyncio
from langchain_groq import ChatGroq
import requests
from typing import Type, Any
from io import BytesIO
from crewai_tools import ScrapeWebsiteTool
import streamlit as st
from docx import Document
from pydantic.v1 import BaseModel, Field
from crewai_tools.tools.base_tool import BaseTool
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew

serp_api_key = ''

# Define SerpApiGoogleSearchToolSchema
class SerpApiGoogleSearchToolSchema(BaseModel):
    q: str = Field(..., description="Query for Google search.")
    tbs: str = Field("qdr:w2", description="Time filter to limit search to the last two weeks.")

# Define SerpApiGoogleSearchTool
class SerpApiGoogleSearchTool(BaseTool):
    name: str = "Google Search"
    description: str = "Search the internet"
    args_schema: Type[BaseModel] = SerpApiGoogleSearchToolSchema
    search_url: str = "https://serpapi.com/search"
    
    def _run(self, q: str, tbs: str = "qdr:w2", **kwargs: Any) -> Any:
        global serp_api_key
        payload = {
            "engine": "google",
            "q": q,
            "tbs": tbs,
            "api_key": serp_api_key,
        }
        headers = {
            'content-type': 'application/json'
        }
    
        response = requests.get(self.search_url, headers=headers, params=payload)
        results = response.json()
    
        summary = ""
        for key in ['answer_box_list', 'answer_box', 'organic_results', 'sports_results', 'knowledge_graph', 'top_stories']:
            if key in results:
                summary += str(results[key])
                break
        
        return summary

# Function to generate text based on topic
def generate_text(llm, topic, serpapi_key):
    inputs = {'topic': topic}
    
    search_tool = SerpApiGoogleSearchTool()
    scrape_tool = ScrapeWebsiteTool()

    researcher_agent = Agent(
        role='Newsletter Content Researcher',
        goal='Search for 5 stories on the given topic, find unique 5 URLs containing the stories, and scrape relevant information from these URLs.',
        backstory="An experienced researcher with strong skills in web scraping, fact-finding, and analyzing recent trends.",
        verbose=True,
        allow_delegation=False,
        max_iter=5,
        llm=llm
    )

    writer_agent = Agent(
        role='Content Writer',
        goal='Write detailed, engaging, and informative summaries of the stories found by the researcher.',
        backstory="An experienced writer with a background in journalism and content creation.",
        verbose=True,
        allow_delegation=False,
        max_iter=5,
        llm=llm
    )

    reviewer_agent = Agent(
        role='Content Reviewer',
        goal='Review and refine content drafts to ensure they meet high standards.',
        backstory="A meticulous reviewer with extensive experience in editing and proofreading.",
        verbose=True,
        allow_delegation=False,
        max_iter=5,
        llm=llm
    )

    final_writer_agent = Agent(
        role='Final Content Writer',
        goal='Compile, refine, and structure all reviewed and approved content into a cohesive newsletter format.',
        backstory="An accomplished writer and editor with extensive experience in journalism and editorial management.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_iter=5
    )

    task_researcher = Task(
        description=f'Research and identify the most interesting 5 stories on the topic of {topic}.',
        agent=researcher_agent,
        expected_output='A list of recent 5 stories with URLs and scraped content.',
        tools=[search_tool, scrape_tool]
    )

    task_writer = Task(
        description='Write detailed summaries of the stories found by the researcher.',
        agent=writer_agent,
        expected_output='Summarized content for all the stories.'
    )

    task_reviewer = Task(
        description='Review the summarized content provided by the writer for accuracy and quality.',
        agent=reviewer_agent,
        expected_output='Reviewed content with suggestions for improvements.'
    )

    task_final_writer = Task(
        description='Compile the reviewed and refined content into a newsletter format.',
        agent=final_writer_agent,
        expected_output='Final newsletter document with all the reviewed summaries.'
    )

    crew = Crew(
        agents=[researcher_agent, writer_agent, reviewer_agent, final_writer_agent],
        tasks=[task_researcher, task_writer, task_reviewer, task_final_writer],
        verbose=2,
        context={"Newsletter Topic is ": topic}
    )

    result = crew.kickoff(inputs=inputs)

    return result

# Streamlit web application
def main():
    st.header('AI Newsletter Content Generator')
    mod = None

    # Initialize session state
    if 'generated_content' not in st.session_state:
        st.session_state.generated_content = None
    if 'topic' not in st.session_state:
        st.session_state.topic = ""

    global serp_api_key

    with st.sidebar:
        with st.form('Gemini/OpenAI/Groq'):
            model = st.radio('Choose Your LLM', ('Gemini', 'OpenAI', 'Groq'))
            api_key = st.text_input(f'Enter your API key', type="password")
            serp_api_key = st.text_input(f'Enter your SerpAPI key', type="password")
            submitted = st.form_submit_button("Submit")

    if api_key and serp_api_key:
        if model == 'OpenAI':
            async def setup_OpenAI():
                loop = asyncio.get_event_loop()
                if loop is None:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                os.environ["OPENAI_API_KEY"] = api_key
                llm = ChatOpenAI(model='gpt-4-turbo', temperature=0.6, max_tokens=2000, api_key=api_key)
                return llm

            llm = asyncio.run(setup_OpenAI())
            mod = 'OpenAI'

        elif model == 'Gemini':
            async def setup_gemini():
                loop = asyncio.get_event_loop()
                if loop is None:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    verbose=True,
                    temperature=0.6,
                    google_api_key=api_key
                )
                return llm

            llm = asyncio.run(setup_gemini())
            mod = 'Gemini'

        elif model == 'Groq':
            async def setup_groq():
                loop = asyncio.get_event_loop()
                if loop is None:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                llm = ChatGroq(
                    api_key=api_key,
                    model='llama3-70b-8192'
                )
                return llm

            llm = asyncio.run(setup_groq())
            mod = 'Groq'

        topic = st.text_input("Enter the newsletter topic:", value=st.session_state.topic)
        st.session_state.topic = topic

        if st.button("Generate Newsletter Content"):
            with st.spinner("Generating content..."):
                st.session_state.generated_content = generate_text(llm, topic, serp_api_key)

        # Display content if it exists in session state
        if st.session_state.generated_content:
            content_lines = st.session_state.generated_content.split('\n')
            first_line = content_lines[0]
            remaining_content = '\n'.join(content_lines[1:])

            st.markdown(first_line)
            st.markdown(remaining_content)

            # Create and download the document
            doc = Document()
            doc.add_heading(topic, 0)
            doc.add_paragraph(first_line)
            doc.add_paragraph(remaining_content)

            buffer = BytesIO()
            doc.save(buffer)
            buffer.seek(0)

            st.download_button(
                label="Download as Word Document",
                data=buffer,
                file_name=f"{topic}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

if __name__ == "__main__":
    main()
