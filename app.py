import os
import docx
import langchain_core
from langchain_openai import OpenAI
import streamlit as st
from docx import Document
from io import BytesIO
from crewai_tools import ScrapeWebsiteTool

import asyncio
import google.generativeai as genai

from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun

# Function to generate text based on topic
def generate_text(llm, topic):
    inputs = {'topic': topic}

    # Initialize DuckDuckGo web search tool
    search_tool = DuckDuckGoSearchRun(
        name="duckduckgo_search",
        description="""Search the web using DuckDuckGo. Action Input should look like this:
                       {"query": "<Whatever you want to search>"}"""
    )
    
    scrape_tool = ScrapeWebsiteTool(
        name="website_scraper",
        description="""Scrape content from web pages. Action Input should look like this:
                       {"website_url": "<URL of the webpage to scrape>"}"""
    )

    # Define Researcher Agent
    researcher_agent = Agent(
        role='Researcher',
        goal='Gather top 5-6 developments on the given topic from the last two weeks.',
        backstory=("An experienced researcher with strong skills in web scraping and fact-finding."),
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    # Define Writer Agent
    writer_agent = Agent(
        role='Writer',
        goal='Write detailed summaries of the developments found by the researcher.',
        backstory='A skilled writer capable of creating engaging and informative content.',
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    # Define Reviewer Agent
    reviewer_agent = Agent(
        role='Reviewer',
        goal='Review the written content for accuracy and coherence.',
        backstory='A meticulous reviewer with a keen eye for detail.',
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    # Define Final Writer Agent
    final_writer_agent = Agent(
        role='Final Writer',
        goal='Compile the reviewed content into a well-structured newsletter.',
        backstory='An experienced editor skilled in creating polished final documents.',
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    # Define Task for Researcher
    task_researcher = Task(
        description='Research top 5-6 developments on the topic from the last two weeks.',
        agent=researcher_agent,
        expected_output='List of 5-6 recent developments.',
        tools=[search_tool, scrape_tool]
    )

    # Define Task for Writer
    task_writer = Task(
        description='Write detailed summaries of the recent developments.',
        agent=writer_agent,
        expected_output='Summarized content for each development.'
    )

    # Define Task for Reviewer
    task_reviewer = Task(
        description='Review the summarized content for accuracy and coherence.',
        agent=reviewer_agent,
        expected_output='Reviewed content.'
    )

    # Define Task for Final Writer
    task_final_writer = Task(
        description='Compile the reviewed content into a newsletter format.',
        agent=final_writer_agent,
        expected_output='Final newsletter document.'
    )

    # Initialize Crew
    crew = Crew(
        agents=[researcher_agent, writer_agent, reviewer_agent, final_writer_agent],
        tasks=[task_researcher, task_writer, task_reviewer, task_final_writer],
        verbose=2,
        context={"Blog Topic is ": topic}
    )

    # Start the workflow and generate the result
    result = crew.kickoff(inputs=inputs)

    return result


# Streamlit web application
def main():
    st.header('AI Newsletter Content Generator')
    mod = None
    with st.sidebar:
        with st.form('Gemini/OpenAI'):
            # User selects the model (Gemini/Cohere) and enters API keys
            model = st.radio('Choose Your LLM', ('Gemini', 'OpenAI'))
            api_key = st.text_input(f'Enter your API key', type="password")
            replicate_api_token = st.text_input('Enter Replicate API key', type="password")
            submitted = st.form_submit_button("Submit")

    # Check if API key is provided and set up the language model accordingly
    if api_key:
        if model == 'OpenAI':
            async def setup_OpenAI():
                loop = asyncio.get_event_loop()
                if loop is None:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                os.environ["OPENAI_API_KEY"] = api_key
                llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0.6)
                print("Configured OpenAI model:", llm)
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
                    model="gemini-pro",
                    verbose=True,
                    temperature=0.6,
                    google_api_key=api_key
                )
                return llm

            llm = asyncio.run(setup_gemini())
            mod = 'Gemini'
        
        # User input for the blog topic
        topic = st.text_input("Enter the blog topic:")

        if st.button("Generate Newsletter Content"):
            with st.spinner("Generating content..."):
                generated_content = generate_text(llm, topic)

                content_lines = generated_content.split('\n')
                first_line = content_lines[0]
                remaining_content = '\n'.join(content_lines[1:])

                st.markdown(first_line)
                st.markdown(remaining_content)

                doc = Document()

                # Option to download content as a Word document
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
