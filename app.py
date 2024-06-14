def main():
    st.header('AI Newsletter Content Generator')
    mod = None
    global serp_api_key
    with st.sidebar:
        with st.form('Gemini/OpenAI'):
            model = st.radio('Choose Your LLM', ('Gemini', 'OpenAI'))
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
                llm = ChatOpenAI(temperature=0.6, max_tokens=3500)
                print("Configured OpenAI model")
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
                print("Gemini Configured")
                return llm

            llm = asyncio.run(setup_gemini())
            mod = 'Gemini'
        
        topic = st.text_input("Enter the blog topic:")

        if st.button("Generate Newsletter Content"):
            with st.spinner("Generating content..."):
                generated_content, images = generate_text(llm, topic)

                content_lines = generated_content.split('\n')
                first_line = content_lines[0]
                remaining_content = '\n'.join(content_lines[1:])

                st.markdown(first_line)
                st.markdown(remaining_content)

                # Display images
                for img in images:
                    st.image(img['link'], caption=img['title'], use_column_width=True)

                # Create Word document
                doc = Document()
                doc.add_heading(topic, 0)
                doc.add_paragraph(first_line)
                doc.add_paragraph(remaining_content)
                
                # Add images to Word document
                for img in images:
                    add_image_to_doc(doc, img['link'])
                
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
