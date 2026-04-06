import streamlit as st

from core import file_handler 
from core.helper import extract_from_text, extract_content_from_screenshots, get_screenshot_from_urls
from core import constants
# from core import output_processor
from core import llm_helper

st.title('✒️ Signature Writing style')

st.markdown('''
    Extract signature keywords from a writer’s style — :rainbow[slang, quirks, niche phrases] — to help you trace their work across the web.
            
    ''')

st.subheader("Input")
llm = llm_helper.LLMInterface()

text_option = "Paste Text"
screenshot_option = "Upload Screenshot(s)*"
url_option = "Enter URLs*"

input_method = st.radio("Select input method:", [text_option, screenshot_option, url_option])
st.markdown(
    "<span style='font-size:0.9em; color:#888888;'><i>*Designed mainly for short forum posts. Long screenshots are split into chunks and may lose context (e.g., author names that appear only at the start) across sections.</i></span>",
    unsafe_allow_html=True
)


       
if input_method == text_option:
    extract_from_text(llm=llm)

elif input_method == screenshot_option:
    content_df = extract_content_from_screenshots(llm=llm)
    if content_df is not None:
        text_input = ' '.join(str(c) for c in content_df['content'] if c)
        extract_from_text(llm=llm, text_input=text_input)
elif input_method == url_option:
    get_screenshot_from_urls()
    screenshot_result = st.session_state.get("screenshot_result", None)

    if screenshot_result and "image_paths" in screenshot_result:
        screenshots = screenshot_result["image_paths"]

        content_df = extract_content_from_screenshots(llm=llm, screenshot_files=screenshots)
        if content_df is not None:
            text_input = ' '.join(str(c) for c in content_df['content'] if c)
            extract_from_text(llm=llm, text_input=text_input)


