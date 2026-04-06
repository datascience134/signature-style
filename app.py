import streamlit as st

from core.helper import extract_from_text, keyword_combo_and_search_ui
from core import llm_helper

st.title('✒️ Signature Writing style')

st.markdown('''
    Extract signature keywords from a writer’s style — :rainbow[slang, quirks, niche phrases] — to help you trace their work across the web.
            
    ''')

st.subheader("Input")
llm = llm_helper.LLMInterface()

extract_from_text(llm=llm)
keyword_combo_and_search_ui(llm=llm)
