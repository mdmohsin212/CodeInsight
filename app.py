import streamlit as st
from codeInsight.pipeline.prediction_pipeline import PredictionPipeline
from codeInsight.logger import logging

st.set_page_config(
    page_title="CodeInsight Assistant",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def load_pipeline():
    try:
        pipeline = PredictionPipeline()
        return pipeline
    except Exception as e:
        logging.error("Failed to load pipeline in Streamlit app")
        st.error(f"Failed to load model pipeline: {e}")
        return None

pipeline = load_pipeline()

st.title("🤖 CodeInsight Assistant")
st.caption("Your fine-tuned CodeLlama-7b model, ready to help with Python.")

if pipeline:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! How can I help you with Python programming today?"}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask me to write python code")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = pipeline.predict(prompt)
                formatted_response = f"```python\n{response}\n```"
                st.markdown(formatted_response)

        st.session_state.messages.append({"role": "assistant", "content": formatted_response})

else:
    st.error("The prediction pipeline could not be loaded. Please check the logs.")