import os
import replicate
import streamlit as st
from langchain.llms import OpenLLM
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

# App title
st.set_page_config(page_title="🦙💬 Llama 2 Chatbot")

# Replicate Credentials
with st.sidebar:
    st.title('💬 Chatbot for text document QA')
    if 'REPLICATE_API_TOKEN' in st.secrets:
        st.success('API key already provided!', icon='✅')
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')

os.environ['REPLICATE_API_TOKEN'] = replicate_api

uploaded_file = st.file_uploader("Add a text file")
if uploaded_file is not None:
    

    with open('_sample.txt', 'w') as file:
        file.write("".join([line.decode() for line in uploaded_file]))

    loader = TextLoader('_sample.txt')
    documents = loader.load()

    text_splitter=CharacterTextSplitter(separator='\n',
                                        chunk_size=1000,
                                        chunk_overlap=50)

    text_chunks=text_splitter.split_documents(documents)

    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore=FAISS.from_documents(text_chunks, embeddings)

    server_url = "https://683c-38-147-83-24.ngrok-free.app"
    llm = OpenLLM(server_url=server_url)

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

    qa.run("What is the capital of Canada?")

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating LLaMA2 response
# Refactored from https://github.com/a16z-infra/llama2-chatbot
def generate_response(prompt_input):
    string_dialogue = """
    You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'.
    Answer all the questions in short and clearly. Below you'll find the conversation between you and the user.
    """
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"

    query = f"{string_dialogue} {prompt_input} Assistant: "
    try:
        output = qa.run(query)
    except NameError:
        output = "You must load the text document first"
    return output
    

# User-provided prompt
if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt)
            placeholder = st.empty()
            placeholder.markdown(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)