import streamlit as st
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

from custom_llm import CustomLLM

st.set_page_config(page_title="💬 Chatbot")
st.title("💬 Chatbot for text document QA")

with st.sidebar:
    st.title('Guidelines 📜')
    st.markdown("""1. You can ask questions regarding your loaded text file;
                 2. Just add your `.txt` file
                 3. **Note**: the user may face 1-2 min delay during the first question. It may take some time to load the model on the server.""")

uploaded_file = st.file_uploader("Add a text file in .txt format")
if uploaded_file is not None:
    with open("_sample.txt", "w") as file:
        file.write("".join([line.decode() for line in uploaded_file]))

    loader = TextLoader("_sample.txt")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=50
    )

    text_chunks = text_splitter.split_documents(documents)

    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(text_chunks, embeddings)

    llm = CustomLLM(n=10)

    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
    )


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you?"}
    ]


st.button("Clear Chat History", on_click=clear_chat_history)



if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you?"}
    ]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def generate_response(prompt_input):
    output = qa.run(prompt_input)
    return output


if prompt := st.chat_input(disabled=not uploaded_file is not None):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)


if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt)
            placeholder = st.empty()
            placeholder.markdown(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
