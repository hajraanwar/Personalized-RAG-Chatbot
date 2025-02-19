import os
import time
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import tiktoken
from langchain_openai import ChatOpenAI

persist_directory = "chroma_storage"

# Specify the filename of your local image
image_filename = 'atomcamp.png'

# Use st.image to display the image
st.image(image_filename, use_container_width=True)

# Initialize Llama model and tokenizer
@st.cache_resource
def load_llama_model():
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_auth_token=True
    )
    return model, tokenizer

# Function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Function to load or create the Chroma database
@st.cache_resource
def get_chroma_db(_embeddings):
    if os.path.exists(persist_directory):
        return Chroma(persist_directory=persist_directory, embedding_function=_embeddings)
    else:
        return Chroma(persist_directory=persist_directory, embedding_function=_embeddings)

# Function to process and store uploaded file
def process_uploaded_file(uploaded_file, embeddings):
    if uploaded_file:
        documents = [uploaded_file.read().decode()]
        text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        texts = text_splitter.create_documents(documents)

        database = get_chroma_db(embeddings)
        database.add_documents(texts)
        database.persist()

        return database
    return None

# Function to count tokens
def count_tokens(text, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Function to summarize context
def summarize_large_context(context, openai_api_key):
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
    summarization_prompt = (
        f"The following context is too large. Summarize it:\n\n{context}\n\n"
        f"Return the most important information in a concise format."
    )
    summary = llm(summarization_prompt)
    return summary.content  # Correctly access the content attribute

# Function to generate response using Llama
def generate_llama_response(model, tokenizer, context, query):
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

# Modified generate_response function to use Llama
def generate_response(query_text, database):
    retriever = database.as_retriever(search_kwargs={"k": 5})
    retrieved_docs = retriever.get_relevant_documents(query_text)
    context = format_docs(retrieved_docs)

    token_count = count_tokens(context, model="gpt-3.5-turbo")
    while token_count > 15000:
        if len(retrieved_docs) <= 1:
            context = context[:15000]  # Truncate context if needed
            break
        else:
            retrieved_docs = retrieved_docs[:-1]
            context = format_docs(retrieved_docs)
            token_count = count_tokens(context, model="gpt-3.5-turbo")

    model, tokenizer = load_llama_model()
    response = generate_llama_response(model, tokenizer, context, query_text)
    return response

# Retry mechanism
def retry_with_backoff(func, max_retries=3, backoff_factor=2):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                time.sleep(backoff_factor ** attempt)
            else:
                raise e

# Initialize session state
if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = ""

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Input for the API key
if not st.session_state["openai_api_key"]:
    st.session_state["openai_api_key"] = st.text_input('OpenAI API Key', type='password')

# File upload
uploaded_file = st.file_uploader('Upload an article', type='txt')

# Input for query
query_text = st.text_input('Enter your question:', placeholder='Ask something...')

if st.button('Submit') and query_text:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=st.session_state["openai_api_key"])

    if uploaded_file:
        with st.spinner('Processing uploaded file...'):
            process_uploaded_file(uploaded_file, embeddings)
            st.success("File processed and added to the database.")

    if os.path.exists(persist_directory):
        database = get_chroma_db(embeddings)
        try:
            with st.spinner('Generating response...'):
                response = retry_with_backoff(
                    lambda: generate_response(query_text, database)
            
                st.session_state["chat_history"].append({"query": query_text, "response": response})
        except Exception as e:
            st.error(f"Error: {e}")

                )

# Display chat history
if st.session_state["chat_history"]:
    st.subheader("Chat History")
    for chat in st.session_state["chat_history"]:
        st.markdown(f"**You:** {chat['query']}")
        st.markdown(f"**Bot:** {chat['response']}")
