import streamlit as st

from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import ServiceContext, set_global_service_context


from transformers import BitsAndBytesConfig

import torch

import os




model="NousResearch/Llama-2-7b-chat-hf"


st.set_page_config(page_title="Mufeed: SJAHI Assistant", page_icon="👨‍🔧👨‍🏭")
st.title("👨‍🔧 Mufeed: Your education assistant")
torch.cuda.empty_cache()



@st.cache_resource(ttl="1h")
def configure_embeddings():
    embeddings = HuggingFaceEmbedding(model_name='BAAI/bge-small-en-v1.5',cache_folder="utils/models")

    return embeddings

@st.cache_resource(ttl="1h")
def create_text_generation_pipeline(_model):
    compute_dtype = getattr(torch, "float16")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )
    return HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=500,
    generate_kwargs={"temperature": 0.1, "do_sample": True},
    tokenizer_name=_model,
    model_name=_model,
    
    device_map="auto",
    tokenizer_kwargs={"max_length": 2048},
    model_kwargs={"torch_dtype": torch.float16,"cache_dir":"utils/models","quantization_config":quant_config,"do_sample":True,"top_k":10})


@st.cache_resource(ttl="1h")
def create_index():
    llm = create_text_generation_pipeline(model)
    Settings.chunk_size = 500
    Settings.chunk_overlap = 100
    Settings.llm = llm
    embed_model=configure_embeddings()
    ctx = ServiceContext.from_defaults(llm=llm,embed_model=embed_model)
    documents = SimpleDirectoryReader("./docs").load_data()
    parser = SentenceSplitter(chunk_size=500, chunk_overlap=100)
    nodes = parser.get_nodes_from_documents(documents)
    index = VectorStoreIndex(
        nodes,service_context=ctx
    )
    index.storage_context.persist("./data") 
    return index



@st.cache_resource(ttl="1h")
def load_Index():
    llm = create_text_generation_pipeline(model)
    Settings.chunk_size = 500
    Settings.chunk_overlap = 100
    Settings.llm = llm
    embed_model=configure_embeddings()
    ctx = ServiceContext.from_defaults(llm=llm,embed_model=embed_model)
    set_global_service_context(ctx)
    storage_context = StorageContext.from_defaults(persist_dir="./data")
    index = load_index_from_storage(storage_context=storage_context,service_context=ctx)
    return index
    

with st.sidebar:

    files = st.file_uploader(
        label="Upload PDF files", type=["pdf"], accept_multiple_files=True
    )


    if len(files) == 0:
        st.error("No file were uploaded")

    for file in files:
        bytes_data = file.read()  # read the content of the file in binary
        if file.name in os.listdir("data/"):
            continue
        else:
            
            print(file.name,os.listdir("data/"))
            with open(os.path.join("docs", file.name), "wb") as f:
                f.write(bytes_data)  # write this content elsewhere
    if files:

        index = create_index()

    else:

        index = load_Index()




    


if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help!"}
    ]

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine( verbose=True,
    similarity_top_k=5,
    similarity_cutoff=0.7,
    chat_mode="condense_plus_context",
    system_prompt=("""<s>[INST] <<SYS>>
You are Mufeed who is an AI project of Saudi Japanese Automobile High Institute (SJAHI).
 Use the context to answer user's question precisly and keep it short no introductions or notes. If you don't know the answer don't make up an answer or rephrase the question.
                   
{context}
               
<</SYS>>
{question}
[/INST]</s>""")

)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])
        

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
        with st.spinner("Answering..."):
           with st.chat_message ("assistant"):
            response = st.session_state.chat_engine.stream_chat(prompt)
            st.write_stream(response.response_gen)
            new_message = {"role": "assistant", "content": response.response }
            st.session_state.messages.append(new_message) # Add response to message history
            if prompt.lower() in 'thanks' or prompt.lower() in 'hello':
                with st.sidebar:
                    st.write("Source: ")
            else:
                with st.sidebar:
                    st.write("Source: " + response.source_nodes[0].metadata['file_name'])