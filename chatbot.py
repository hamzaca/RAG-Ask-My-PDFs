from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
import configparser

#------------------------------------------------------------------------------

config = configparser.ConfigParser()
config.read("config.ini")
 

# Embeddings + Vector DB -----------------------------------------------------
 
embeddings = OllamaEmbeddings(model=config["MODELS"]["embedder_model"])

db = Chroma(
            collection_name=config["DATABASE"]["collection_name"],
            embedding_function=embeddings,
            persist_directory= config["DATABASE"]["persist_directory"]
            )

# Chat model ------------------------------------------------------------------
 
llm = init_chat_model(model=config["MODELS"]["model_name"], model_provider="ollama", temperature=0)

#  Prompt ---------------------------------------------------------------------
prompt_template = PromptTemplate.from_template("""
You are a helpful assistant. You will be provided with a query and relevant context.
Please provide a concise and informative response based on the context.
Query: {input}

Context: {context}

Chat History: {chat_history}

Please provide a response based on the context above.
If the context doesn't contain relevant information, say "I don't know".

For every piece of information you provide, also provide the source from the context.

Return text as follows:

<Answer to the question>
Source: source_url
""")

#  Retrieval function ---------------------------------------------------------
def retrieve_docs(query: str) -> str:
    """Retrieve relevant documents from the vector store"""
    docs = db.similarity_search(query, k=2)
    return "\n\n".join(
        [f"Source: {d.metadata.get('source','N/A')}\nContent: {d.page_content}" for d in docs]
    )

# Run LLM -------------------------------------------------------------------- 
def run_llm(inputs) -> str:
    formatted_prompt = prompt_template.format(
        input=inputs["input"],
        context=inputs["context"],
        chat_history="\n".join([m.content for m in inputs.get("chat_history", [])])
    )

    response = llm.generate([[HumanMessage(content=formatted_prompt)]])
    return response.generations[0][0].text

#  Fixed Runnable agent ----------------------------------------------------------
agent_runnable = (
    RunnablePassthrough.assign(context=lambda x: retrieve_docs(x["input"]))
    | run_llm
)

#  Streamlit UI -------------------------------------------------------------------

st.set_page_config(page_title="Ask my PDFs Chatbot", page_icon="❓")
st.title("Ask My PDFs")

# Initialize chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# User input
user_question = st.chat_input("Ask me anything...")

if user_question:
    # Add user message
    human_msg = HumanMessage(content=user_question)
    st.session_state.messages.append(human_msg)

    with st.chat_message("user"):
        st.markdown(user_question)

    # Run agent pipeline
    try:
        ai_response = agent_runnable.invoke({
            "input": user_question,
            "chat_history": st.session_state.messages[:-1]  # Exclude current message
        })

        # Add assistant message
        ai_msg = AIMessage(content=ai_response)
        st.session_state.messages.append(ai_msg)

        with st.chat_message("assistant"):
            st.markdown(ai_response)
    except Exception as e:
        st.error(f"Error: {str(e)}")