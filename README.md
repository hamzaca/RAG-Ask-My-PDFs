This project is a RAG-based (Retrieval-Augmented Generation) system extracting text from PDFs to answer questions via Ollama.
First  : text is chunked into manageable sections, with embeddings stored in Chroma for efficient retrieval.
Second : using similarity search, like cosine similarity, RAG finds the chunks most relevant to the question ask by the user.(Retriever)
Last   : the chunks of text retrieved are fed into a language model that takes them as context to generate an answer.(generator)

To read pdfs and store them in chroma vector store : 
'python prepare_vector_store.py'

To launch the streamlit UI launch  : 
'streamlit run chatbot.py'
