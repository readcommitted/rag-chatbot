# app.py
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import openai
import os
import torch
from dotenv import load_dotenv
import streamlit as st

st.set_page_config(page_title="RAG Chatbot", layout="wide")

torch.classes.__path__ = []

os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

# Load from environment variables
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = os.environ["PINECONE_INDEX"]
# all-MiniLM-L6-v2 produces 384-dimensional embeddings
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
DEFAULT_TOP_K = int(os.environ.get("TOP_K", 5))
DEFAULT_CHUNKS_FOR_GPT = int(os.environ.get("CHUNKS_FOR_GPT", 2))
DEFAULT_CONFIDENCE_SCORE = float(os.environ.get("MIN_SCORE", .40))

# Initialize session_state
if "confidence_score" not in st.session_state:
    st.session_state["confidence_score"] = DEFAULT_CONFIDENCE_SCORE
if "top_k" not in st.session_state:
    st.session_state["top_k"] = DEFAULT_TOP_K
if "chunks_for_gpt" not in st.session_state:
    st.session_state["chunks_for_gpt"] = DEFAULT_CHUNKS_FOR_GPT

# --- Init OpenAI client ---
openai.api_key = OPENAI_API_KEY

# --- Load Embedding Model ---
#    Loads the sentence embedding model used for semantic search.
#
#    This function uses Streamlit's `@st.cache_resource` decorator to cache the model
#    instance across reruns, which avoids reloading it from disk or the internet.
#
#    Currently configured to load the 'all-MiniLM-L6-v2' model, which:
#    - Produces 384-dimensional sentence embeddings
#    - Is optimized for semantic similarity tasks
#    - Provides a good balance between performance and speed
#    - Is commonly used for vector search and RAG pipelines
#
#    Returns:
#        SentenceTransformer: A loaded SentenceTransformer model.
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

# --- Init Pinecone ---
#     Initializes and returns a connection to a Pinecone vector index.
#
#     This function uses Streamlit's `@st.cache_resource` to cache the Pinecone client
#     and index connection, ensuring it only initializes once per session.
#
#     Pinecone is used to:
#     - Store and retrieve high-dimensional embeddings
#     - Perform vector similarity search using the configured metric (e.g., cosine)
#
#     Environment variables required:
#         - PINECONE_API_KEY: Your Pinecone API key
#         - INDEX_NAME: The name of the Pinecone index to connect to
#
#     Returns:
#         pinecone.Index: A live connection to the specified Pinecone index.
@st.cache_resource
def init_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(INDEX_NAME)

# --- Vector Search Function ---
#     Performs a vector similarity search in Pinecone for the given query.
#
#     The query is first embedded using the currently loaded embedding model, then
#     passed to Pinecone to retrieve the top_k most similar vectors. The results are
#     filtered by a minimum similarity score threshold.
#
#     Args:
#         query (str): The user's natural language question or input.
#         top_k (int): The number of top similar vectors to retrieve from Pinecone.
#         min_score (float, optional): The minimum similarity score to include in the result.
#             If None, the current value from `st.session_state["confidence_score"]` is used.
#
#     Returns:
#         List[dict]: A list of Pinecone match objects (with metadata), filtered by score.
def search_pinecone(query, top_k, min_score=None):
    if min_score is None:
        min_score = st.session_state.get("confidence_score")
    query_emb = embedding_model.encode([query]).tolist()
    result = pinecone_index.query(
        vector=query_emb[0],
        top_k=top_k,
        include_metadata=True
    )
    matches = result["matches"]
    filtered_matches = [m for m in matches if m["score"] >= min_score]
    return filtered_matches

# --- GPT-4 Completion Function ---
#     Generates a grounded natural language response using an OpenAI LLM (e.g., GPT-4),
#     based on a user question and retrieved context.
#
#     The function constructs a prompt that includes:
#     - A system-style instruction to act as a helpful assistant
#     - The retrieved context (e.g., top N Pinecone chunks)
#     - The user's question
#
#     It then sends the prompt to OpenAI's Chat API using the specified model.
#
#     Args:
#         question (str): The user's natural language question.
#         context (str): A concatenated string of retrieved context chunks.
#
#     Returns:
#         Tuple[str, str]:
#             - The LLM-generated response as a string
#             - The full prompt that was sent (for optional display or logging)
def run_openai_llm(question, context):
    prompt = f"""You are a helpful assistant. Use the context below to answer the question accurately and concisely.

    Context:
    {context}
    
    Question: {question}
    Answer:"""

    response = openai.chat.completions.create(
        model=OPENAI_MODEL, # or gpt-4
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip(), prompt

# --- Load Resources ---
embedding_model = load_embedding_model()
pinecone_index = init_pinecone()

# --- UI ---
with st.sidebar:
    st.markdown("## ⚙️ Controls")

    st.slider(
        "Minimum similarity threshold",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state["confidence_score"],
        step=0.01,
        key="confidence_score",
        help="Filter out low-confidence matches. Lower = more matches, higher = stricter relevance."
    )

    st.slider(
        "Retrieve Top K Chunks",
        min_value=1,
        max_value=20,
        value=st.session_state["top_k"],
        step=1,
        key="top_k",
        help="How many chunks to retrieve from Pinecone (based on similarity)."
    )

    st.slider(
        "Chunks to Send to GPT",
        min_value=1,
        max_value=st.session_state["top_k"],
        value=st.session_state["chunks_for_gpt"],
        step=1,
        key="chunks_for_gpt",
        help="How many of the top K chunks should be passed to the LLM."
    )

    show_raw_scores = st.checkbox("🔢 Show raw Pinecone scores", value=True)
    show_chunk_sources = st.checkbox("📄 Display source info", value=True)
    dev_mode = st.checkbox("🧪 Developer Mode", value=False)
    show_architecture_diagram = st.checkbox("📐 Architecture Diagram", value=False)

    st.markdown("---")
    st.markdown(
        "Use these controls to tune your search precision and how much detail is shown in the results. "
        "A score around 0.40–0.60 is often a good starting point."
    )

if show_architecture_diagram:
    st.title("📐 Architecture")
    st.image("assets/RAG Pipeline.jpg",  use_container_width=True, caption="RAG Chatbot Architecture")

st.title("🔍 RAG Chatbot with Pinecone + OpenAI GPT")

query = st.text_input("Ask a question:")
mode = st.radio("Choose mode:", ["Pinecone Search Only", "Pinecone Search + GPT"])
submit = st.button("Submit")

matches = []  # Default so nothing renders until submit is clicked


# This section handles query submission. It retrieves relevant context chunks from Pinecone based on a user-defined
# similarity threshold, then either displays the top result or sends the top N chunks to GPT for a grounded answer.
# Retrieved chunks and similarity scores are displayed for transparency and debugging.
if submit and query:
    with st.spinner("Retrieving context from Pinecone..."):
        TOP_K = st.session_state["top_k"]
        matches = search_pinecone(query, TOP_K)

    if not matches:
        st.warning("❌ No high-confidence context found. Try rephrasing your question.")
    else:
        CHUNKS_FOR_GPT = st.session_state["chunks_for_gpt"]
        used_chunks = [m["metadata"]["text"] for m in matches[:CHUNKS_FOR_GPT]]
        combined_context = "\n\n".join(used_chunks)

        if mode == "Pinecone Search + GPT":
            with st.spinner("Generating response from GPT..."):
                answer, prompt_text = run_openai_llm(query, combined_context)
            st.markdown("### 💬 GPT Answer")
            st.write(answer)
            if dev_mode:
                st.markdown("### Prompt Sent to LLM")
                st.code(prompt_text, language="markdown")
        elif mode == "Pinecone Search Only":
            st.markdown("### 📄 Top Retrieved Answer (No LLM)")
            st.markdown(
                f'<div style="white-space: pre-wrap; background-color: #f6f8fa; padding: 0.5em; border-radius: 6px;">{used_chunks[0]}</div>',
                unsafe_allow_html=True
            )
        st.markdown(f"### 📚 Retrieved Contexts from Pinecone (Score ≥ {st.session_state['confidence_score']:.2f})")
        for i, match in enumerate(matches):
            score = match["score"]
            chunk = match["metadata"]["text"]
            source = match["metadata"].get("source", "Unknown")

            # Build the chunk header line
            title = f"**Rank {i + 1}**"
            if show_raw_scores:
                title += f" — Score: {score:.3f}"
            if show_chunk_sources:
                title += f" — Source: {source}"

            st.markdown(title)
            st.markdown(
                f'<div style="white-space: pre-wrap; background-color: #f6f8fa; padding: 0.5em; border-radius: 6px;">{chunk}</div>',
                unsafe_allow_html=True
            )