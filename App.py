import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

import re

def extract_video_id(url):
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    return match.group(1) if match else None

def get_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        return transcript
    except TranscriptsDisabled:
        return None

def chunk_transcript(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.create_documents([text])

def create_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.from_documents(chunks, embeddings)

def retrieve_context(vector_store, query, k=4):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    return retriever.invoke(query)

def generate_answer(context_docs, question):
    context_text = "\n\n".join(doc.page_content for doc in context_docs)
    
    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        Context:
        {context}

        Question:
        {question}
        """,
        input_variables=["context", "question"]
    )

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.5)
    final_prompt = prompt.invoke({"context": context_text, "question": question})
    return llm.invoke(final_prompt).content


st.set_page_config(page_title="YouTube Video Summarizer", layout="centered")

st.title("🎬 YouTube Video Summarizer")
url = st.text_input("Paste a YouTube video URL")

question = st.text_input("Ask a question about the video", value="Give a summary of the video.")

st.markdown(
    """
    <div style="position: absolute; top: 10px; right: 20px; color: gray; font-size: 20px;">
        <strong>Your Name</strong>
    </div>
    """,
    unsafe_allow_html=True
)

if st.button("Get Answer"):
    if not url:
        st.warning("Please enter a valid YouTube URL.")
    else:
        video_id = extract_video_id(url)
        if not video_id:
            st.error("Could not extract video ID. Check your URL.")
        else:
            with st.spinner("Fetching transcript..."):
                transcript = get_transcript(video_id)
                if not transcript:
                    st.error("Transcript not available for this video.")
                else:
                    chunks = chunk_transcript(transcript)
                    vector_store = create_vector_store(chunks)
                    relevant_docs = retrieve_context(vector_store, question)
                    answer = generate_answer(relevant_docs, question)
                    st.success("Answer:")
                    st.markdown(answer)
