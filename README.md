#  YouTube Video Summarizer

A simple Streamlit web application that summarizes YouTube videos and answers questions using **Google Gemini (Generative AI)**, **LangChain**, **FAISS**, and **YouTube Transcript API**.

---

##  Features

- ✅ Extracts transcript from YouTube videos
- ✅ Splits transcript into manageable chunks
- ✅ Embeds and stores transcript using FAISS
- ✅ Answers user questions using Google Gemini
- ✅ Minimal, clean Streamlit UI

---

##  Demo

![Screenshot 2025-05-24 133715](https://github.com/user-attachments/assets/07210a74-ad25-43ce-b38a-111e74139a23)


---

##  How It Works

1. User inputs a YouTube video URL
2. Transcript is fetched using `youtube-transcript-api`
3. Text is chunked using `LangChain` text splitter
4. Vector embeddings are generated using `GoogleGenerativeAIEmbeddings`
5. Chunks are stored in FAISS for similarity-based retrieval
6. Gemini (via `ChatGoogleGenerativeAI`) answers the user's query based on transcript context

---

## 🧰 Tech Stack

- [Streamlit](https://streamlit.io/)
- [LangChain](https://docs.langchain.com/)
- [Google Gemini via langchain-google-genai](https://pypi.org/project/langchain-google-genai/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [YouTube Transcript API](https://pypi.org/project/youtube-transcript-api/)

---

