import os
from dotenv import load_dotenv
from flask import Flask, request, render_template, session, redirect, url_for

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")

# Updated category-to-filename mapping with new categories
CATEGORY_FILES = {
    "general": "company_faq.txt",
    "academics": "academics.txt",
    "finance": "finance.txt",
    "operations": "operations.txt",
    "hr": "hr.txt",
    "it": "it.txt",
    "marketing": "marketing.txt",
    "admissions": "admissions.txt"
}

def build_qa_chain(filename):
    """Build a LangChain QA chain from a given text file."""
    if not os.path.exists(filename):
        return None  # File missing
    loader = TextLoader(filename, encoding="utf-8")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0, openai_api_key=openai_api_key),
        retriever=vectorstore.as_retriever()
    )
    return qa_chain

@app.route("/", methods=["GET"])
def index():
    session.clear()
    session["category"] = "general"  # Default category
    return redirect(url_for("chat"))

@app.route("/set_category/<category>", methods=["POST"])
def set_category(category):
    if category not in CATEGORY_FILES:
        return "Invalid category", 400
    session["category"] = category
    session["chat_history"] = []  # Clear chat when category changes
    return redirect(url_for("chat"))

@app.route("/chat", methods=["GET", "POST"])
def chat():
    category = session.get("category", "general")
    filename = CATEGORY_FILES.get(category)

    qa = build_qa_chain(filename)
    if qa is None:
        return f"Error: File for '{category}' not found.", 500

    if "chat_history" not in session:
        session["chat_history"] = []

    if request.method == "POST":
        query = request.form.get("query")
        if query:
            chat_history = session["chat_history"]
            chat_history.append({"user": query, "bot": None})
            session["chat_history"] = chat_history

            result = qa.invoke(query)
            answer = result.get('result') if isinstance(result, dict) else str(result)

            chat_history[-1]["bot"] = answer
            session["chat_history"] = chat_history

    return render_template("chat.html", chat_history=session.get("chat_history", []), category=category)

@app.route("/clear_chat", methods=["POST"])
def clear_chat():
    session["chat_history"] = []
    return redirect(url_for("chat"))

if __name__ == "__main__":
    app.run(debug=True)
