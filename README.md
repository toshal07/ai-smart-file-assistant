<<<<<<< HEAD
# AI File Assistant 🤖📁

![Project Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-Educational-orange)

An intelligent, AI-powered system built with Flask, OpenAI, and ChromaDB that allows users to upload documents (PDFs) and ask questions about their content using natural language. This was developed as part of an **Infosys Internship Project**.

---

## 🌟 Key Features

*   **PDF Processing & Text Extraction**: Seamlessly extracts and processes text from multiple uploaded PDF documents.
*   **Semantic Search**: Utilizes advanced vector similarity search (via ChromaDB) to retrieve the most relevant document chunks based on user queries.
*   **AI-Powered Q&A**: Employs OpenAI's GPT models (GPT-3.5-turbo / GPT-4) to generate accurate, context-aware answers.
*   **Source Citations**: Enhances credibility by pointing exactly to the source document segments that the answers are derived from.
*   **Beautiful UI**: A responsive, clean, and dynamic frontend for easy uploading and interactive chatting.

---

## 🛠️ Technology Stack

*   **Frontend**: HTML, CSS (Vanilla), JavaScript, Bootstrap 
*   **Backend**: Python, Flask REST API
*   **Language Models**: OpenAI (`gpt-3.5-turbo` or `gpt-4`)
*   **Embeddings**: OpenAI `text-embedding-ada-002`
*   **Vector Database**: ChromaDB
*   **Document Processing**: PyMuPDF (`fitz`), `pdfplumber`

---

## 📁 Project Structure

```text
ai-file-assistant/
├── .github/                 # GitHub specific scripts/configs
├── docs/                    # Additional documentation and assets
├── scripts/                 # Utility scripts
├── src/                     # Main Application Source Code
│   ├── api_server.py        # Flask REST API backend
│   ├── ingest_pdfs.py       # Script to manually ingest standard PDFs
│   └── modules/             # Core functionality (PDF Processor, OpenAI integration, etc)
├── static/                  # Frontend Web UI (HTML, CSS, JS)
├── tests/                   # Automated tests
├── data/                    # PDF data storage
├── requirements.txt         # Project Dependencies
├── .env.example             # Example environment variables file
├── Dockerfile               # Docker configuration for deployment
└── docker-compose.yml       # Docker Compose setup
```

---

## 🚀 Getting Started

Follow these steps to get the project up and running on your local machine.

### 1️⃣ Prerequisites

Ensure you have the following installed:
*   [Python 3.10+](https://www.python.org/downloads/)
*   [Git](https://git-scm.com/downloads)
*   An [OpenAI API Key](https://platform.openai.com/api-keys)

### 2️⃣ Clone the Repository

```bash
git clone <your-repository-url>
cd ai-file-assistant
```

### 3️⃣ Set Up Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

**For Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**For macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 5️⃣ Environment Validation

You must provide your OpenAI API key for the embedding and completion models to work. 

1. Copy `.env.example` to create an active `.env` file:
   ```bash
   # Windows
   copy .env.example .env
   
   # macOS/Linux
   cp .env.example .env
   ```
2. Open the `.env` file and insert your actual API key:
   ```env
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```
3. Optional: enable LangChain pipeline by adding:
   ```env
   USE_LANGCHAIN_PIPELINE=true
   ```

---

## 💻 Running the Application

To start the backend flask server along with the web interface:

```bash
# Ensure your virtual environment is activated
python src/api_server.py
```

The application will launch locally at `http://127.0.0.1:5050` (or the port specified in terminal). Open this URL in your browser to access the Web UI.

---

## 🧪 Testing the Application

If you have test files (such as NIST cybersecurity PDFs) inside `data/test_data/`, you can test the ingestion scripts and unit tests via:

```bash
pytest tests/
```

To manually ingest specific seed PDFs before starting the server, use:
```bash
python src/ingest_pdfs.py
=======
# 🤖 AI Smart File Assistant

An AI-powered document assistant that allows users to query multiple PDF documents and receive context-aware answers using advanced NLP and Retrieval-Augmented Generation (RAG).

---

## 🚀 Features

* 📄 Upload and process multiple PDF documents
* 💬 AI-powered chat interface
* 🧠 Context-aware responses using LLMs
* 🔍 Semantic search using vector embeddings
* 📊 Key insights extraction
* 📁 Document explorer
* 🔐 Secure authentication system
* ⚙️ Customizable AI settings

---

## 🧠 Tech Stack

**Frontend:**

* HTML, CSS, JavaScript / Streamlit

**Backend:**

* Python (Flask / Streamlit)

**AI/ML:**

* OpenAI API (LLM + Embeddings)

**Database:**

* FAISS / ChromaDB (Vector Database)

---

## 🏗️ Architecture

1. Upload PDFs
2. Extract and clean text
3. Split into chunks
4. Convert into embeddings
5. Store in vector database
6. Perform similarity search
7. Generate AI response

---

## 📸 Screenshots

(Add your UI screenshots here)

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/ai-smart-file-assistant.git
cd ai-smart-file-assistant
pip install -r requirements.txt
streamlit run app.py
>>>>>>> 4545a9109d9430301232e4416ce0c8e10950c214
```

---

<<<<<<< HEAD
## ⚠️ Troubleshooting

*   **`ModuleNotFoundError`**: Ensure you have activated your virtual environment and successfully installed the requirements (`pip install -r requirements.txt`).
*   **OpenAI Authentication Errors**: Double-check that your `OPENAI_API_KEY` in the `.env` file is valid and has sufficient credits.
*   **ChromaDB SQLite Issues**: Make sure your SQLite environment meets ChromaDB's requirements. Upgrading Python or SQLite might be necessary on older setups.

---

## 👥 Contributors

Created and maintained by the FileAI Team as part of the Infosys Spring Internship Pipeline. 
=======
## 🔮 Future Scope

* 🎤 Voice-based queries
* 🌍 Multi-language support
* ⚡ Real-time document updates
* 📈 Advanced analytics

---

## 👨‍💻 Contributors

* Your Name + Team Members

---

## ⭐ License

This project is for academic and learning purposes.
>>>>>>> 4545a9109d9430301232e4416ce0c8e10950c214
