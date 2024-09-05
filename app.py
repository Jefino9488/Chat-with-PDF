import os
import chromadb
import warnings
from chromadb import Settings
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from werkzeug.utils import secure_filename
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from flask import Flask, request, jsonify, render_template
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

warnings.filterwarnings("ignore")

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CHROMA_DB_DIR = 'db/chroma'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_pdf(pdf_path):
    pdf_loader = PyPDFLoader(pdf_path)
    pages = pdf_loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    context = "\n\n".join(page.page_content for page in pages)
    chunks = text_splitter.split_text(context)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv('GOOGLE_API_KEY')
    )

    client = chromadb.PersistentClient(path="./path/to/chroma", settings=Settings(allow_reset=True))
    client.reset()

    vector = Chroma.from_texts(
        chunks, embeddings, client=client
    ).as_retriever(search_kwargs={"k": 5})

    return vector


vector_store = None
qa_chain = None


def initialize_qa_chain():
    global vector_store
    if vector_store is None:
        raise ValueError("Vector store is not initialized.")

    if model is None:
        raise ValueError("Model is not initialized.")

    return RetrievalQA.from_chain_type(
        llm=model,
        retriever=vector_store,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )


model = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=os.getenv('GOOGLE_API_KEY'),
    temperature=0.5,
    convert_system_message_to_human=True
)

template = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know and say it is not in the provided doc, don't try to make up an answer.
Keep the answer as concise as possible.

{context}

Question: {question}

Helpful Answer:
"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/api', methods=['POST'])
def api():
    global qa_chain
    if qa_chain is None:
        return jsonify({'error': 'Vector store is not initialized. Upload a PDF first.'}), 400

    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': 'No question provided.'}), 400

    question = data['question']
    try:
        result = qa_chain({"query": question})
        answer = result.get("result", "I'm sorry, I couldn't find an answer.")
    except Exception as e:
        answer = f"An error occurred: {str(e)}"

    return jsonify({'answer': answer})


@app.route('/upload', methods=['POST'])
def upload_file():
    global vector_store, qa_chain
    if 'file' not in request.files:
        return jsonify({'error': 'No file part.'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        for old_file in os.listdir(app.config['UPLOAD_FOLDER']):
            old_file_path = os.path.join(app.config['UPLOAD_FOLDER'], old_file)
            if os.path.isfile(old_file_path):
                os.remove(old_file_path)

        file.save(file_path)
        vector_store = process_pdf(file_path)
        qa_chain = initialize_qa_chain()
        return jsonify({'message': 'File uploaded and processed successfully.'}), 200
    else:
        return jsonify({'error': 'Invalid file type.'}), 400


if __name__ == '__main__':
    app.run(debug=True)
