from langchain.document_loaders import PyPDFLoader, DirectoryLoader, UnstructuredPDFLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import Chroma
import os
from pprint import pprint
import json
import re
CHROMA_PATH = "./chroma_langchain_db"
DATA_PATH = r"D:\Projects\RAG\data_source\generative_ai"
DATA_PATH_2 = r"C:\Users\LAPTOP\OneDrive\Documents\Zalo Received Files"


def load_pdf(file_path):
    """
    Load PDF documents.

    Args:
        file_path: str

    Returns: list[Document]
    """
    loader = UnstructuredPDFLoader(file_path=file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents from {file_path}.")
    return documents


def load_documents(data_path):
    """
    load documents from a directory of PDF files.

    Args:
        data_path: str

    returns: list[Document]
    """
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents from {data_path}.")
    return documents


def load_json(file_path):
    """
    Load JSON documents and extract content.

    Args:
        file_path: str

    Returns: list[Document]
    """
    loader = JSONLoader(file_path=file_path,
                        jq_schema=".markdown"
                        )
    documents = loader.load()
    return documents



def split_text(documents: list[Document]):
    """
    Split documents into smaller chunks.

    Args:
        documents: list[Document]

    returns: list[Document]
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=30,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    # print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks


def create_db(collection_name="simple-rag"):
    """
    Create a new vector database and persist it to disk.

    Returns: ChromaDB named vector_db
    """
    embedding = OllamaEmbeddings(model="llama3.2")
    vector_db = Chroma(
        embedding_function=embedding,
        collection_name=collection_name,
        persist_directory=CHROMA_PATH
    )
    vector_db.persist()
    return vector_db


def load_saved_db(collection_name="simple-rag"):
    """
    Load an existing vector database from disk.

    Returns: ChromaDB named vector_db
    """
    embedding = OllamaEmbeddings(model="llama3.2")
    vector_db = Chroma(
        embedding_function=embedding,
        collection_name=collection_name,
        persist_directory=CHROMA_PATH
    )
    return vector_db


def initialize_db(collection_name="simple-rag"):
    """
    Initialize the vector database.

    if the database does not exist, create a new one.
    else load the existing one.

    Returns: ChromaDB named vector_db
    """
    if not os.path.exists(CHROMA_PATH):
        vector_db = create_db(collection_name=collection_name)
    else:
        vector_db = load_saved_db(collection_name=collection_name)
    return vector_db


def add_to_db(vector_db, documents):
    """
    Add documents to an existing vector database.

    Args:
        vector_db: ChromaDB
        documents: list[Document]

    Returns: ChromaDB named vector_db which added documents
    """
    vector_db.add_documents(documents)
    vector_db.persist()
    return vector_db


def delete(vector_db, file):
    """
    Delete a document from the vector database.

    Args:
        vector_db: ChromaDB
        file: str

    Returns: ChromaDB named vector_db which deleted document
    """
    ids = vector_db.get(where={'source': file})['ids']
    vector_db.delete(ids)
    print(f"Deleted {file}")
    return vector_db


def load_and_add_json_to_db(file_path, vector_db):
    """
    Load JSON file and add its content to the vector database.

    Args:
        file_path: str
        vector_db: ChromaDB

    Returns: ChromaDB named vector_db which added documents
    """
    documents = load_json(file_path)
    vector_db = add_to_db(vector_db, documents)
    return vector_db
    
def clean_data(raw_text):
    # Loại bỏ khoảng trắng thừa
    cleaned_text = re.sub(r'\s+', ' ', raw_text)
    
    # Loại bỏ các ký tự thừa như dấu chấm câu, biểu tượng
    cleaned_text = re.sub(r'[\n\r\t]', '', cleaned_text)  # Loại bỏ xuống dòng, tab
    cleaned_text = re.sub(r'[^\w\s,./-]', '', cleaned_text)  # Giữ lại các ký tự cơ bản
    
    # Loại bỏ khoảng trắng thừa còn sót lại
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)
    
    # Trim khoảng trắng ở đầu và cuối
    cleaned_text = cleaned_text.strip()

    return cleaned_text
vector_db = initialize_db()
j = load_json(r"C:\Users\LAPTOP\OneDrive\Máy tính\check\result_without_links.json")
chunk = split_text(j)
vector_db = add_to_db(vector_db, chunk)
print("Done")