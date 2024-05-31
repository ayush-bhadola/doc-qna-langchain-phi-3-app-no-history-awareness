from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import ChatPromptTemplate
import torch
import tempfile
import os
from datetime import timedelta
import datetime
import warnings

warnings.filterwarnings("ignore")

def delete_temp_files():
    temp_folder_path = tempfile.gettempdir()
    current_date = datetime.datetime.now() - timedelta(minutes=5)
 
    try:
        files = os.listdir(temp_folder_path)
 
        for file in files:
            file_path = os.path.join(temp_folder_path, file)
 
            try:
                modification_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
 
                if modification_time < current_date:
                    os.remove(file_path)
                else:
                    pass
 
            except Exception as e:
                continue
 
    except FileNotFoundError:
        pass
    except PermissionError:
        pass
    except Exception as e:
        pass

DATA_PATH = 'data/'

loader = DirectoryLoader(DATA_PATH, glob="**/*", loader_cls=PyMuPDFLoader)


pdf_docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40,
    length_function=len,
    is_separator_regex=True
)



pdf_docs_text = text_splitter.split_documents(pdf_docs)


DB_FAISS_PATH = 'vectorestore_new/db_faiss'
embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-V2', model_kwargs = {"device": "cuda"} if torch.cuda.is_available() else {"device": "cpu"})

retriever = FAISS.from_documents(pdf_docs_text, embeddings).as_retriever()


model_path='models\Phi-3-mini-4k-instruct-q4.gguf'

n_gpu_layers = 1 
n_batch = 512

llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=1024,
    f16_kv=True, 
    verbose=True,
)



prompt = ChatPromptTemplate.from_template("""\nYou are here solely to assist with user inquiries within the confines of the provided context:

<context>
{context}
</context>

Question: {input}

\n""")


document_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(retriever, document_chain)

# cmd test
response = retrieval_chain.invoke({"context": pdf_docs, "input": 'Who"s Sherlock Holmes ?'})

print(response['answer'])

