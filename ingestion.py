import os
import wikipedia
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
<<<<<<< Updated upstream
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_chroma import Chroma
from langchain.docstore.document import Document 
from dotenv import load_dotenv

load_dotenv()
modelloLLM = os.getenv("LLM_MODEL")
modello_embeddings= os.getenv("EMBEDDINGs_MODEL")
document_directory = os.getenv("DOCUMENTS_DIRECTORY")
db_directory = os.getenv("DB_DIRECTORY")


=======
from langchain_chroma import Chroma  # Per il database vettoriale Chroma
from langchain.docstore.document import Document  # Per creare documenti personalizzati
from langchain_ollama import OllamaEmbeddings

from dotenv import load_dotenv
load_dotenv()
pdf_directory = os.getenv("PDF_DIRECTORY") 
persist_directory = os.getenv("PERSISTENT_DIRECTORY")
model_embeddings = os.getenv("MODEL_EMBEDDING")

embeddings = OllamaEmbeddings(
    model = model_embeddings
)

vector_db = Chroma(
    embedding_function=embeddings,
    persist_directory=persist_directory
)
>>>>>>> Stashed changes

def load_pdf_documents(pdf_directory):
    """Carica tutti i documenti PDF dalla directory specificata."""
    pdf_files = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) if f.endswith(".pdf")]
    documents = []
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(pdf_file)
            docs = loader.load()
            documents.extend(docs)
            print(f"Caricato PDF: {pdf_file}")
        except Exception as e:
            print(f"Errore nel caricare il PDF {pdf_file}: {e}")
    return documents

def load_wikipedia_pages(page_titles):
    """Recupera pagine da Wikipedia date le relative voci e crea documenti LangChain."""
    wiki_documents = []
    for title in page_titles:
        try:
            # Imposta la lingua, se necessario (es. 'it' per italiano)
            wikipedia.set_lang("en")
            page = wikipedia.page(title)
            # Crea un documento con il contenuto della pagina e metadata che indicano la fonte
            doc = Document(page_content=page.content, metadata={"source": f"Wikipedia: {title}"})
            wiki_documents.append(doc)
            print(f"Caricata pagina Wikipedia: {title}")
        except Exception as e:
            print(f"Errore nel caricare la pagina Wikipedia '{title}': {e}")
    return wiki_documents

def main():
    # 1. Carica documenti PDF
     # modifica questo percorso secondo le tue esigenze
    pdf_documents = load_pdf_documents(pdf_directory)

    # 2. Carica pagine da Wikipedia
    wikipedia_titles = [
        "John F. Kennedy"
    ]
    wiki_documents = load_wikipedia_pages(wikipedia_titles)

    # 3. Combina tutti i documenti
    documents = pdf_documents + wiki_documents
    if not documents:
        print("Nessun documento caricato. Controlla i percorsi e i titoli delle pagine.")
        return

    # 4. Suddividi i documenti in chunk per una migliore granularità
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_chunks = text_splitter.split_documents(documents)
    print(f"Documento suddiviso in {len(docs_chunks)} chunk.")


    # 6. Ingerisci i chunk nel database Chroma (persistente)
    vector_db.add_documents(docs_chunks)
    print("Database Chroma creato/aggiornato con successo.")

   
if __name__ == "__main__":
    main()
