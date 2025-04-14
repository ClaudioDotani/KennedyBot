import os
from langchain_ollama import OllamaEmbeddings
import wikipedia
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_chroma import Chroma
from langchain.docstore.document import Document 
from dotenv import load_dotenv

def load_txt_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                documents.append(Document(page_content=text, metadata={"source": filename}))
    return documents


load_dotenv()
modelloLLM = os.getenv("LLM_MODEL")
modello_embeddings= os.getenv("EMBEDDINGS_MODEL")
document_directory = os.getenv("DOCUMENTS_DIRECTORY")
db_directory = os.getenv("DB_DIRECTORY")


#se la cartella db_directory è già presente chiedi se cancellare il contenuto e quindi cancellare cartella e contenuti e creare un nuovo db

if os.path.exists(db_directory):
    delete = input(f"Il database {db_directory} esiste già. Vuoi eliminarlo? (s/n): ")
    if delete.lower() == 's':
        import shutil
        shutil.rmtree(db_directory)
        print(f"Database {db_directory} eliminato.")
    else:
        print("Utilizzerò il database esistente.")
        exit()

embeddings = OllamaEmbeddings(
    model = modello_embeddings
)

vector_db = Chroma(
    embedding_function=embeddings,
    persist_directory=db_directory
)





#per ogni file di testo in una directory leggo i testi .txt  e creo un oggetto Document
documents = load_txt_documents(document_directory)


#cerco su wikipedia le pagine che contengono il titolo del file e la storia di robert kennedy e i suoi discorsi
wiki_titles = ["Robert F. Kennedy", "Robert F. Kennedy Jr.", "Robert F. Kennedy speeches"]
wiki_documents = []
for title in wiki_titles:
    try:
        page = wikipedia.page(title)
        content = page.content
        wiki_documents.append(Document(page_content=content, metadata={"source": title}))
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"Disambiguation error for {title}: {e}")
    except wikipedia.exceptions.PageError as e:
        print(f"Page error for {title}: {e}")
    except Exception as e:
        print(f"Errore durante il caricamento della pagina Wikipedia: {e}")



#  Combina tutti i documenti
documenti = documents + wiki_documents
if not documents:
    print("Nessun documento caricato. Controlla i percorsi e i titoli delle pagine.")

#  Suddividi i documenti in chunk per una migliore granularità
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs_chunks = text_splitter.split_documents(documents)
print(f"Documento suddiviso in {len(docs_chunks)} chunk.")

#scrivo tutti i chunk in un file di testo con il nome chunk.txt
with open("chunk.txt", "w", encoding="utf-8") as f:
    for doc in docs_chunks:
        f.write(str(doc.metadata)+"\n"+doc.page_content + "\n\n")



# 6. Ingerisci i chunk nel database Chroma (persistente)
vector_db.add_documents(docs_chunks)
print("Database Chroma creato/aggiornato con successo.")

   

