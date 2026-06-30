import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

client  = chromadb.Client()

collection = client.create_collection(
    name="attention_paper"
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 100
)



from pypdf import PdfReader
from ollama import chat
from sentence_transformers import SentenceTransformer,util

reader = PdfReader("attention.pdf")
text = ""
for page in reader.pages:
    text+=page.extract_text()+"\n"

model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_split(text,chunk_size=300):
    chunks = []
    for i in range(0,len(text),chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks
chunks = splitter.split_text(text)
for i,chunk in enumerate(chunks):
    collection.add(ids=[str(i)],documents=[chunk])

results = collection.query(
    query_texts = ["what is positional encoding"],
    n_results = 1
)

print(results["documents"])