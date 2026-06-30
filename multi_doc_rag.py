import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

from ollama import chat

def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text+=page_text+"\n"
    return text

client  = chromadb.Client()
splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,chunk_overlap=100
    )

try:
    client.delete_collection("documents")
except:
    pass
collection = client.get_or_create_collection(name="documents")

docs_folder = "docs"
id_counter = 0

for file in os.listdir(docs_folder):
    if not file.endswith(".pdf"):
        continue
    file_path = os.path.join(docs_folder,file)

    text = extract_text(file_path)
    chunks = splitter.split_text(text)

    for chunk in chunks:
        collection.add(
            ids=[str(id_counter)],
            documents=[chunk],
            metadatas=[{
                "source":file
            }]
        )
        id_counter+=1
while True:
    query = input("You : ")
    if query.lower()=="/exit":
        break
    results = collection.query(
        query_texts=[query],
        n_results = 3
    )

    context = ""
    for doc in results["documents"][0]:
        context+=doc+"\n\n"
    print("\nSources")
    for meta in results["metadatas"][0]:
        print(meta["source"])


    prompt = f"""
    Answer only from the context.

    Context:
    {context}

    Question:
    {query}

    If the answer is not present, say:
    'I could not find that information in the documents.'
    """

    response = chat(
        model="qwen2.5:1.5b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    print("\nAI:", response["message"]["content"])