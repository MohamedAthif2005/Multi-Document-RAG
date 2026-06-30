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

chunks = chunk_split(text)
chunk_embedding = model.encode(chunks,normalize_embeddings=True)

while True:
    query = input("You : ")
    if query.lower()=="exit":
        break
    query_embedding = model.encode([query],normalize_embeddings=True)

    scores = util.cos_sim(query_embedding,chunk_embedding)[0]
    top_indices = scores.argsort(descending=True)[:3]

    context = ""
    for index in top_indices:
        context+=chunks[index]+"\n\n"
    propmt = f"""
    Answer only from the context below.

Context:
{context}

Question:
{query}

If the answer is not in the context, say:
'I could not find that information in the document.'
"""
    response = chat(model="qwen2.5:1.5b",messages=[{
        "role":"user","content":propmt
    }])
    print("AI : ",response["message"]["content"])

    

    

