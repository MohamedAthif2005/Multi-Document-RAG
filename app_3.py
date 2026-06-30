from sentence_transformers import SentenceTransformer,util

with open("notes.txt","r",encoding="utf-8") as f:
    text = f.read()

def chunk_text(text,chunk_size=200):
    chunks = []
    for i in range(0,len(text),chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

chunks = chunk_text(text,300)

model  = SentenceTransformer("all-MiniLM-L6-v2")

chunk_embeddings  = model.encode(chunks)

query = input("You : ")
query_embedding = model.encode([query])

scores = util.cos_sim(query_embedding,chunk_embeddings)[0]

top_indices = scores.argsort(descending=True)[:3]

context = ""
for index in top_indices:
    context+=chunks[index]+"\n\n"
propmt = f"""
Answer the question using only the context below

Context :
{context}

Question:
{query}
"""

from ollama import chat

response = chat(model="qwen2.5:1.5b",messages=[{
    "role":"user","content":propmt
}])
print(response["message"]["content"])