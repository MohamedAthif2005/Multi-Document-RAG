import re
from ollama import chat

with open("data.txt","r") as f:
    text = f.read()
chunks  = text.split("\n\n")


def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]','',text)
    return set(text.split())

query = input("you : ").lower()

query_word = preprocess(query)

best_chunk = ""
best_score = 0




for chunk in chunks:
    chunk_words = preprocess(chunk)
    score = len(chunk_words & query_word)

    if score > best_score:
        best_score = score
        best_chunk = chunk

context  = best_chunk
propmt  = f"""Use the context below to answer the question 

Context:{context}

Question:{query}
"""
response  = chat(model="qwen2.5:1.5b",messages=[{
    "role":"user",
    "content":propmt
}])

print(response["message"]["content"])


