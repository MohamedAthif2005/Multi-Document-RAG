from sentence_transformers import SentenceTransformer,util

model = SentenceTransformer("all-MiniLM-L6-V2")

sentences = ["Cats are animals","Dogs are pets","Cars use fuel"]

embeddings  = model.encode(sentences)

query = "which sentence is about transportation"

query_embedding  = model.encode(query)

scores = util.cos_sim(query_embedding,embeddings)

best_index  = scores.argmax()

print(sentences[best_index])

