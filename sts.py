#STEP 1
from sentence_transformers import SentenceTransformer

#STEP 2
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

#STEP 3
sentence1 = "부자되고싶어"
sentence2 = "백수가 꿈이야"
# sentences = [
#     "The weather is lovely today.",
#     "It's so sunny outside!",
#     "He drove to the stadium.",
# ]

#STEP 4
embeddings1 = model.encode(sentence1)
embeddings2 = model.encode(sentence2)
print(embeddings1.shape)
# (384,):집에가고 싶다
# [3, 384]

#STEP 5
similarities = model.similarity(embeddings1, embeddings2)
print(similarities)
# tensor([[0.4235]]) 집에가고 싶다, 살려주세요
# tensor([[0.9256]]) 죽겠다, 살려주세요
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])