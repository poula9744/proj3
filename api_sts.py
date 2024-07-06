from fastapi import FastAPI, Form

#STEP 1
from sentence_transformers import SentenceTransformer


#STEP 2
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


app = FastAPI()


@app.post("/sts/")
async def sts(sentence1: str = Form(), sentence2: str = Form()):
    #STEP 3
    #STEP 4
    embeddings1 = model.encode(sentence1)
    embeddings2 = model.encode(sentence2)

    print("111111111", embeddings1)
    print("222222222",embeddings2)
    print(embeddings1.shape)

    #STEP 5
    similarities = model.similarity(embeddings1, embeddings2)
    print(similarities)
    return {"result": similarities.item()}