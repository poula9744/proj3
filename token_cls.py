# STEP 1
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification

#STEP 2 
# classifier = pipeline("ner", model="stevhliu/my_awesome_wnut_model")

tokenizer = AutoTokenizer.from_pretrained("Leo97/KoELECTRA-small-v3-modu-ner")
model = AutoModelForTokenClassification.from_pretrained("Leo97/KoELECTRA-small-v3-modu-ner")
ner = pipeline("ner", model=model, tokenizer=tokenizer)

#STEP 3
# text = "The Golden State Warriors are an American professional basketball team based in San Francisco."
text = "아이유 노래 검색해줘."

#STEP 4
result = ner(text)

#STEP 5
print(result)
#[{'entity': 'B-PS', 'score': np.float32(0.98335534), 'index': 1, 'word': '아이유', 'start': 0, 'end': 3}]
# entity 분류가 사람

