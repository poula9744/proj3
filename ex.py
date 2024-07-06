# STEP 1
from transformers import pipeline


#STEP 2 : 모델을 자동으로 받아줌
classifier = pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC") 
question_answerer = pipeline("question-answering", model="yjgwak/klue-bert-base-finetuned-squard-kor-v1", tokenizer="yjgwak/klue-bert-base-finetuned-squard-kor-v1")
ner = pipeline("ner", model="Leo97/KoELECTRA-small-v3-modu-ner", tokenizer="Leo97/KoELECTRA-small-v3-modu-ner")

# STEP 3 
text = "찌그러진 ‘철밥통’...공무원 월급 보니 '충격'" 
body = """
9급 초임(1호봉) 공무원의 월평균 급여액이 최저임금보다 16만원 많은 수준인 것으로 나타났다.
6일 전국공무원노동조합 자료에 따르면 올해 9급 1호봉은 매달 본봉 187만7000원, 직급 보조비 17만5000원, 정액 급식비 14만원, 정근수당 가산금 3만원을 더해 세전 222만2000원을 받는다.
특히 이는 올해 최저시급(9860원)을 바탕으로 환산한 월급 206만740원보다 16만1260원 많은 수준이다.
내년 최저시급이 5% 인상된다고 가정하면 이 차이는 불과 5만8850원밖에 되지 않는다.
심지어 9급 공무원이 월 10시간까지 가능한 초과근무의 시간당 수당 단가는 9414원에 불과하다. 올해 최저시급보다도 낮다.
올해 초 인사혁신처는 9급 1호봉의 연봉이 작년보다 6% 넘게 오른 3010만원(월평균 251만원)으로, 역대 처음으로 3000만원을 넘었다고 발표했다.
다만 이 금액은 공무원이 월 최대로 받을 수 있는 초과근무 수당과 연 2회 지급받는 명절 휴가비까지 합산한 수치라는 설명이다.
올해 정부가 9급 1호봉의 보수 인상률을 전체 공무원 보수 평균 인상률(2.5%) 대비 높게 책정했지만, 하위직 공무원이 받는 보수는 고물가 시대에 터무니없이 적은 수준이라는 지적이 제기된다.
낮은 월급 때문에 공무원의 인기도 줄어들고 있다.
올해 9급 공채시험의 경쟁률은 21.8대 1로, 1992년(19.3대 1) 이후 가장 낮았다. 경쟁률은 2016년(53.8대 1) 이후 8년 연속 하락세다. 2011년만 해도 9급 공채 경쟁률이 93.3대 1에 달할 정도로 공무원에 대한 직업 선호도가 높았던 바 있다.
낮은 급여와 부족한 처우 탓에 한때 ‘철밥통’으로 여겨졌던 공무원에 대한 선호도가 갈수록 낮아지고 있다는 분석이 나온다.
공무원노조 관계자는 “흔히 공무원을 철밥통이라고 부르는데, 그 철밥통은 찌그러진 지 오래”라며 “악성 민원과 업무 과중도 문제지만, 이보다 더욱 근본적인 문제는 낮은 임금”이라고 지적했다.
"""

question = ""

#STEP 4

qa = question_answerer(question=question, context=body)
# result = classifier(text)

token_cls1 = ner(text)
result1 = classifier(text)

token_cls = ner(body)


print(token_cls1)
print(qa)
print(result1)



#STEP 5
