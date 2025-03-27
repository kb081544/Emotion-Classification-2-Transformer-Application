# HeartGPT Application
## 심박 데이터 생성 트랜스포머 모델을 활용한 Emotion Classification 모델 개발
[HeartGPT 트랜스포머 모델 논문 링크](https://www.arxiv.org/abs/2407.20775)

concat_ppt_data.py를 통해 삼성 스마트워치로 수집한 청색광 PPG 데이터를 감정 별로 전부 concatenate 시켜 학습에 최적화 된 필터링 방법을 통해 Negative PPG data와 Positive PPG data의 전처리를 진행한 후,
![PPG data visualization](https://github.com/kb081544/Emotion-Classification-2-Transformer-Application/blob/ef11ab0c1c8b70e42b84cb0013e2fbe99de86ba6/processed_data/emotion_data_visualization.png)
Heart_PT_generate.py로 각 데이터를 증강시킵니다. Negative의 경우, μ+1σ 데이터를 증강시켜 1로 레이블링, Positive는 전부 0으로 레이블링합니다.
![학습 데이터 설명](https://github.com/kb081544/Emotion-Classification-2-Transformer-Application/blob/0c1dea0c2aa538182cd2b8fab406fba49711fdc4/processed_data/figures/train_data_explanation.png)
