# HeartGPT Application
PPG 및 ECG 데이터를 생성해주는 트랜스포머 모델을 활용하여 Emotion Classification 모델 개발
[HeartGPT 트랜스포머 모델 논문 링크](https://www.arxiv.org/abs/2407.20775)

Negative PPG data와 Positive PPG data의 전처리를 진행한 후,
![PPG data visualization](processed_data/emotion_data_visualization.png)
Heart_PT_generate.py로 각 데이터를 증강시킨다. Negative의 경우, μ+1σ 데이터를 증강시켜 1로 레이블링, Positive는 전부 0으로 레이블링한다.
