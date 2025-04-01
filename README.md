HeartGPT Application
===================================================
심박 데이터 생성 트랜스포머 모델을 활용한 Emotion Classification 모델 개발
-----------------------------
[HeartGPT 트랜스포머 모델 논문 링크](https://www.arxiv.org/abs/2407.20775)


## Train Data Preprocessing
__*concat_ppt_data.py*__ 를 통해 삼성 스마트워치로 수집한 청색광 PPG 데이터를 각 감정 별로 분류하여 피험자 별 PPG 데이터를 한 파일로 전부 concatenate 합니다. 
학습에 최적화 된 필터링 방법을 통해 Negative PPG data와 Positive PPG data의 전처리를 진행하여 학습 데이터의 전처리를 진행합니다. 


__*preprocess_and_save_negative.py*__ 와 __*preprocess_and_save_positive.py*__ 를 앞서 진행한 concatenate 시킨 각 감정별 데이터를 500 블록으로 나누어 
cutoff frequency ```[0.5, 8]```인 band-pass filter를 통과시켜 심박 신호에 맞게 처리를 해주는 ```heartpy``` 라이브러리 내의 함수를 사용하여 나온 각 블록 별 피크 개수를 기준으로 사용 가능한 피크를 추출합니다.
![PPG data visualization](https://github.com/kb081544/Emotion-Classification-2-Transformer-Application/blob/ef11ab0c1c8b70e42b84cb0013e2fbe99de86ba6/processed_data/emotion_data_visualization.png)


__*Heart_PT_generate_negative.py*__ 와 __*Heart_PT_generate_positive.py*__ 로 각 데이터를 증강시킵니다. Negative의 경우, 각 청크의 피크 값의 가우시안 분포를 활용하여 피크의 μ+1σ 데이터를 증강시켜 1로 레이블링, Positive는 전부 0으로 레이블링합니다.
![학습 데이터 설명](https://github.com/kb081544/Emotion-Classification-2-Transformer-Application/blob/0c1dea0c2aa538182cd2b8fab406fba49711fdc4/processed_data/figures/train_data_explanation.png)
μ+1σ, 즉 negative 데이터를 확보하기 위해 필요한 threshold 값은 ```threshold_value_1.csv```로 저장해 test 데이터를 레이블링 할 때 사용합니다. 
이러한 전처리는 __*augmented_data_process.py*__ 에서 진행되며 train 데이터와 test 데이터를 ```.npy``` 파일로 X, y 각각 나누어 모델 개발에 필요한 데이터 구축을 완료합니다. 



## 1D CNN Classification 모델
위와 같은 과정을 거친 PPG 데이터는 *1 Dimensional Convolutional Neural Network* 모델의 입력으로 사용됩니다. __*augmented_data_1d_CNN.py*__ 에서 모델을 만들고 h5 파일 형식으로 모델을 저장하여 활용할 수 있도록 합니다.
정확도와 loss는 모델의 진행 과정에 따라 시각화되어 'training_history.png' 파일로 저장되고, ![정확도 및 손실율](https://github.com/kb081544/Emotion-Classification-2-Transformer-Application/blob/13258b28d5a6d8e436badf2cbeffa710135f0316/Dataset/processed_data/figures/training_history.png)
혼동 행렬 또한 모델의 예측 값을 시각화하여 'confusion_matrix.png 파일로 저장됩니다. ![혼동 행렬](https://github.com/kb081544/Emotion-Classification-2-Transformer-Application/blob/13258b28d5a6d8e436badf2cbeffa710135f0316/Dataset/processed_data/figures/confusion_matrix.png)



## Test Data Preprocessing and Application
__*model_application_results_CNN_results_segments.py*__ 및 __*model_application_results.py*__ 파일은 VR 공포게임을 통해 확보한 테스트 데이터로 결과를 나타내는 코드로, test_visualization 파일 안에 시각화 결과가 저장됩니다.
테스트 데이터의 경우 __절반 앞부분은 0__ , __절반 뒷 부분은 *청크의 평균이 threshold 값을 넘는 경우* 1__ 로 레이블링합니다. 
또한, 훈련 데이터 확보 때와는 다르게 __청크의 피크 개수가 특정 개수를 넘지 않는 경우 청크를 -1__ 로 레이블링하여 해당 청크를 reject 시키지 않으나, 테스트 결과에 영향을 미치지 않도록 합니다.
PPG 신호에 노이즈가 많이 있어 ```heartpy``` 라이브러리가 __심박 신호를 전처리 할 수 없는 경우__ 또한 활용을 하되, 테스트 결과에 영향을 주지 않도록 -1로 레이블링합니다. 


- __*model_application_results*__ 는 각 파일을 ```{파일 이름}_analysis.png``` 로 저장하여 예측 결과 및 label 별 모델 accuracy, 레이블 분포와 분석 요약과 전체 파일을 시각화 한 차트를 저장합니다. 
  - 파일 별 분석 시각화 예시
    - ![file name analysis](https://github.com/kb081544/Emotion-Classification-2-Transformer-Application/blob/13258b28d5a6d8e436badf2cbeffa710135f0316/Dataset/processed_data/test_visualization/jh_left_analysis.png)
  - 전체 파일 분석 시각화
    - ![overall analysis](https://github.com/kb081544/Emotion-Classification-2-Transformer-Application/blob/13258b28d5a6d8e436badf2cbeffa710135f0316/Dataset/processed_data/test_visualization/overall_results.png)
- __*model_application_results_CNN_results_segments.py*__ 또한 거의 비슷하지만 앞/뒤 정보를 포함하여 분석 결과를 파일 별로 저장하였으며 앞/뒤 청크를 시각화하여 차이를 보여줍니다. 또한 오분류 및 맞게 분류된 청크를 시각화하여 분석에 용이하도록 하였습니다.
  - 파일별 분석 시각화 예시 
    - ![file name analysis colored](https://github.com/kb081544/Emotion-Classification-2-Transformer-Application/blob/13258b28d5a6d8e436badf2cbeffa710135f0316/Dataset/processed_data/test_visualization/jh_left_analysis_colored.png)
    - ![file name avg comparison](https://github.com/kb081544/Emotion-Classification-2-Transformer-Application/blob/13258b28d5a6d8e436badf2cbeffa710135f0316/Dataset/processed_data/test_visualization/jh_left_avg_comparison.png)
    - ![file name overalpping chunks](https://github.com/kb081544/Emotion-Classification-2-Transformer-Application/blob/13258b28d5a6d8e436badf2cbeffa710135f0316/Dataset/processed_data/test_visualization/jh_left_overlapping_chunks.png)
  - 전체 파일 분석 시각화 : 파이 분포 예시 및 파형 시각화 예시
    - ![true1 pred0 pie](https://github.com/kb081544/Emotion-Classification-2-Transformer-Application/blob/13258b28d5a6d8e436badf2cbeffa710135f0316/Dataset/processed_data/test_visualization/true1_pred0_sources_pie.png)
    - ![true1 pred0 visualization](https://github.com/kb081544/Emotion-Classification-2-Transformer-Application/blob/13258b28d5a6d8e436badf2cbeffa710135f0316/Dataset/processed_data/test_visualization/all_true1_pred0_segments.png)
    - 분류 결과별 평균 PPG 파형
      - ![ppg signal visualization by the results](https://github.com/kb081544/Emotion-Classification-2-Transformer-Application/blob/13258b28d5a6d8e436badf2cbeffa710135f0316/Dataset/processed_data/test_visualization/classification_average_comparison.png)

## File Directory Summarize
### Dataset
- ```.py``` 파일이 아닌 대부분의 파일은 ```Dataset``` 파일 안에 있습니다. 모델의 input 모양에 맞체 전처리가 완료된 ```.npy``` 파일, ```.h5``` 딥러닝 모델이 바로 존재합니다
  - ```N```, ```P``` : raw train 데이터
  - ```train_data``` : 전처리 및 증강된 train 데이터
  - ```test_data``` : raw test 데이터
  - ```sample_plots``` : 랜덤한 train 데이터의 시각화
  - ```test_visualization``` : test 데이터의 시각화
  - ```figures``` : 분석을 한 데이터들의 시각화
### HeartGPT-main
- __*Transformer_Generating_and_1D-CNN_Classification_Model*__ : 현재 진행 중인 연구로 HeartGPT로 증강한 데이터를 활용해 분류 모델을 개발
  - 파일 별 자세한 설명은 앞서 설명한 내용에 전부 포함되어 있습니다.
- __*Transformer_Pretraining_and_Finetuning*__ : 현재 연구에서 활용하는 데이터셋을 활용해 HeartGPT 모델을 활용한 생성형 모델 개발 및 Finetuning
  - finetuning : 마지막 레이어를 변경하여 0과 1로 구분하는 classication 모델 
- __*Model_files*__ : 생성형 모델을 개발하면서 저장하는 ```.pth``` 파일 저장 경로
- __*classfication_three_classes*__ : 0과 1로 분류되는 청크 및 활용 불가능한 청크를 살리되 -1로 classification하는 모델 개발

### Review for Emotion Classification 1
- 이전에 진행한 ```Emotion Classification``` 연구를 설명하기 위해 진행한 실습 코드
