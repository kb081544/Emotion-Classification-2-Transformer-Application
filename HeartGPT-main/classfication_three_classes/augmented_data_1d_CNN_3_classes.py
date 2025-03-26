import numpy as np
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 3개 클래스 데이터 경로
data_path = r"C:\Users\user\PycharmProjects\Emotion Classification 3\processed_data_3class"

X_train = np.load(os.path.join(data_path, "X_train.npy"))
X_test = np.load(os.path.join(data_path, "X_test.npy"))
y_train = np.load(os.path.join(data_path, "y_train.npy"))
y_test = np.load(os.path.join(data_path, "y_test.npy"))

print(f"train 데이터: {X_train.shape}, {y_train.shape}")
print(f"test 데이터: {X_test.shape}, {y_test.shape}")
print(f"input 데이터 shape: {X_train.shape}")

print("\n클래스 분포:")
for class_val in [-1, 0, 1]:
    print(f"클래스 {class_val}: {np.sum(y_train == class_val)} 샘플 (train), {np.sum(y_test == class_val)} 샘플 (test)")


def create_multiclass_model(input_shape, num_classes=3):
    model = Sequential([
        Conv1D(filters=64, kernel_size=5, activation='relu', padding='same', input_shape=input_shape),
        MaxPooling1D(pool_size=2),

        Conv1D(filters=128, kernel_size=5, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),

        Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),

        # 출력층 뉴런 수 변경 (3개 클래스)
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        # 멀티클래스 분류에 적합한 손실 함수로 변경
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

y_train_categorical = to_categorical(y_train_encoded, num_classes=3)
y_test_categorical = to_categorical(y_test_encoded, num_classes=3)

input_shape = (X_train.shape[1], X_train.shape[2])
model = create_multiclass_model(input_shape, num_classes=3)

model.summary()

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    filepath=os.path.join(data_path, 'best_emotion_model_3class.h5'),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

history = model.fit(
    X_train, y_train_categorical,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(data_path, 'training_history_3class.png'))
plt.show()

y_pred_prob = model.predict(X_test)
y_pred_encoded = np.argmax(y_pred_prob, axis=1)
y_pred = label_encoder.inverse_transform(y_pred_encoded)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"\n테스트 정확도: {accuracy:.4f}")
print("\n혼동 행렬:")
print(conf_matrix)
print("\n분류 보고서:")
print(class_report)

# 혼동 행렬 시각화
plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('혼동 행렬')
plt.colorbar()

classes = ['판단불가 (-1)', '부정 (0)', '긍정 (1)']
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = conf_matrix.max() / 2
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, conf_matrix[i, j],
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

plt.ylabel('실제 레이블')
plt.xlabel('예측 레이블')
plt.tight_layout()
plt.savefig(os.path.join(data_path, 'confusion_matrix_3class.png'))
plt.show()

from sklearn.metrics import roc_curve, auc
from itertools import cycle

plt.figure(figsize=(10, 8))

colors = cycle(['blue', 'red', 'green'])
class_names = ['판단불가 (-1)', '부정 (0)', '긍정 (1)']

for i, color, class_name in zip(range(3), colors, class_names):
    y_test_bin = (y_test_encoded == i).astype(int)
    y_score = y_pred_prob[:, i]

    fpr, tpr, _ = roc_curve(y_test_bin, y_score)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, color=color, lw=2,
             label=f'ROC 곡선 {class_name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('다중 클래스 ROC 곡선')
plt.legend(loc="lower right")
plt.savefig(os.path.join(data_path, 'roc_curve_3class.png'))
plt.show()

print(f"모든 결과와 그래프가 {data_path}에 저장되었습니다.")

model.save(os.path.join(data_path, 'emotion_classification_model_3class.h5'))
print("모델이 저장되었습니다.")