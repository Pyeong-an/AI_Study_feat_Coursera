# The dataset is taken from the here (Original Source): Roboflow Aircraft Dataset Provided by a Roboflow user, License: CC BY 4.
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import zipfile
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.applications import VGG16
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
import random

# 랜덤 시드 고정 (재현성 확보)
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# 학습 관련 하이퍼파라미터 설정
batch_size = 32
n_epochs = 5
img_rows, img_cols = 224, 224
input_shape = (img_rows, img_cols, 3)

# 추가 라이브러리 (다운로드 및 압축해제용)
import tarfile
import urllib.request
import os
import shutil

# 다운로드할 데이터셋 tar 파일의 URL
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZjXM4RKxlBK9__ZjHBLl5A/aircraft-damage-dataset-v1.tar"

# 다운로드 파일 이름과 추출 폴더 이름 지정
tar_filename = "aircraft_damage_dataset_v1.tar"
extracted_folder = "aircraft_damage_dataset_v1"  # Folder where contents will be extracted

# 데이터셋 tar 파일 다운로드
urllib.request.urlretrieve(url, tar_filename)
print(f"Downloaded {tar_filename}. Extraction will begin now.")

# 기존 폴더가 존재하면 삭제 후 새로 추출
if os.path.exists(extracted_folder):
    print(f"The folder '{extracted_folder}' already exists. Removing the existing folder.")

    # Remove the existing folder to avoid overwriting or duplication
    shutil.rmtree(extracted_folder)
    print(f"Removed the existing folder: {extracted_folder}")

# tar 파일 압축 해제
with tarfile.open(tar_filename, "r") as tar_ref:
    tar_ref.extractall()  # This will extract to the current directory
    print(f"Extracted {tar_filename} successfully.")

# 학습, 검증, 테스트 디렉토리 경로 지정
extract_path = "aircraft_damage_dataset_v1"
train_dir = os.path.join(extract_path, 'train')
test_dir = os.path.join(extract_path, 'test')
valid_dir = os.path.join(extract_path, 'valid')

# 이미지 전처리용 ImageDataGenerator 생성
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# 학습 데이터 제너레이터 생성
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_rows, img_cols),   # Resize images to the size VGG16 expects
    batch_size=batch_size,
    seed = seed_value,
    class_mode='binary',
    shuffle=True # Binary classification: dent vs crack
)

# 검증 데이터 제너레이터 생성
valid_generator =  valid_datagen.flow_from_directory(
    directory=valid_dir,
    class_mode='binary',
    seed=seed_value,
    batch_size=batch_size,
    shuffle=False,
    target_size=(img_rows, img_cols)
)

# 테스트 데이터 제너레이터 생성
test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    class_mode='binary',
    seed=seed_value,
    batch_size=batch_size,
    shuffle=False,
    target_size=(img_rows, img_cols)
)

# VGG16 모델 로드 (imagenet 사전학습 가중치 사용)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, 3))

# VGG16 출력 후 flatten 처리
output = base_model.layers[-1].output
output = keras.layers.Flatten()(output)
base_model = Model(base_model.input, output)

# VGG16 층들 동결 (학습되지 않도록)
for layer in base_model.layers:
    layer.trainable = False

# 전체 모델 정의 (VGG16 + 사용자 정의 FC층)
model = Sequential()
model.add(base_model)
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid')) # 이진 분류용 출력층

model.compile(
    optimizer=Adam(learning_rate=0.0001),   # 작은 학습률의 Adam 옵티마이저
    loss='binary_crossentropy',             # 이진 분류 손실 함수
    metrics=['accuracy']                    # 정확도 지표
)

# 모델 학습 시작
history = model.fit(
    train_generator,
    epochs=n_epochs,
    validation_data=valid_generator,
)

# 학습 완료 후 히스토리 객체 저장
train_history = model.history.history  # After training


# --------------------------------------------
# ① 학습 손실(loss) 시각화
# --------------------------------------------
plt.title("Training Loss")
plt.ylabel("Loss")
plt.xlabel('Epoch')
plt.plot(train_history['loss'])
plt.show()

# --------------------------------------------
# ② 검증 손실(validation loss) 시각화
# --------------------------------------------
plt.title("Validation Loss")
plt.ylabel("Loss")
plt.xlabel('Epoch')
plt.plot(train_history['val_loss'])
plt.show()

# --------------------------------------------
# ③ 정확도(accuracy) 시각화
# --------------------------------------------
plt.figure(figsize=(5, 5))
plt.plot(train_history['accuracy'], label='Training Accuracy')
plt.plot(train_history['val_accuracy'], label='Validation Accuracy')
plt.title("Accuracy Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

# --------------------------------------------
# ④ 테스트셋 평가
# --------------------------------------------
# test_generator.samples // test_generator.batch_size 만큼 스텝 계산
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)

# 평가 결과 출력
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# 단일 이미지와 예측 결과를 시각화하는 함수 정의
def plot_image_with_title(image, model, true_label, predicted_label, class_names):
    plt.figure(figsize=(6, 6))
    plt.imshow(image)

    # Convert labels from one-hot to class indices if needed, but for binary labels it's just 0 or 1
    true_label_name = class_names[true_label]  # 실제 라벨 이름
    pred_label_name = class_names[predicted_label]  # 예측 라벨 이름

    # 제목에 실제/예측 라벨 출력
    plt.title(f"True: {true_label_name}\nPred: {pred_label_name}")
    plt.axis('off')
    plt.show()

# 테스트 데이터셋에서 이미지를 하나 뽑아 예측하는 함수
def test_model_on_image(test_generator, model, index_to_plot=0):
    # 테스트 제너레이터에서 한 배치 가져오기
    test_images, test_labels = next(test_generator)

    # 모델로 예측 (확률값)
    predictions = model.predict(test_images)

    # 이진 분류라 0.5 이상이면 1, 이하이면 0
    predicted_classes = (predictions > 0.5).astype(int).flatten()

    # 클래스 인덱스 → 클래스 이름 매핑
    class_indices = test_generator.class_indices
    class_names = {v: k for k, v in class_indices.items()}  # Invert the dictionary

    # 선택할 이미지 인덱스
    image_to_plot = test_images[index_to_plot]
    true_label = test_labels[index_to_plot]
    predicted_label = predicted_classes[index_to_plot]

    # 위에서 정의한 시각화 함수 호출
    plot_image_with_title(image=image_to_plot, model=model, true_label=true_label, predicted_label=predicted_label, class_names=class_names)

# 테스트셋에서 예시 이미지 1장을 뽑아 결과 확인
test_model_on_image(test_generator, model, index_to_plot=1)


#Load the required libraries
import torch
import tensorflow as tf
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Hugging Face에서 BLIP Processor와 모델을 불러오기
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 커스텀 Keras Layer를 정의
class BlipCaptionSummaryLayer(tf.keras.layers.Layer):
    def __init__(self, processor, model, **kwargs):
        """
        BLIP Processor와 Model을 저장하는 사용자 정의 레이어 초기화
        """
        super().__init__(**kwargs)
        self.processor = processor
        self.model = model

    def call(self, image_path, task):
        """
        TensorFlow 그래프 내에서 tf.py_function을 사용해 외부 코드 실행
        """
        return tf.py_function(self.process_image, [image_path, task], tf.string)

    def process_image(self, image_path, task):
        """
        실제로 이미지 경로를 처리하고 캡션 또는 요약을 생성하는 로직
        """
        try:
            # 텐서 → 문자열 변환
            image_path_str = image_path.numpy().decode("utf-8")

            # 이미지 로드 및 RGB 변환
            image = Image.open(image_path_str).convert("RGB")

            # 작업 타입에 따라 프롬프트 결정
            if task.numpy().decode("utf-8") == "caption":
                prompt = "This is a picture of"  # Modify prompt for more natural output
            else:
                prompt = "This is a detailed photo showing"  # Modify for summary

            # BLIP 입력 생성
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")

            # BLIP 모델로 출력 생성
            output = self.model.generate(**inputs)

            # 출력 텍스트 디코딩
            result = self.processor.decode(output[0], skip_special_tokens=True)
            return result
        except Exception as e:
            # 오류 발생 시 메시지 출력
            print(f"Error: {e}")
            return "Error processing image"

# 위에서 정의한 Layer를 간단히 사용하는 헬퍼 함수
def generate_text(image_path, task):
    # BLIP Layer 인스턴스 생성
    blip_layer = BlipCaptionSummaryLayer(processor, model)
    # Layer를 호출하여 결과 문자열 생성
    return blip_layer(image_path, task)


# 예제 이미지 경로 설정
image_path = tf.constant("aircraft_damage_dataset_v1/test/dent/144_10_JPG_jpg.rf.4d008cc33e217c1606b76585469d626b.jpg")  # actual path of image

# 캡션 생성
caption = generate_text(image_path, tf.constant("caption"))
# Decode and print the generated caption
print("Caption:", caption.numpy().decode("utf-8"))

# 요약 생성
summary = generate_text(image_path, tf.constant("summary"))
# Decode and print the generated summary
print("Summary:", summary.numpy().decode("utf-8"))

# We will use the following image to display Caption and Summary for Task 9 and 10
# 다른 예제 이미지 경로
image_url = "aircraft_damage_dataset_v1/test/dent/149_22_JPG_jpg.rf.4899cbb6f4aad9588fa3811bb886c34d.jpg"
# 이미지 파일을 읽어와 시각화
img = plt.imread(image_url)
plt.imshow(img)
plt.axis('off')  # Hide the axis
plt.show()

# 두 번째 이미지 경로 Tensor 생성
image_path = tf.constant("aircraft_damage_dataset_v1/test/dent/149_22_JPG_jpg.rf.4899cbb6f4aad9588fa3811bb886c34d.jpg")  # actual path of image

# Generate a caption for the image
caption = generate_text(image_path, tf.constant("caption"))
# Decode and print the generated caption
print("Caption:", caption.numpy().decode("utf-8"))

# Generate a summary for the image
summary = generate_text(image_path, tf.constant("summary"))
# Decode and print the generated summary
print("Summary:", summary.numpy().decode("utf-8"))
