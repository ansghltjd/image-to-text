#캡셔닝 모델 생성
import os
import pickle
import numpy as np
from requests import head
from tqdm.notebook import tqdm

#VGG16 모델에 입력하기 전에 이미지 데이터를 적절히 전처리 'preprocess_input'
from keras.applications.vgg16 import VGG16, preprocess_input
#load_img는 이미지를 불러오고, img_to_array는 해당 이미지를 배열로 변환
from keras.utils import load_img, img_to_array
# Tokenizer 각 단어를 고유한 정수에 매핑하여 텍스트 데이터를 숫자 데이터로 변환할 수 있게 해줌
from keras_preprocessing.text import Tokenizer
#pad_sequences 이 함수는 모든 텍스트 시퀀스를 동일한 길이로 패딩합니다. 이는 모델에 입력하기 전에 모든 데이터가 동일한 차원을 갖도록 필요한 작업
from keras_preprocessing.sequence import pad_sequences
#Model 입력과 출력을 연결하여 전체 모델을 정의. 모델을 구성한 후, 이를 컴파일하고 훈련시킴
from keras import Model
#to_categorical은 레이블을 원-핫 인코딩으로 변환하는 데 사용되며, plot_model은 모델의 구조를 시각화하는 데 사용
from keras.utils import to_categorical, plot_model
#모델을 구성하는 데 사용되는 다양한 케라스 레이어. 
#Input 레이어는 모델의 입력을 정의하고, Dense는 완전 연결 계층을, LSTM은 순환 신경망 계층을, Embedding은 단어 임베딩 계층을, Dropout은 과적합을 방지하기 위한 계층을 의미합니다. add 함수는 여러 계층의 출력을 합치는 데 사용
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from PIL import Image



os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
BASE_DIR = './kaggle/input/flickr30k/'
WORKING_DIR = './kaggle/working'

# vgg16 모델 로드
model = VGG16()
# 모델을 재구성
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

# 요약
print(model.summary())

# 이미지에서 특징 추출
features = {}
directory = os.path.join(BASE_DIR, 'Images')

#tqdm for문의 진행상황 확인, os.listdir 디렉토리에 있는 모든 파일을 가져온다.
print(directory)
a=0
for img_name in tqdm(os.listdir(directory)):
    
    # 파일에서 이미지 불러오기
    img_path = directory + '/' + img_name
    image = load_img(img_path, target_size=(224, 224))
    # 이미지 픽셀을 numpy 배열로 변환
    image = img_to_array(image)
    print(image)
    # 모델에 대한 데이터 재구성
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    
     # vgg를 위한 전처리 이미지
    image = preprocess_input(image)
    
     # 특징 추출
    feature = model.predict(image, verbose=0)

     # 이미지 ID 가져오기
    image_id = img_name.split('.')[0]
     # store feature
    features[image_id] = feature
    

# store features in pickle
pickle.dump(features, open(os.path.join(WORKING_DIR, 'features.pkl'), 'wb'))  #pickle.dump(data, file)
# 피클에서 특징 로드
with open(os.path.join(WORKING_DIR, 'features.pkl'), 'rb') as f:  #rb : 이진파일
    features = pickle.load(f)   #load : 파이썬 객체로 다시 불러옴

with open(os.path.join(BASE_DIR, 'captions.txt'), 'r',encoding="UTF-8") as f:
    next(f)   # 첫줄 건너뛰고 다음 데이터부터 읽어옴
    captions_doc = f.read()

# 캡션에 대한 이미지 매핑 생성
mapping = {}
# 캡션을 한 줄씩 읽어옴
for line in tqdm(captions_doc.split('\n')):
    # 쉼표(,)기준으로 나눕니다.
    tokens = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    
    # 이미지 ID에서 확장자 제거
    image_id = image_id.split('.')[0]
    
    # 캡션 목록을 문자열로 변환
    caption = " ".join(caption)
    
    # 필요한 경우 목록 생성
    if image_id not in mapping:
        mapping[image_id] = []
    # 캡션 저장
    mapping[image_id].append(caption)

