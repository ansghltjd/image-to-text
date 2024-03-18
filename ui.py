import streamlit as st
import numpy as np
from PIL import Image
from caption_module import idx_to_word, predict_caption
from keras.models import load_model
from pickle import load
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.utils import load_img, img_to_array
from keras import Model
from keras_preprocessing.sequence import pad_sequences
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 특징 추출
def extract_features(filename):
    # vgg16 모델 로드
    model = VGG16()
    # 모델을 재구성
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    image = load_img(filename, target_size=(224, 224))
    # 이미지 픽셀을 numpy 배열로 변환
    image = img_to_array(image)
    
    # 모델에 대한 데이터 재구성
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    
     # vgg를 위한 전처리 이미지
    image = preprocess_input(image)
    
     # 특징 추출
    feature = model.predict(image, verbose=0)
    return feature
max_length=74
def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break
      
    return in_text

def load_image(image_file):
    img = Image.open(image_file)
    return img

st.title("Image Captioning Model")

st.write("Upload your image and Generate Caption automatically")

image_file = st.file_uploader("Insert Image")

tokenizer = load(open('./kaggle/working/tokenizer.pkl', 'rb'))
model = load_model('./kaggle/working/best_model.h5')

from googletrans import Translator
tran = Translator()
if image_file is not None:
    # 이미지를 불러온다.
    image = load_image(image_file)
    st.image(image, caption="업로드한 이미지", use_column_width=True)
    
    # 이미지 캡셔닝 버튼
    if st.button('캡션 생성'):
        # 이미지 캡셔닝 함수 호출 (여기서는 가상의 함수를 사용)
        photo = extract_features(image_file)
        caption = predict_caption(model,photo , tokenizer, max_length)
        caption = caption.replace('startseq','').replace('endseq','')
        result = tran.translate(caption, src='en', dest='ko')
        st.write(result.text)
