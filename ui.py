import streamlit as st
from PIL import Image
from keras.models import load_model
from pickle import load
import os
from caption_module import extract_features
from caption_module import predict_caption
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

max_length=74

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
        # 이미지 캡셔닝 함수 호출 
        photo = extract_features(image_file)
        caption = predict_caption(model,photo , tokenizer, max_length)
        caption = caption.replace('startseq','').replace('endseq','')
        result = tran.translate(caption, src='en', dest='ko')
        st.write(result.text)
