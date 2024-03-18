import re
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
import os
import pickle
WORKING_DIR = './kaggle/working'
BASE_DIR = './kaggle/input/flickr30k/'
#텍스트 전처리
def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            # 한 번에 하나의 캡션을 가짐
            caption = captions[i]
            # 전처리 단계
            # 소문자로 변환
            caption = caption.lower()
            # 숫자, 특수 문자 등 삭제,
            caption = re.sub('[^A-Za-z\s]', '', caption)
            # 추가 공백 삭제
            caption =re.sub('\s+',' ', caption) 
            # 캡션에 시작 및 종료 태그 추가
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
            captions[i] = caption

with open(os.path.join(WORKING_DIR, 'tokenizer.pkl'), 'rb') as f:  #rb : 이진파일
    tokenizer = pickle.load(f)  
def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    # loop over images
    X1, X2, y = list(), list(), list()
    n = 0
    while 1:
        for key in data_keys:
            n += 1
            captions = mapping[key]
            # 각 캡션 처리
            for caption in captions:
                # 시퀀스를 인코딩
                seq = tokenizer.texts_to_sequences([caption])[0]
                # 시퀀스를 X, y 쌍으로 분할
                for i in range(1, len(seq)):
                    # 입력 및 출력 쌍으로 분할
                    in_seq, out_seq = seq[:i], seq[i]
                    # 패드 입력 시퀀스
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # 인코딩 출력 시퀀스
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    
                    # 시퀀스 저장
                    X1.append(features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)
            if n == batch_size:
                X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                yield [X1, X2], y
                X1, X2, y = list(), list(), list()
                n = 0

# a=tokenizer.texts_to_sequences('Two young guys with shaggy hair look at their hands while hanging out in the yard')
# print(pad_sequences(a[:1], maxlen=32)[0])
# print(to_categorical(a[1], num_classes=18320)[0])

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# generate caption for an image
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

# from PIL import Image
# import matplotlib.pyplot as plt
# def generate_caption(image_name):
#     # load the image
#     # image_name = "1001773457_577c3a7d70.jpg"
#     image_id = image_name.split('.')[0]
#     img_path = os.path.join(BASE_DIR, "Images", image_name)
#     image = Image.open(img_path)
#     captions = mapping[image_id]
#     print('---------------------Actual---------------------')
#     for caption in captions:
#         print(caption)
#     # predict the caption
#     y_pred = predict_caption(model, features[image_id], tokenizer, max_length)
#     print('--------------------Predicted--------------------')
#     print(y_pred)
#     plt.imshow(image)
#     plt.show()
# generate_caption("1001773457_577c3a7d70.jpg")