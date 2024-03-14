import re
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
            caption = re.sub('[^A-Za-z]', '', caption)
            # 추가 공백 삭제
            caption =re.sub('\s+',' ', caption) 
            # 캡션에 시작 및 종료 태그 추가
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
            captions[i] = caption
