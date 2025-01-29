import numpy as np
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont

# ocr 리더 생성
ocr = PaddleOCR(lang='korean')

# 이미지 로드
image_path = 'datasets/ocr1.jpg'
image = Image.open(image_path)

# 텍스트 검출
image_np = np.array(image)
result = ocr.ocr(image_np, cls=True)

# 이미지를 수정하기 위한 객체 생성
draw = ImageDraw.Draw(image)
font_path = 'fonts/NanumGothicBold.ttf'
font = ImageFont.truetype(font_path, 24)

# 결과 시각화
boxes = [line[0] for line in result[0]]
texts = [line[1][0] for line in result[0]]
scores = [line[1][1] for line in result[0]]

# 이미지박스와 텍스트 그리기
for (box, text, score) in zip(boxes, texts, scores):
    (top_left, top_right, bottom_right, bottom_left) = box
    top_left = tuple(map(int, top_left))
    bottom_right = tuple(map(int, bottom_right))
    print(f"{text} {score:.2f}")

    # 박스 그리기
    draw.rectangle([top_left, bottom_right], outline=(0,255,0), width=5)
    draw.text(bottom_left, text, font=font, fill=(255,0,0))

image.save("result.jpg")