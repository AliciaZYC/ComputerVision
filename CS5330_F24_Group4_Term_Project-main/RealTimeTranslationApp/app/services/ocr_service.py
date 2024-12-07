import cv2
import numpy as np
import easyocr
from googletrans import Translator
from PIL import Image, ImageDraw, ImageFont

def detect_text(image, lan, gpu):
    reader = easyocr.Reader([lan], gpu=gpu)
    return reader.readtext(image, detail=1, paragraph=False)