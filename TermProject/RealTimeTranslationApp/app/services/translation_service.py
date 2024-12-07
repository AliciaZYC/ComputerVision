import cv2
import numpy as np
import easyocr
from googletrans import Translator
from PIL import Image, ImageDraw, ImageFont

def translate_texts(results, src, dest):
    translator = Translator()
    translations = []
    for (bbox, text, prob) in results:
        try:
            if text:
                result = translator.translate(text, src=src, dest=dest)
                translations.append((bbox, result.text))
                print(text + " to: " + result.text)
        except Exception as e:
            translations.append((bbox, text))  # Use original text if translation fails
            print(f"An error occurred: {e}")
    return translations