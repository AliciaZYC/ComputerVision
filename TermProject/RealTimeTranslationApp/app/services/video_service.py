import cv2
import numpy as np
import easyocr
from googletrans import Translator
from PIL import Image, ImageDraw, ImageFont
import os
import subprocess
from flask import current_app
from app.services.ocr_service import detect_text
from app.services.translation_service import translate_texts


def process_image(path, target_language):
    try:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Image not found or unable to read: {path}")
        font_path = select_font(target_language)
        processed_img = process_image_and_frame(img, target_language, font_path)
        processed_filename = 'processed_' + os.path.basename(path)
        processed_filepath = os.path.join(current_app.config['PROCESSED_FOLDER'], processed_filename)
        cv2.imwrite(processed_filepath, processed_img)
        return processed_filename
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def draw_rectangles_and_text(image, translations, font_path):
    try:
        # Convert the image to a PIL image for better text rendering
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # Load a font for text rendering
        if not os.path.exists(font_path):
            raise FileNotFoundError(f"Font file not found: {font_path}")

        font = ImageFont.truetype(font_path, 24)

        for (bbox, translated_text) in translations:
            # Draw the rectangle
            tl = (int(bbox[0][0]), int(bbox[0][1]))  # Top-left corner
            br = (int(bbox[2][0]), int(bbox[2][1]))  # Bottom-right corner
            draw.rectangle([tl, br], outline=(0, 255, 0), width=2)

            # Draw the translated text above the rectangle
            text_position = (tl[0], tl[1] - 30)  # Slightly above the rectangle
            draw.text(text_position, translated_text, font=font, fill=(139, 0, 0))

        # Convert the PIL image back to a numpy array
        return np.array(img_pil)

    except Exception as e:
        print(f"Error drawing rectangles and text: {e}")
        return image


def select_font(target_language):
    try:
        # 根据目标语言返回合适的字体文件路径
        font_dir = current_app.config['FONT_FOLDER']
        if target_language == 'zh-CN':
            font_path = os.path.join(font_dir, 'chinese_font.ttc')  # 请替换为您的中文字体文件
        elif target_language == 'hi':
            font_path = os.path.join(font_dir, 'hindi_font.ttf')  # 印地语字体
        else:
            # 默认字体，例如 Arial Unicode 或其他支持多语言的字体
            font_path = os.path.join(font_dir, 'aria_unicode.ttf')

        if not os.path.exists(font_path):
            raise FileNotFoundError(f"Font file not found: {font_path}")

        return font_path
    except Exception as e:
        print(f"Error selecting font for language {target_language}: {e}")
        raise


def process_image_and_frame(img, target_language, font_path):
    try:
        results = detect_text(img, 'en', True)
        translations = translate_texts(results, 'en', target_language)
        final_frame = draw_rectangles_and_text(img, translations, font_path)
        if final_frame is None:
            raise ValueError("Final frame is None after drawing.")
        return cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error processing image frame: {e}")
        return img


def convert_video_for_webstream(processed_filepath_temp, processed_filepath):
    # Re-encode the video to H.264 and AAC using ffmpeg
    ffmpeg_command = [
        "ffmpeg",
        "-i", processed_filepath_temp,
        "-vcodec", "libx264",
        "-acodec", "aac",
        processed_filepath
    ]

    try:
        if os.path.exists(processed_filepath):
            print(f"File '{processed_filepath}' already exists. Overwriting...")
            os.remove(processed_filepath)  # Delete the old file
        subprocess.run(ffmpeg_command, check=True)
        print(f"Video re-encoded successfully: {processed_filepath}")
        os.remove(processed_filepath_temp)
    except subprocess.CalledProcessError as e:
        print(f"Error during video re-encoding: {e}")
    except FileNotFoundError:
        print("Error: FFmpeg not found. Ensure it is installed and added to PATH.")
    except Exception as e:
        print(f"Unexpected error during video conversion: {e}")


def process_video(video_path, target_language):
    try:
        # 初始化视频捕获
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")

        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        font_path = select_font(target_language)

        # 输出视频路径
        processed_filename_temp = 'temp_' + os.path.basename(video_path)
        processed_filepath_temp= os.path.join(current_app.config['PROCESSED_FOLDER'], processed_filename_temp)

        processed_filename = 'processed_' + os.path.basename(video_path)
        processed_filepath = os.path.join(current_app.config['PROCESSED_FOLDER'], processed_filename)

        # 初始化视频写入器
        out = cv2.VideoWriter(processed_filepath_temp, fourcc, fps, (width, height))
        if not out.isOpened():
            raise ValueError("Error initializing video writer.")

        frame_skip = 10  # 每隔 5 帧处理一次
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                processed_frame = process_image_and_frame(frame, target_language, font_path)
                if processed_frame is not None:
                    for _ in range(frame_skip):
                        out.write(processed_frame)
                else:
                    print(f"Warning: Processed frame is None at frame {frame_count}.")

            frame_count += 1

        cap.release()
        out.release()

        convert_video_for_webstream(processed_filepath_temp, processed_filepath)

        return processed_filename

    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return None
