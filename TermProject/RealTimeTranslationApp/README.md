
# Real-Time Text Detection, Recognition, and Translation System for Live Video Streams

## Project Purpose
This project is a real-time text recognition and translation system for live video streams. It leverages OpenCV for video processing, EasyOCR for text detection and recognition, and Google Translate for multilingual translations. This system can detect text from video frames, translate it into a chosen language, and overlay the translated text onto the video in real time, making it highly applicable for language learning, accessibility, and international video content enhancement.

## Features
- Detect and recognize text from live video or prerecorded video files
- Translate detected text into selected languages (supports Chinese, Hindi, and Spanish)
- Display the translated text on the video with language-specific fonts

## Requirements
Ensure you have Python 3.7 or higher installed. Below are the required dependencies:

1. easyOCR == 1.6.2
2. matplotlib == 3.6.2
3. opencv-python-headless == 4.5.4.60
4. googletrans == 4.0.0-rc1
5. Flask==2.3.3
6. numpy==1.24.3
7. Pillow==9.5.0
8. certifi==2023.7.22

5. Run `Install Certificates.command` (for Mac users) through Spotlight Search to activate

You can install these packages with:
```bash
pip install easyocr==1.6.2 pillow googletrans==4.0.0-rc1 matplotlib==3.6.2 opencv-python-headless==4.5.4.60
```

## Project Structure

```
RealTimeTranslationApp/
├── app/
│   ├── models/                     # Database models (may be applicable later)
│   │   ├── __init__.py             # Module initialization
│   ├── routes/                     # Contains route definitions
│   │   ├── __init__.py             # Module initialization
│   │   ├── main.py                 # Main application routes (upload, process, etc.)
│   ├── services/                   # Business logic for processing tasks
│   │   ├── __init__.py             # Module initialization
│   │   ├── file_service.py         # Handles file saving and management
│   │   ├── ocr_service.py          # Text detection and recognition (OCR)
│   │   ├── translation_service.py  # Handles text translation
│   │   ├── video_service.py        # Handles video processing
│   ├── static/                     # Static files (CSS, JS, etc.)
│   │   ├── css/                    # Stylesheets (empty, will be applicable later)
│   │   ├── js/                     # JavaScript files
│   │   │   ├── main.js             # Frontend logic for UI interactions
│   │   ├── font/                   # Font files for rendering translations
│   │   ├── uploads/                # Uploaded files
│   │   ├── processed/              # Processed files
│   ├── templates/                  # HTML templates
│   │   ├── index.html              # Main upload page
│   │   ├── processed.html          # Processed image results page
│   │   ├── processed_video.html    # Processed video results page
│   ├── __init__.py                 # Module initialization
│   ├── config.py                   # Configuration for uploads
├── tests/                          # Test files (empty, will be applicable later)
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies for the project
├── run.py                          # Application entry point
```

### **Directory Descriptions**

### `app/`
Contains the core application logic, divided into multiple subdirectories for modularity:
- **`models/`**: For database models (currently empty).
- **`routes/`**: Contains Flask route definitions.
- **`services/`**: Implements processing logic for files, text recognition, translation, and video processing.
- **`static/`**: Stores static assets such as JavaScript and CSS files.
- **`templates/`**: Stores HTML templates for rendering pages.

### `uploads/`
Stores uploaded files temporarily. Configuration for this directory is managed in `config.py`.

### `font/`
Contains font files used for rendering translated text in different languages.

### `tests/`
Reserved for unit and integration test cases (currently empty).

### `run.py`
The main entry point for running the Flask application.

### `requirements.txt`
Lists all Python dependencies required for the project. Install them with:
```bash
pip install -r requirements.txt
```

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.khoury.northeastern.edu/aliciazyc/CS5330_F24_Group4_Term_Project.git
   cd RealTimeTranslationApp
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Flask application:
   ```bash
   flask run
   ```

5. Access the application at `http://127.0.0.1:5000`.

6. Upload Image/Video and Select the Translation Language:
   When prompted, select the desired language,
   If no valid option is selected, the system defaults to English.
   - **1**: Chinese
   - **2**: Hindi
   - **3**: Spanish

7. Webcam Live: will be available later

## Module Descriptions

- **`detect_text`**: Initializes the OCR reader for English text detection and Detects text in the given frame.
- **`translate_texts`**: Translates detected text using Google Translate API.
- **`draw_rectangles_and_text`**: Overlays the translated text on the video frame.
- **`choose_language`**: Allows the user to select a language for translation.
- **`process_video`**: Processes the video or webcam stream, applying text detection, translation, and display in real-time.

## Known Issues and Considerations
- **Google Translate API Limitations**: The `googletrans` package uses an unofficial API for translation, which may occasionally fail. For robust translation, consider using the official Google Translate API.
- **Font Compatibility**: Ensure font files for non-English characters (e.g., Chinese and Hindi) are available. Missing fonts will result in blank text display for translations.
- **Performance**: Processing may slow down on machines with limited resources, especially when using a GPU for OCR. Lowering the frame rate of the input video may help.

## Test Videos
- ** [Webcam Live] https://drive.google.com/file/d/1GywAcsgwzJivwvf9JOm4L9UMPyL1oegU/view?usp=sharing
- ** [Upload Video] https://drive.google.com/file/d/1hvD2DTCaWXYnZixo2QUCC7z6l-vzhhGS/view?usp=sharing