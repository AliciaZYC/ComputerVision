import os

class Config:
    # Folders
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'app', 'static/uploads')
    PROCESSED_FOLDER = os.path.join(os.getcwd(), 'app', 'static/processed')
    FONT_FOLDER = os.path.join(os.getcwd(), 'app', 'static/font')

    # Other configurations
    DEBUG = True
    SECRET_KEY = 'your_secret_key'
