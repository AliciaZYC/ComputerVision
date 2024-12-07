# app/services/file_service.py
import os
from werkzeug.utils import secure_filename
from flask import current_app

def save_uploaded_file(file):
    """Save the uploaded file to the configured upload folder."""
    filename = secure_filename(file.filename)
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    return filepath

def get_processed_file(filename):
    """Save the uploaded file to the configured upload folder."""
    filepath = os.path.join(current_app.config['PROCESSED_FOLDER'], filename)
    return filepath