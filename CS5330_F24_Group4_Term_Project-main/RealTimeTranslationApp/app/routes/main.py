import os
from fileinput import filename

from flask import Flask, render_template, request, redirect, url_for, Blueprint, current_app, jsonify

from app.services.file_service import save_uploaded_file, get_processed_file
from app.services.video_service import process_video, process_image

main_bp = Blueprint('main', __name__)

@main_bp.route('/test')
def test():
    return render_template("index.html")


@main_bp.route('/')
def main_page():
    return render_template("index.html")


@main_bp.route('/upload', methods=['POST'])
def upload_file():
    # 检查是否有文件被上传
    if 'file' not in request.files:
        return 'No files were uploaded'
    file = request.files['file']
    file_type = request.form.get("file_type")
    target_language = request.form.get('language')

    if not file or not file_type or not target_language:
        return render_template('index.html', error='Please provide all required inputs.')

    filepath = save_uploaded_file(file)

    if file_type == 'image':
        processed_filename = process_image(filepath, target_language)
        print(f"Processed filename: {processed_filename}")  # Debugging
        # Respond with a redirect URL for the processed image page
        return jsonify({'redirect_url': url_for('main.processed_file', filename=processed_filename)}), 200
    elif file_type == 'video':
        processed_filename = process_video(filepath, target_language)
        print(f"Processed filename: {processed_filename}")  # Debugging
        # Respond with a redirect URL for the processed video page
        return jsonify({'redirect_url': url_for('main.processed_video', filename=processed_filename)}), 200
    else:
        return jsonify({'error': 'Invalid file type'}), 400



@main_bp.route('/processed/<filename>')
def processed_file(filename):
    return render_template('processed.html', filename=filename)


@main_bp.route('/processed_video/<filename>')
def processed_video(filename):
    return render_template('processed_video.html', filename=filename)
