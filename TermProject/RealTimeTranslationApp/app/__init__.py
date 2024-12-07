from flask import Flask
import os

from app.routes.main import main_bp


def create_app():
    app = Flask(__name__)

    # Set configurations
    app.config.from_object('app.config.Config')

    # Ensure folders exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

    app.register_blueprint(main_bp)
    # app.register_blueprint(video.bp)

    return app
