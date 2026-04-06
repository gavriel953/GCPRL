"""Configuration settings for GCPRL Medical Image Enhancement Application."""

import os

class Config:
    """Base configuration."""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'gcprl-medical-dev-key-2024')
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100 MB max upload

    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm', 'tiff', 'tif', 'bmp'}

    # GCPRL defaults
    DEFAULT_K = 2.0
    DEFAULT_WINDOW_SIZE = 7
    MIN_K = 0.5
    MAX_K = 3.0
    MIN_WINDOW = 3
    MAX_WINDOW = 15

    # CLAHE defaults
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_GRID_SIZE = (8, 8)

    # File retention (seconds)
    FILE_TTL = 3600

    LOG_LEVEL = 'DEBUG'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


class ProductionConfig(Config):
    DEBUG = False
    LOG_LEVEL = 'INFO'


class DevelopmentConfig(Config):
    DEBUG = True
    TESTING = False


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig,
}
