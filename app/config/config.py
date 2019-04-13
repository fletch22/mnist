import logging
import os
import random
import string
from datetime import datetime

from app.services import file_services

PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_FOLDER_PATH = os.path.join(PROJECT_ROOT_PATH, 'data')
NBA_MATCHES_DIR = 'C:\\Users\\Chris\workspaces\\nba_scraper\\matches\\united_states\\nba'

os.makedirs(DATA_FOLDER_PATH, exist_ok=True)

TEMP_DIR = os.path.join(PROJECT_ROOT_PATH, 'temp')

os.makedirs(TEMP_DIR, exist_ok=True)


def clean_temp_folders():
    folders = file_services.get_folders(TEMP_DIR)
    for folder_path in folders:
        all_files = file_services.get_files(folder_path)
        age_hours = get_age_file_object(folder_path)

        if len(all_files) == 0 and age_hours > 24:
            logging.info(f'Deleting old empty folders {folder_path}')
            os.rmdir(folder_path)


def get_age_file_object(folder_path):
    mod_time = os.stat(folder_path).st_mtime
    dt_object = datetime.fromtimestamp(mod_time)
    dt_now = datetime.now()
    diff = dt_now - dt_object
    days, seconds = diff.days, diff.seconds
    hours = days * 24 + seconds // 3600
    return hours


clean_temp_folders()


def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


SESSION_ID = id_generator()
SESSION_DIR = os.path.join(TEMP_DIR, SESSION_ID)

os.makedirs(SESSION_DIR, exist_ok=True)

LOG_PATH = ensure_dir(os.path.join(PROJECT_ROOT_PATH, 'logs'))
SPORTS_DIR = ensure_dir(os.path.join(DATA_FOLDER_PATH, 'sports'))
NBA_DIR = ensure_dir(os.path.join(SPORTS_DIR, 'nba'))
TEAM_GAME_DIR = ensure_dir(os.path.join(NBA_DIR, 'team_game'))
SHELVES_DIR = os.path.join(TEMP_DIR, 'shelves')
