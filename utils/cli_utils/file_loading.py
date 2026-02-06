from pathlib import Path
import json

def load_movies():
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    data_dir = BASE_DIR / 'data'
    data_file = data_dir / 'movies.json'

    with open(data_file, 'r') as f:
        data = json.load(f)

    documents = data['movies']
    return documents