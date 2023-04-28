import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from modules import twitter_api_key

print(twitter_api_key())

