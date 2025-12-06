import os
import sys

# Ensure we use the local kaggle.json
# os.environ['KAGGLE_CONFIG_DIR'] = r'C:\Users\canoz\.kaggle'

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    print("Authentication successful!")
    print(api.dataset_list(search="mobile-games-ab-testing")[0])
except Exception as e:
    print(f"Error: {e}")
    # Print where it's looking for config
    print(f"Looking for config in: {os.path.expanduser('~/.kaggle')}")
