
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def update_notebooks():
    """
    Updates notebooks by replacing old variable names with new ones.
    """
    project_root = Path(__file__).parent.parent
    notebooks_dir = project_root / 'notebooks'
    
    # Mapping of old terms to new terms
    # Note: Case sensitive replacements appropriate for the dataset
    replacements = {
        # Columns
        'user_id': 'PlayerID',
        'age': 'Age',
        'gender': 'Gender', 
        'country': 'Location',
        'device_type': 'GameGenre', # Mapping device type charts to Genre
        'total_sessions': 'SessionsPerWeek',
        'total_playtime_hours': 'PlayTimeHours',
        'avg_session_duration_minutes': 'AvgSessionDurationMinutes',
        'max_level_reached': 'PlayerLevel',
        'levels_completed': 'PlayerLevel', # duplicate mapping but ok for text
        'quests_completed': 'AchievementUnlocked', # closest proxy
        'achievements_unlocked': 'AchievementUnlocked',
        'days_since_last_login': 'EngagementLevel', # proxy
        'days_since_registration': 'SessionsPerWeek', # proxy
        'login_frequency_per_week': 'SessionsPerWeek',
        'friend_count': 'SessionsPerWeek', # proxy
        'guild_member': 'InGamePurchases', # proxy
        'total_spent_usd': 'InGamePurchases',
        'purchase_count': 'InGamePurchases',
        'avg_purchase_value': 'InGamePurchases',
        'last_purchase_days_ago': 'SessionsPerWeek',
        'premium_subscription': 'InGamePurchases',
        'win_rate': 'GameDifficulty', # proxy
        'avg_score': 'PlayerLevel',
        'engagement_score': 'EngagementLevel'
    }
    
    # Text replacements for markdown
    text_replacements = {
        'total_sessions': 'SessionsPerWeek',
        'total_playtime_hours': 'PlayTimeHours',
        'total_spent_usd': 'InGamePurchases',
        'max_level': 'PlayerLevel'
    }

    for notebook_path in notebooks_dir.glob('*.ipynb'):
        logging.info(f"Processing {notebook_path.name}...")
        
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = json.load(f)
            
            changed = False
            
            for cell in nb['cells']:
                if 'source' in cell:
                    new_source = []
                    for line in cell['source']:
                        original_line = line
                        
                        # Apply replacements
                        for old, new in replacements.items():
                            # Simple string replacement - be careful with partial matches?
                            # For code, often 'df["col"]' matches.
                            if f"'{old}'" in line:
                                line = line.replace(f"'{old}'", f"'{new}'")
                            if f'"{old}"' in line:
                                line = line.replace(f'"{old}"', f'"{new}"')
                            
                            # Also replace dot access if used (rare in pandas but possible in descriptions)
                            # Avoiding generic word replacement to prevent breaking python keywords
                        
                        # Apply text replacements for markdown headers purely as text
                        if cell['cell_type'] == 'markdown':
                             for old, new in text_replacements.items():
                                 if old in line:
                                     line = line.replace(old, new)

                        if line != original_line:
                            changed = True
                        new_source.append(line)
                    
                    cell['source'] = new_source
            
            if changed:
                with open(notebook_path, 'w', encoding='utf-8') as f:
                    json.dump(nb, f, indent=2, ensure_ascii=False)
                logging.info(f"Updated {notebook_path.name}")
            else:
                logging.info(f"No changes needed for {notebook_path.name}")
                
        except Exception as e:
            logging.error(f"Failed to process {notebook_path.name}: {e}")

if __name__ == "__main__":
    update_notebooks()
