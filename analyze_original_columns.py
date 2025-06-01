import pandas as pd
import os
from collections import Counter

def analyze_original_dataset_columns():
    """Analyze all available columns in the original Oracle's Elixir datasets."""
    
    print("🔍 ANALYZING ORIGINAL ORACLE'S ELIXIR DATASET COLUMNS")
    print("=" * 60)
    
    data_dir = "Dataset collection"
    
    # Get all CSV files
    csv_files = []
    for file in os.listdir(data_dir):
        if file.endswith('.csv') and 'LoL_esports_match_data' in file and 'processed' not in file:
            csv_files.append(os.path.join(data_dir, file))
    
    csv_files = sorted(csv_files)
    print(f"📁 Found {len(csv_files)} original CSV files")
    
    # Analyze columns from the most recent file (2024)
    latest_file = None
    for file in csv_files:
        if '2024' in file:
            latest_file = file
            break
    
    if not latest_file:
        latest_file = csv_files[-1]  # Use the last file if 2024 not found
    
    print(f"📖 Analyzing structure from: {os.path.basename(latest_file)}")
    
    # Load just the header and a few rows to understand structure
    sample_df = pd.read_csv(latest_file, nrows=1000)
    
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"   Total columns: {len(sample_df.columns)}")
    print(f"   Sample size: {len(sample_df)} rows")
    
    # Show unique positions to understand data structure
    if 'position' in sample_df.columns:
        positions = sample_df['position'].unique()
        print(f"   Positions: {', '.join(positions)}")
    
    # Categorize columns by type
    print(f"\n📋 ALL AVAILABLE COLUMNS ({len(sample_df.columns)}):")
    print("=" * 60)
    
    # Group columns by category
    categories = {
        'Match Info': [],
        'Team/Player Info': [],
        'Champion/Bans': [],
        'Game Stats': [],
        'Economy': [],
        'Combat': [],
        'Vision': [],
        'Objectives': [],
        'Other': []
    }
    
    for col in sample_df.columns:
        col_lower = col.lower()
        
        if any(keyword in col_lower for keyword in ['gameid', 'date', 'patch', 'league', 'split', 'playoffs', 'game']):
            categories['Match Info'].append(col)
        elif any(keyword in col_lower for keyword in ['team', 'player', 'position', 'side']):
            categories['Team/Player Info'].append(col)
        elif any(keyword in col_lower for keyword in ['champion', 'ban', 'pick']):
            categories['Champion/Bans'].append(col)
        elif any(keyword in col_lower for keyword in ['gold', 'xp', 'cs', 'earned', 'spent']):
            categories['Economy'].append(col)
        elif any(keyword in col_lower for keyword in ['kill', 'death', 'assist', 'kda', 'damage', 'dpm']):
            categories['Combat'].append(col)
        elif any(keyword in col_lower for keyword in ['ward', 'vision', 'control']):
            categories['Vision'].append(col)
        elif any(keyword in col_lower for keyword in ['dragon', 'baron', 'herald', 'tower', 'inhibitor', 'elder']):
            categories['Objectives'].append(col)
        elif any(keyword in col_lower for keyword in ['result', 'win', 'length']):
            categories['Match Info'].append(col)
        else:
            categories['Other'].append(col)
    
    # Display categorized columns
    for category, columns in categories.items():
        if columns:
            print(f"\n🏷️ {category.upper()} ({len(columns)} columns):")
            for i, col in enumerate(sorted(columns), 1):
                print(f"   {i:2d}. {col}")
    
    # Show sample data for key columns
    print(f"\n📋 SAMPLE DATA PREVIEW:")
    print("=" * 60)
    
    # Show a few key columns with sample values
    key_columns = ['position', 'league', 'teamname', 'champion', 'result']
    available_key_cols = [col for col in key_columns if col in sample_df.columns]
    
    if available_key_cols:
        sample_preview = sample_df[available_key_cols].head(10)
        print(sample_preview.to_string(index=False))
    
    # Analyze what's available for pre-match prediction
    print(f"\n🎯 PRE-MATCH PREDICTION FEATURES:")
    print("=" * 60)
    
    pre_match_features = {
        'Essential': [],
        'Champion Strategy': [],
        'Team Context': [],
        'Meta/Time': [],
        'Avoid (Post-Match)': []
    }
    
    for col in sample_df.columns:
        col_lower = col.lower()
        
        # Essential features
        if col in ['league', 'teamname', 'side', 'result', 'gameid']:
            pre_match_features['Essential'].append(col)
        
        # Champion strategy (available pre-match)
        elif any(keyword in col_lower for keyword in ['champion', 'ban', 'pick']):
            pre_match_features['Champion Strategy'].append(col)
        
        # Team/meta context
        elif any(keyword in col_lower for keyword in ['split', 'playoffs', 'patch', 'date']):
            pre_match_features['Meta/Time'].append(col)
        
        # Post-match data (avoid for prediction)
        elif any(keyword in col_lower for keyword in ['kill', 'death', 'damage', 'gold', 'cs', 'ward', 'dragon', 'baron', 'tower', 'length']):
            pre_match_features['Avoid (Post-Match)'].append(col)
        
        else:
            pre_match_features['Team Context'].append(col)
    
    for category, columns in pre_match_features.items():
        if columns:
            print(f"\n✅ {category} ({len(columns)} columns):")
            for col in sorted(columns)[:10]:  # Show first 10
                print(f"   • {col}")
            if len(columns) > 10:
                print(f"   ... and {len(columns) - 10} more")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS FOR CHAMPION-BASED PREDICTION:")
    print("=" * 60)
    print("✅ USE THESE FEATURES:")
    print("   🏆 Match Context: league, split, playoffs, patch, date")
    print("   👥 Teams: teamname, side")
    print("   🎮 Champions: champion, ban1-5, pick1-5 (if available)")
    print("   📅 Time: year, month, patch version")
    print("   🎯 Target: result (match outcome)")
    
    print("\n❌ AVOID THESE (POST-MATCH DATA):")
    print("   💰 Economy: gold, xp, cs, items")
    print("   ⚔️ Combat: kills, deaths, damage")
    print("   👁️ Vision: wards, vision score")
    print("   🐉 Objectives: dragons, barons, towers")
    print("   ⏱️ Game length: gamelength")
    
    return sample_df.columns.tolist(), categories

def show_champion_pick_data():
    """Analyze champion pick/ban structure specifically."""
    print(f"\n🎮 CHAMPION PICK/BAN ANALYSIS:")
    print("=" * 60)
    
    data_dir = "Dataset collection"
    latest_file = None
    
    for file in os.listdir(data_dir):
        if '2024' in file and 'LoL_esports_match_data' in file and 'processed' not in file:
            latest_file = os.path.join(data_dir, file)
            break
    
    if latest_file:
        # Load sample to check pick/ban structure
        sample_df = pd.read_csv(latest_file, nrows=500)
        
        # Find champion-related columns
        champion_cols = [col for col in sample_df.columns if any(keyword in col.lower() 
                        for keyword in ['champion', 'ban', 'pick'])]
        
        print(f"🔍 Champion-related columns found: {len(champion_cols)}")
        for col in sorted(champion_cols):
            print(f"   • {col}")
        
        # Check if we have individual player picks
        if 'position' in sample_df.columns:
            positions = sample_df['position'].unique()
            print(f"\n📍 Available positions: {', '.join(positions)}")
            
            # Show sample for each position
            for pos in positions[:3]:  # Show first 3 positions
                pos_data = sample_df[sample_df['position'] == pos]
                if 'champion' in pos_data.columns and len(pos_data) > 0:
                    sample_champs = pos_data['champion'].dropna().head(5).tolist()
                    print(f"   {pos}: {', '.join(sample_champs)}")

if __name__ == "__main__":
    columns, categories = analyze_original_dataset_columns()
    show_champion_pick_data() 