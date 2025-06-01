import pandas as pd
import numpy as np
import os

def filter_target_leagues():
    """Filter the focused dataset to include only target leagues."""
    print("🎯 FILTERING DATASET TO TARGET LEAGUES")
    print("=" * 60)
    
    # Load the focused features dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    input_path = os.path.join(project_root, "data", "focused_features_match_level.csv")

    # Try alternative paths if not found
    if not os.path.exists(input_path):
        alternative_paths = [
            'Dataset collection/focused_features_match_level.csv',
            '../Dataset collection/focused_features_match_level.csv'
        ]
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                input_path = alt_path
                break
        else:
            raise FileNotFoundError(f"Could not find focused_features_match_level.csv")

    print(f"📂 Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"📊 Original dataset: {df.shape}")
    
    # Show current league distribution
    print(f"\n📋 CURRENT LEAGUE DISTRIBUTION:")
    league_counts = df['league'].value_counts()
    for league, count in league_counts.items():
        print(f"   • {league}: {count:,} matches")
    
    # Define target leagues
    target_leagues = [
        'CBLOL',     # Brazil
        'EU LCS',    # Europe (old name)
        'LEC',       # Europe (new name)
        'LCK',       # Korea
        'LCS',       # North America (new name)
        'LPL',       # China
        'MSI',       # Mid-Season Invitational
        'NA LCS',    # North America (old name)
        'WLDs'       # World Championship (as it appears in the dataset)
    ]
    
    print(f"\n🎯 TARGET LEAGUES:")
    for league in target_leagues:
        print(f"   • {league}")
    
    # Check which target leagues exist in the data
    existing_target_leagues = []
    missing_leagues = []
    
    for league in target_leagues:
        if league in df['league'].values:
            count = len(df[df['league'] == league])
            existing_target_leagues.append(league)
            print(f"   ✅ {league}: {count:,} matches found")
        else:
            missing_leagues.append(league)
            print(f"   ❌ {league}: Not found in dataset")
    
    # Check for similar league names (in case of naming variations)
    print(f"\n🔍 CHECKING FOR SIMILAR LEAGUE NAMES:")
    all_leagues = df['league'].unique()
    for missing in missing_leagues:
        similar = [league for league in all_leagues if missing.lower() in league.lower() or league.lower() in missing.lower()]
        if similar:
            print(f"   🔍 Similar to '{missing}': {similar}")
    
    # Check for Worlds variations
    worlds_variations = [league for league in all_leagues if 'world' in league.lower() or 'wcs' in league.lower()]
    if worlds_variations:
        print(f"   🌍 Possible Worlds leagues: {worlds_variations}")
    
    # Filter the dataset
    filtered_df = df[df['league'].isin(existing_target_leagues)].copy()
    
    print(f"\n📊 FILTERED DATASET:")
    print(f"   Original: {len(df):,} matches")
    print(f"   Filtered: {len(filtered_df):,} matches")
    print(f"   Reduction: {len(df) - len(filtered_df):,} matches removed")
    
    # Show filtered league distribution
    print(f"\n🏆 FILTERED LEAGUE DISTRIBUTION:")
    filtered_counts = filtered_df['league'].value_counts()
    total_filtered = 0
    for league, count in filtered_counts.items():
        print(f"   • {league}: {count:,} matches")
        total_filtered += count
    
    print(f"\n📈 TOTAL FILTERED MATCHES: {total_filtered:,}")
    
    # Check if we're close to the expected 41,296 matches
    expected_matches = 41296
    if abs(total_filtered - expected_matches) < 1000:
        print(f"✅ Close to expected {expected_matches:,} matches!")
    else:
        print(f"⚠️  Different from expected {expected_matches:,} matches")
        print(f"   Difference: {abs(total_filtered - expected_matches):,}")
    
    # Save the filtered dataset
    output_path = os.path.join(project_root, "data", "target_leagues_dataset.csv")
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    filtered_df.to_csv(output_path, index=False)
    print(f"\n💾 SAVED: {output_path}")
    print(f"📊 Final dataset: {filtered_df.shape}")
    print(f"🎯 Ready for model training!")
    
    # Show sample of filtered data
    print(f"\n📋 SAMPLE FILTERED DATA:")
    sample_cols = ['league', 'team', 'side', 'result', 'year', 'top_champion', 'mid_champion']
    available_cols = [col for col in sample_cols if col in filtered_df.columns]
    print(filtered_df[available_cols].head(10).to_string(index=False))
    
    return filtered_df

def suggest_league_corrections():
    """Suggest corrections for league names that might be variations."""
    print(f"\n🔧 LEAGUE NAME SUGGESTIONS:")
    print("=" * 60)
    
    df = pd.read_csv('Dataset collection/focused_features_match_level.csv')
    all_leagues = sorted(df['league'].unique())
    
    # Look for potential matches
    target_keywords = ['world', 'msi', 'lck', 'lpl', 'lcs', 'lec', 'cblol']
    
    print("🔍 Leagues containing target keywords:")
    for keyword in target_keywords:
        matching = [league for league in all_leagues if keyword.lower() in league.lower()]
        if matching:
            print(f"   {keyword.upper()}: {matching}")
    
    print(f"\n📋 ALL AVAILABLE LEAGUES ({len(all_leagues)}):")
    for i, league in enumerate(all_leagues, 1):
        count = len(df[df['league'] == league])
        print(f"   {i:2d}. {league}: {count:,} matches")

if __name__ == "__main__":
    # First show suggestions for league corrections
    suggest_league_corrections()
    
    print("\n" + "="*80 + "\n")
    
    # Then filter to target leagues
    filtered_df = filter_target_leagues() 