import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FocusedFeatureExtractor:
    """
    Extracts only the essential features for pre-match prediction:
    - Essential Match Context
    - Champion bans and picks  
    - Patch and time info
    - Team and player info
    """
    
    def __init__(self, data_dir="Dataset collection"):
        self.data_dir = data_dir
        self.feature_columns = self._define_feature_columns()
        
    def _define_feature_columns(self):
        """Define the specific columns we want to extract."""
        return {
            # Essential Match Context
            'match_context': [
                'gameid', 'league', 'result', 'side', 'game'
            ],
            
            # Champion Strategy (bans and picks)
            'champion_strategy': [
                'champion', 'ban1', 'ban2', 'ban3', 'ban4', 'ban5',
                'pick1', 'pick2', 'pick3', 'pick4', 'pick5'
            ],
            
            # Patch and Time Info
            'patch_time': [
                'patch', 'date', 'split', 'playoffs', 'year'
            ],
            
            # Team and Player Info
            'team_player': [
                'teamname', 'teamid', 'playerid', 'playername', 'position'
            ]
        }
    
    def load_and_extract_features(self):
        """Load all datasets and extract only the focused features."""
        print("🎯 FOCUSED FEATURE EXTRACTION")
        print("=" * 50)
        
        # Get all CSV files
        csv_files = []
        for file in os.listdir(self.data_dir):
            if file.endswith('.csv') and 'LoL_esports_match_data' in file and 'processed' not in file:
                csv_files.append(os.path.join(self.data_dir, file))
        
        csv_files = sorted(csv_files)
        print(f"📁 Found {len(csv_files)} dataset files")
        
        all_data = []
        
        for file_path in csv_files:
            print(f"📖 Processing: {os.path.basename(file_path)}")
            
            # Load the dataset
            df = pd.read_csv(file_path)
            
            # Get all required columns that exist in this dataset
            all_required_cols = []
            for category, cols in self.feature_columns.items():
                all_required_cols.extend(cols)
            
            # Find which columns actually exist
            existing_cols = [col for col in all_required_cols if col in df.columns]
            missing_cols = [col for col in all_required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"   ⚠️  Missing columns: {missing_cols}")
            
            # Extract only the existing required columns
            focused_df = df[existing_cols].copy()
            
            # Add year if not present
            if 'year' not in focused_df.columns and 'date' in focused_df.columns:
                focused_df['year'] = pd.to_datetime(focused_df['date']).dt.year
            
            print(f"   ✅ Extracted {len(existing_cols)} features, {len(focused_df)} rows")
            all_data.append(focused_df)
        
        # Combine all datasets
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\n📊 COMBINED DATASET:")
        print(f"   Total rows: {len(combined_df):,}")
        print(f"   Total features: {len(combined_df.columns)}")
        
        return combined_df
    
    def analyze_extracted_features(self, df):
        """Analyze the extracted features."""
        print(f"\n🔍 FEATURE ANALYSIS:")
        print("=" * 50)
        
        # Show features by category
        for category, expected_cols in self.feature_columns.items():
            existing_cols = [col for col in expected_cols if col in df.columns]
            print(f"\n📋 {category.upper().replace('_', ' ')} ({len(existing_cols)} features):")
            for col in existing_cols:
                unique_count = df[col].nunique()
                null_count = df[col].isnull().sum()
                print(f"   • {col}: {unique_count:,} unique values, {null_count:,} nulls")
        
        # Show data distribution
        print(f"\n📈 DATA DISTRIBUTION:")
        print("=" * 50)
        
        if 'league' in df.columns:
            print("🏆 Leagues:")
            league_counts = df['league'].value_counts().head(10)
            for league, count in league_counts.items():
                print(f"   • {league}: {count:,} records")
        
        if 'position' in df.columns:
            print("\n📍 Positions:")
            pos_counts = df['position'].value_counts()
            for pos, count in pos_counts.items():
                print(f"   • {pos}: {count:,} records")
        
        if 'year' in df.columns:
            print("\n📅 Years:")
            year_counts = df['year'].value_counts().sort_index()
            for year, count in year_counts.items():
                print(f"   • {year}: {count:,} records")
        
        return df
    
    def create_match_level_features(self, df):
        """Create match-level features from player-level data."""
        print(f"\n🔄 CREATING MATCH-LEVEL FEATURES:")
        print("=" * 50)
        
        # Group by match (gameid) to create match-level records
        match_features = []
        
        for gameid, match_data in df.groupby('gameid'):
            # Skip if no data
            if len(match_data) == 0:
                continue
                
            # Get match context (same for all players in a match)
            match_context = match_data.iloc[0]
            
            # Create base match record
            match_record = {
                'gameid': gameid,
                'league': match_context.get('league'),
                'date': match_context.get('date'),
                'patch': match_context.get('patch'),
                'split': match_context.get('split'),
                'playoffs': match_context.get('playoffs'),
                'year': match_context.get('year'),
                'game': match_context.get('game')
            }
            
            # Get team information
            teams = match_data['teamname'].dropna().unique()
            if len(teams) >= 2:
                # Take first two teams
                team1, team2 = teams[0], teams[1]
                
                # Create records for both teams
                for team in [team1, team2]:
                    team_data = match_data[match_data['teamname'] == team]
                    
                    # Skip if no team data
                    if len(team_data) == 0:
                        continue
                    
                    team_record = match_record.copy()
                    team_record.update({
                        'team': team,
                        'side': team_data.iloc[0].get('side') if len(team_data) > 0 else None,
                        'result': team_data.iloc[0].get('result') if len(team_data) > 0 else None
                    })
                    
                    # Add champion picks by position
                    for _, player in team_data.iterrows():
                        pos = player.get('position')
                        champion = player.get('champion')
                        if pos and pos != 'team' and pd.notna(champion):
                            team_record[f'{pos}_champion'] = champion
                    
                    # Add bans (same for whole team)
                    for i in range(1, 6):
                        ban_col = f'ban{i}'
                        if ban_col in team_data.columns and len(team_data) > 0:
                            ban_value = team_data[ban_col].iloc[0]
                            if pd.notna(ban_value):
                                team_record[ban_col] = ban_value
                    
                    match_features.append(team_record)
        
        match_df = pd.DataFrame(match_features)
        print(f"   ✅ Created {len(match_df)} match-level records")
        print(f"   📊 Features: {len(match_df.columns)}")
        
        return match_df
    
    def save_focused_dataset(self, df, filename="focused_features_dataset.csv"):
        """Save the focused dataset."""
        output_path = os.path.join(self.data_dir, filename)
        df.to_csv(output_path, index=False)
        print(f"\n💾 SAVED: {output_path}")
        print(f"   📊 {len(df)} rows × {len(df.columns)} columns")
        return output_path

def main():
    """Main execution function."""
    print("🚀 STARTING FOCUSED FEATURE EXTRACTION")
    print("=" * 60)
    
    # Initialize extractor
    extractor = FocusedFeatureExtractor()
    
    # Extract features
    df = extractor.load_and_extract_features()
    
    # Analyze features
    df = extractor.analyze_extracted_features(df)
    
    # Create match-level features
    match_df = extractor.create_match_level_features(df)
    
    # Save both datasets
    player_level_path = extractor.save_focused_dataset(df, "focused_features_player_level.csv")
    match_level_path = extractor.save_focused_dataset(match_df, "focused_features_match_level.csv")
    
    print(f"\n🎉 EXTRACTION COMPLETE!")
    print(f"📁 Player-level data: {os.path.basename(player_level_path)}")
    print(f"📁 Match-level data: {os.path.basename(match_level_path)}")
    
    # Show sample of match-level data
    print(f"\n📋 SAMPLE MATCH-LEVEL DATA:")
    print("=" * 60)
    if len(match_df) > 0:
        sample_cols = ['gameid', 'league', 'team', 'side', 'result', 'top_champion', 'mid_champion']
        available_cols = [col for col in sample_cols if col in match_df.columns]
        print(match_df[available_cols].head(10).to_string(index=False))
    
    return df, match_df

if __name__ == "__main__":
    player_df, match_df = main() 