import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class Top4LeaguesAnalyzer:
    """
    Analyzer for Oracle's Elixir LoL esports dataset focusing on TOP 4 leagues:
    - LCK (Korea)
    - LPL (China) 
    - LEC/EU LCS (Europe)
    - LCS/NA LCS (North America)
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.files = self._get_csv_files()
        self.combined_data = None
        
        # Define the top 4 leagues with all possible naming variations
        self.top_4_leagues = [
            'LCK',           # Korea
            'LPL',           # China
            'LEC',           # Europe (current)
            'EU LCS',        # Europe (old name)
            'EULCS',         # Europe (alternative)
            'LCS',           # North America (current)
            'NA LCS',        # North America (old name)
            'NALCS'          # North America (alternative)
        ]
        
    def _get_csv_files(self) -> List[str]:
        """Get all CSV files in the directory."""
        files = []
        for file in os.listdir(self.data_dir):
            if file.endswith('.csv') and 'LoL_esports_match_data' in file:
                files.append(os.path.join(self.data_dir, file))
        return sorted(files)
    
    def analyze_dataset_overview(self) -> Dict:
        """Get overview of all files without loading full data."""
        overview = {
            'total_files': len(self.files),
            'file_sizes': {},
            'estimated_total_size': 0,
            'years_covered': []
        }
        
        print("🏆 TOP 4 LEAGUES DATASET OVERVIEW")
        print("=" * 50)
        print("🎯 Target Leagues: LCK, LPL, LEC/EU LCS, LCS/NA LCS")
        print("=" * 50)
        
        total_size_mb = 0
        for file_path in self.files:
            filename = os.path.basename(file_path)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            total_size_mb += size_mb
            
            # Extract year from filename
            year = filename.split('_')[0]
            overview['years_covered'].append(year)
            overview['file_sizes'][year] = f"{size_mb:.1f}MB"
            
            print(f"📅 {year}: {size_mb:.1f}MB")
        
        overview['estimated_total_size'] = f"{total_size_mb:.1f}MB"
        overview['years_covered'] = sorted(overview['years_covered'])
        
        print(f"\n🎯 TOTAL: {total_size_mb:.1f}MB across {len(self.files)} years")
        print(f"📈 Years: {', '.join(overview['years_covered'])}")
        
        return overview
    
    def load_sample_data(self, year: str = "2024", sample_size: int = 2000) -> pd.DataFrame:
        """Load a sample of data to understand structure."""
        file_path = None
        for file in self.files:
            if year in file:
                file_path = file
                break
        
        if not file_path:
            print(f"❌ No file found for year {year}")
            return pd.DataFrame()
        
        print(f"📖 Loading sample from {year} data...")
        
        # Load sample
        sample_df = pd.read_csv(file_path, nrows=sample_size)
        
        print(f"✅ Loaded {len(sample_df)} rows with {len(sample_df.columns)} columns")
        
        # Show available leagues in sample
        if 'league' in sample_df.columns:
            available_leagues = sample_df['league'].value_counts()
            print(f"\n🏆 Available leagues in {year} sample:")
            for league, count in available_leagues.head(10).items():
                is_top4 = "🎯" if league in self.top_4_leagues else "  "
                print(f"   {is_top4} {league}: {count} games")
        
        return sample_df
    
    def analyze_data_structure(self, sample_df: pd.DataFrame) -> Dict:
        """Analyze the structure of the dataset."""
        if sample_df.empty:
            return {}
        
        print("\n🔍 DATA STRUCTURE ANALYSIS")
        print("=" * 50)
        
        structure = {
            'total_columns': len(sample_df.columns),
            'column_types': sample_df.dtypes.value_counts().to_dict(),
            'missing_data': sample_df.isnull().sum().sum(),
            'unique_leagues': sample_df['league'].nunique() if 'league' in sample_df.columns else 0,
            'unique_teams': sample_df['teamname'].nunique() if 'teamname' in sample_df.columns else 0,
            'unique_players': sample_df['playername'].nunique() if 'playername' in sample_df.columns else 0,
            'positions': sample_df['position'].unique().tolist() if 'position' in sample_df.columns else []
        }
        
        print(f"📊 Columns: {structure['total_columns']}")
        print(f"🏆 Leagues: {structure['unique_leagues']}")
        print(f"👥 Teams: {structure['unique_teams']}")
        print(f"🎮 Players: {structure['unique_players']}")
        print(f"📍 Positions: {structure['positions']}")
        
        # Show key columns for champion-based prediction
        champion_columns = [col for col in sample_df.columns if any(keyword in col.lower() 
                          for keyword in ['champion', 'ban', 'pick', 'team', 'result', 'side'])]
        
        print(f"\n🎯 Champion-related columns ({len(champion_columns)}):")
        for col in champion_columns[:10]:  # Show first 10
            print(f"   • {col}")
        if len(champion_columns) > 10:
            print(f"   ... and {len(champion_columns) - 10} more")
        
        return structure
    
    def load_and_combine_top4_leagues(self, years: List[str]) -> pd.DataFrame:
        """Load and combine data from top 4 leagues only."""
        print(f"\n🚀 LOADING TOP 4 LEAGUES: {', '.join(years)}")
        print("🎯 Filtering for: LCK, LPL, LEC/EU LCS, LCS/NA LCS")
        print("=" * 50)
        
        dataframes = []
        total_rows = 0
        league_counts = {}
        
        for year in years:
            file_path = None
            for file in self.files:
                if year in file:
                    file_path = file
                    break
            
            if not file_path:
                print(f"❌ No file found for year {year}")
                continue
            
            print(f"📖 Loading {year}...")
            
            try:
                # Load data
                df = pd.read_csv(file_path, low_memory=False)
                original_size = len(df)
                
                # Filter for top 4 leagues only
                if 'league' in df.columns:
                    df_filtered = df[df['league'].isin(self.top_4_leagues)].copy()
                    
                    # Count leagues for this year
                    year_league_counts = df_filtered['league'].value_counts()
                    for league, count in year_league_counts.items():
                        if league not in league_counts:
                            league_counts[league] = 0
                        league_counts[league] += count
                    
                    print(f"   📊 {original_size:,} → {len(df_filtered):,} rows (Top 4 leagues only)")
                    print(f"   🏆 Leagues found: {', '.join(year_league_counts.index.tolist())}")
                    
                    df_filtered['data_year'] = year
                    dataframes.append(df_filtered)
                    total_rows += len(df_filtered)
                else:
                    print(f"   ❌ No 'league' column found in {year}")
                
            except Exception as e:
                print(f"❌ Error loading {year}: {e}")
                continue
        
        if not dataframes:
            print("❌ No data loaded!")
            return pd.DataFrame()
        
        print(f"\n🔄 Combining {len(dataframes)} datasets...")
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        print(f"✅ COMBINED TOP 4 LEAGUES DATASET: {len(combined_df):,} total rows")
        
        # Show final league distribution
        print(f"\n🏆 FINAL LEAGUE DISTRIBUTION:")
        for league, count in sorted(league_counts.items()):
            print(f"   🎯 {league}: {count:,} games")
        
        self.combined_data = combined_df
        return combined_df
    
    def create_champion_prediction_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a dataset optimized for champion-based match prediction."""
        print(f"\n🎮 CREATING CHAMPION PREDICTION DATASET")
        print("=" * 50)
        
        # Focus on team-level data (not individual player stats)
        if 'position' in df.columns:
            team_data = df[df['position'] == 'team'].copy()
            print(f"📊 Team-level data: {len(team_data):,} rows")
        else:
            team_data = df.copy()
        
        # Create match-level dataset (one row per match, not per team)
        matches = []
        
        print("🔄 Processing matches...")
        
        processed_count = 0
        for game_id, game_data in team_data.groupby('gameid'):
            if len(game_data) != 2:  # Should have exactly 2 teams
                continue
            
            # Sort by side (Blue first, then Red)
            teams = game_data.sort_values('side') if 'side' in game_data.columns else game_data
            
            if len(teams) == 2:
                blue_team = teams.iloc[0]
                red_team = teams.iloc[1]
                
                match_data = {
                    'gameid': game_id,
                    'league': blue_team.get('league', ''),
                    'year': blue_team.get('year', ''),
                    'split': blue_team.get('split', ''),
                    'playoffs': blue_team.get('playoffs', 0),
                    'date': blue_team.get('date', ''),
                    'patch': blue_team.get('patch', ''),
                    'gamelength': blue_team.get('gamelength', 0),
                    
                    # Blue team info
                    'blue_team': blue_team.get('teamname', ''),
                    'blue_side': blue_team.get('side', ''),
                    'blue_result': blue_team.get('result', 0),
                    
                    # Red team info
                    'red_team': red_team.get('teamname', ''),
                    'red_side': red_team.get('side', ''),
                    'red_result': red_team.get('result', 0),
                    
                    # Match outcome (1 if blue wins, 0 if red wins)
                    'blue_wins': 1 if blue_team.get('result', 0) == 1 else 0
                }
                
                # Add ban information if available
                for i in range(1, 6):
                    ban_col = f'ban{i}'
                    if ban_col in blue_team.index:
                        match_data[f'blue_{ban_col}'] = blue_team.get(ban_col, '')
                        match_data[f'red_{ban_col}'] = red_team.get(ban_col, '')
                
                matches.append(match_data)
                processed_count += 1
                
                # Progress indicator
                if processed_count % 1000 == 0:
                    print(f"   Processed {processed_count:,} matches...")
        
        match_df = pd.DataFrame(matches)
        
        print(f"✅ Created match dataset: {len(match_df):,} matches")
        
        return match_df
    
    def get_league_statistics(self, df: pd.DataFrame) -> Dict:
        """Get detailed statistics by league."""
        if 'league' not in df.columns:
            return {}
        
        print(f"\n🏆 TOP 4 LEAGUES DETAILED STATISTICS")
        print("=" * 50)
        
        league_stats = {}
        
        for league in self.top_4_leagues:
            if league in df['league'].values:
                league_data = df[df['league'] == league]
                
                stats = {
                    'total_games': len(league_data),
                    'unique_teams': league_data['teamname'].nunique() if 'teamname' in league_data.columns else 0,
                    'unique_players': league_data['playername'].nunique() if 'playername' in league_data.columns else 0,
                    'date_range': {
                        'start': league_data['date'].min() if 'date' in league_data.columns else '',
                        'end': league_data['date'].max() if 'date' in league_data.columns else ''
                    },
                    'years': sorted(league_data['data_year'].unique().tolist()) if 'data_year' in league_data.columns else []
                }
                
                league_stats[league] = stats
                print(f"🎯 {league}:")
                print(f"   📊 Games: {stats['total_games']:,}")
                print(f"   👥 Teams: {stats['unique_teams']}")
                print(f"   🎮 Players: {stats['unique_players']}")
                print(f"   📅 Years: {', '.join(map(str, stats['years']))}")
                print()
        
        return league_stats
    
    def save_processed_data(self, match_df: pd.DataFrame, filename: str = "top4_leagues_processed.csv"):
        """Save the processed dataset."""
        output_path = os.path.join(self.data_dir, filename)
        match_df.to_csv(output_path, index=False)
        
        print(f"\n💾 SAVED TOP 4 LEAGUES DATASET")
        print(f"📁 File: {output_path}")
        print(f"📊 Size: {len(match_df):,} matches")
        print(f"🎯 Columns: {len(match_df.columns)}")
        print(f"🏆 Leagues: {', '.join(match_df['league'].unique())}")
        
        # Show size comparison
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"💾 File size: {file_size_mb:.1f}MB")
        
        return output_path

def main():
    """Main analysis function for Top 4 leagues."""
    
    # Set the path to your dataset directory
    data_dir = "Dataset collection"
    
    # Initialize analyzer
    analyzer = Top4LeaguesAnalyzer(data_dir)
    
    # 1. Get dataset overview
    overview = analyzer.analyze_dataset_overview()
    
    # 2. Load sample to understand structure
    sample_df = analyzer.load_sample_data("2024", sample_size=3000)
    
    # 3. Analyze data structure
    structure = analyzer.analyze_data_structure(sample_df)
    
    # 4. Load and combine recent years focusing on Top 4 leagues
    recent_years = ["2024", "2023", "2022", "2021", "2020"]
    
    combined_df = analyzer.load_and_combine_top4_leagues(recent_years)
    
    if not combined_df.empty:
        # 5. Get detailed league statistics
        league_stats = analyzer.get_league_statistics(combined_df)
        
        # 6. Create champion prediction dataset
        match_df = analyzer.create_champion_prediction_dataset(combined_df)
        
        # 7. Save processed data
        if not match_df.empty:
            output_file = analyzer.save_processed_data(match_df)
            
            print(f"\n🎉 TOP 4 LEAGUES ANALYSIS COMPLETE!")
            print(f"📈 Your dataset is {len(match_df):,}x larger than your original 195 matches!")
            print(f"🏆 Focused on highest quality professional leagues!")
            print(f"🎯 Perfect for champion-based ML model training!")
            
            # Show improvement
            improvement_factor = len(match_df) / 195
            print(f"\n📊 DATASET IMPROVEMENT:")
            print(f"   Before: 195 matches")
            print(f"   After: {len(match_df):,} matches")
            print(f"   Improvement: {improvement_factor:.0f}x larger!")
        
        return match_df
    
    return pd.DataFrame()

if __name__ == "__main__":
    processed_data = main() 