"""
🔧 COMPLETE TARGET LEAGUES DATA EXTRACTION - PRE-MATCH PREDICTION
Extract ALL matches from top LoL leagues with ONLY pre-match information (2014-2024)

🚨 CRITICAL: NO DATA LEAKAGE - Only picks, bans, and meta information allowed!

Target Leagues & Tournaments:
- LPL (China)
- LCK (Korea) 
- LEC (Europe, current)
- EU LCS (Europe, historical)
- LCS (North America, current)
- NA LCS (North America, historical)
- WLDs (Worlds Championship)
- MSI (Mid-Season Invitational)

Pre-Match Information Only:
- Match metadata: gameid, league, date, patch, split, playoffs, year, game
- Team info: teamname/team, side
- Champion picks: top_champion, jng_champion, mid_champion, bot_champion, sup_champion
- Champion bans: ban1, ban2, ban3, ban4, ban5
- Target: result (0/1)

❌ EXCLUDED: All post-match statistics (kills, deaths, gold, damage, vision, etc.)
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class CompleteTargetDatasetCreator:
    """Extract comprehensive dataset from all major LoL leagues."""
    
    def __init__(self, data_dir="Data"):
        self.data_dir = Path(data_dir)
        
        # Define target leagues (Big 4 + historical names + major tournaments)
        self.target_leagues = {
            'current': ['LPL', 'LCK', 'LEC', 'LCS'],
            'historical': ['EU LCS', 'NA LCS'],
            'tournaments': ['WLDs', 'MSI'],  # Major international tournaments
            'all': ['LPL', 'LCK', 'LEC', 'LCS', 'EU LCS', 'NA LCS', 'WLDs', 'MSI']
        }
        
        # Track statistics
        self.stats = {
            'files_processed': 0,
            'total_rows_read': 0,
            'target_league_matches': 0,
            'final_matches': 0,
            'leagues_found': set(),
            'years_processed': [],
            'data_quality': {}
        }
    
    def get_available_files(self):
        """Get all available yearly LoL data files."""
        pattern = "*_LoL_esports_match_data_from_OraclesElixir.csv"
        files = list(self.data_dir.glob(pattern))
        
        # Sort by year
        files.sort(key=lambda x: self.extract_year_from_filename(x.name))
        
        print(f"📂 Found {len(files)} yearly data files:")
        for file in files:
            year = self.extract_year_from_filename(file.name)
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"   📅 {year}: {file.name} ({size_mb:.1f}MB)")
        
        return files
    
    def extract_year_from_filename(self, filename):
        """Extract year from filename."""
        try:
            return int(filename.split('_')[0])
        except:
            return 0
    
    def load_and_filter_year(self, file_path):
        """Load and filter a single year's data for PRE-MATCH PREDICTION."""
        year = self.extract_year_from_filename(file_path.name)
        print(f"\n📊 Processing {year} data...")
        
        try:
            # Load data
            df = pd.read_csv(file_path, low_memory=False)
            original_size = len(df)
            self.stats['total_rows_read'] += original_size
            
            print(f"   📥 Loaded: {original_size:,} total matches")
            
            # 🚨 CRITICAL: Only extract PRE-MATCH information (no data leakage!)
            pre_match_columns = [
                'gameid', 'league', 'date', 'patch', 'split', 'playoffs', 'year', 'game',
                'teamname', 'team', 'side', 'result',  # Core info
                'champion',  # Individual champion (will map to positions)
                'position',  # Position info
                'ban1', 'ban2', 'ban3', 'ban4', 'ban5',  # Bans
                # Champion picks by position (if available)
                'top_champion', 'jng_champion', 'mid_champion', 'bot_champion', 'sup_champion'
            ]
            
            # Filter to available pre-match columns only
            available_cols = [col for col in pre_match_columns if col in df.columns]
            print(f"   🔒 Pre-match columns found: {len(available_cols)}/{len(pre_match_columns)}")
            print(f"   ✅ Using: {', '.join(available_cols)}")
            
            # Filter columns first (prevent data leakage)
            df_prematch = df[available_cols].copy()
            
            # Check available leagues
            if 'league' in df_prematch.columns:
                available_leagues = df_prematch['league'].value_counts()
                print(f"   🏆 Available leagues: {list(available_leagues.index)}")
                self.stats['leagues_found'].update(available_leagues.index)
            else:
                print("   ⚠️ No 'league' column found")
                return pd.DataFrame()
            
            # Filter for target leagues
            target_mask = df_prematch['league'].isin(self.target_leagues['all'])
            df_filtered = df_prematch[target_mask].copy()
            target_matches = len(df_filtered)
            
            print(f"   🎯 Target league matches: {target_matches:,}")
            print(f"   📈 Extraction rate: {(target_matches/original_size)*100:.1f}%")
            
            # Add year column if missing
            if 'year' not in df_filtered.columns:
                df_filtered['year'] = year
            
            self.stats['target_league_matches'] += target_matches
            self.stats['years_processed'].append(year)
            
            # Show league distribution for this year
            if not df_filtered.empty:
                league_dist = df_filtered['league'].value_counts()
                print(f"   📊 League distribution:")
                for league, count in league_dist.items():
                    print(f"      {league}: {count:,} matches")
            
            # 🔒 VERIFICATION: Ensure no post-match data leakage
            post_match_indicators = ['kills', 'deaths', 'assists', 'gold', 'damage', 'cs', 'vision']
            leakage_cols = [col for col in df_filtered.columns 
                           if any(indicator in col.lower() for indicator in post_match_indicators)]
            
            if leakage_cols:
                print(f"   ⚠️ WARNING: Potential data leakage columns detected: {leakage_cols}")
                print(f"   🔒 These will be excluded from the final dataset")
                # Remove leakage columns
                df_filtered = df_filtered.drop(columns=leakage_cols)
            else:
                print(f"   ✅ No data leakage detected - only pre-match information included")
            
            return df_filtered
            
        except Exception as e:
            print(f"   ❌ Error processing {file_path.name}: {str(e)}")
            return pd.DataFrame()
    
    def combine_all_years(self):
        """Combine data from all available years."""
        print("🚀 EXTRACTING COMPLETE TARGET LEAGUES DATASET")
        print("=" * 80)
        print(f"🎯 Target Leagues: {', '.join(self.target_leagues['all'])}")
        print(f"   📊 Major Regional Leagues: {', '.join(self.target_leagues['current'])}")
        print(f"   📅 Historical Names: {', '.join(self.target_leagues['historical'])}")
        print(f"   🏆 International Tournaments: {', '.join(self.target_leagues['tournaments'])}")
        
        # Get all files
        files = self.get_available_files()
        
        if not files:
            raise FileNotFoundError("No LoL data files found in Data/ directory")
        
        # Process each year
        all_dataframes = []
        
        for file_path in files:
            df_year = self.load_and_filter_year(file_path)
            if not df_year.empty:
                all_dataframes.append(df_year)
                self.stats['files_processed'] += 1
        
        if not all_dataframes:
            raise ValueError("No target league data found in any files")
        
        # Combine all years
        print(f"\n🔄 Combining data from {len(all_dataframes)} years...")
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        print(f"✅ Combined dataset: {len(combined_df):,} matches")
        
        return combined_df
    
    def clean_and_validate(self, df):
        """Clean and validate the combined dataset."""
        print(f"\n🧹 CLEANING AND VALIDATING DATASET")
        print("=" * 50)
        
        original_size = len(df)
        print(f"📊 Starting size: {original_size:,} rows (individual player records)")
        
        # Check available columns and adapt
        print(f"   🔍 Available columns: {list(df.columns)[:10]}...")
        
        # 🔥 CRITICAL: Standardize league names BEFORE transformation
        print(f"   🏆 Standardizing league names...")
        league_mapping = {
            'NA LCS': 'LCS',    # Historical name → Current name
            'EU LCS': 'LEC',    # Historical name → Current name
            'LCK': 'LCK',       # Keep as is
            'LPL': 'LPL',       # Keep as is
            'WLDs': 'WLDs',     # Worlds Championship
            'MSI': 'MSI',       # Mid-Season Invitational
            'Worlds': 'WLDs',   # Alternative name for Worlds
        }
        
        # Apply league mapping
        df['league'] = df['league'].map(league_mapping).fillna(df['league'])
        
        # Show league distribution before transformation
        print(f"   📊 League distribution after standardization:")
        league_dist_before = df['league'].value_counts()
        for league, count in league_dist_before.items():
            if league in ['LCS', 'LEC', 'LCK', 'LPL']:
                print(f"      {league}: {count:,} player records")
        
        # 🔥 CRITICAL: Transform Oracle's Elixir format to team-match format
        print(f"   🔄 Transforming player-rows to team-match format...")
        
        # Group by game and team to create one row per team per match
        group_cols = ['gameid']
        team_col = 'teamname' if 'teamname' in df.columns else 'team'
        group_cols.append(team_col)
        
        print(f"   📊 Grouping by: {group_cols}")
        grouped = df.groupby(group_cols)
        
        team_matches = []
        valid_groups = 0
        invalid_groups = 0
        
        for (gameid, team), group in grouped:
            # Filter to only target leagues (including tournaments)
            if group.iloc[0]['league'] not in ['LCS', 'LEC', 'LCK', 'LPL', 'WLDs', 'MSI']:
                continue
                
            # Skip incomplete groups but be more lenient
            if len(group) < 4:  # Allow 4+ players instead of exactly 5
                invalid_groups += 1
                continue
            
            valid_groups += 1
            
            # Get first row for match metadata (same for all players)
            first_row = group.iloc[0]
            
            # Create team match record
            team_match = {
                'gameid': gameid,
                'league': first_row.get('league', 'Unknown'),
                'date': first_row.get('date'),
                'patch': first_row.get('patch', 'Unknown'),
                'split': first_row.get('split', 'Unknown'),
                'playoffs': first_row.get('playoffs', 0),
                'year': first_row.get('year', 2023),
                'game': first_row.get('game', 1),
                'team': team,
                'side': first_row.get('side', 'Blue'),
                'result': first_row.get('result', 0)
            }
            
            # Add bans (same for all players in team)
            for i in range(1, 6):
                ban_col = f'ban{i}'
                team_match[ban_col] = first_row.get(ban_col, 'NoBan')
            
            # Pivot champion positions to columns
            position_champion_map = {}
            for _, player in group.iterrows():
                position = player.get('position')
                champion = player.get('champion')
                
                if pd.notna(position) and pd.notna(champion) and position != 'team':
                    # Map positions to expected column names (more flexible)
                    position_mapping = {
                        'top': 'top_champion',
                        'jng': 'jng_champion', 
                        'jungle': 'jng_champion',
                        'jgl': 'jng_champion',
                        'mid': 'mid_champion',
                        'middle': 'mid_champion',
                        'bot': 'bot_champion',
                        'adc': 'bot_champion',
                        'carry': 'bot_champion',
                        'sup': 'sup_champion',
                        'support': 'sup_champion',
                        'supp': 'sup_champion'
                    }
                    
                    mapped_position = position_mapping.get(position.lower())
                    if mapped_position:
                        position_champion_map[mapped_position] = champion
            
            # Add champion columns
            required_positions = ['top_champion', 'jng_champion', 'mid_champion', 'bot_champion', 'sup_champion']
            for pos in required_positions:
                team_match[pos] = position_champion_map.get(pos, 'Unknown')
            
            # Be more lenient - allow up to 2 unknown champions
            unknown_champions = sum(1 for pos in required_positions if team_match[pos] == 'Unknown')
            if unknown_champions <= 2:  # More lenient threshold
                team_matches.append(team_match)
        
        # Convert to DataFrame
        df_transformed = pd.DataFrame(team_matches)
        
        print(f"   ✅ Transformation results:")
        print(f"      📊 Original: {original_size:,} player records")
        print(f"      📊 Valid groups: {valid_groups:,}")
        print(f"      📊 Invalid groups: {invalid_groups:,}")
        print(f"      📊 Transformed: {len(df_transformed):,} team matches")
        print(f"      📊 Reduction ratio: {len(df_transformed)/original_size:.3f}")
        print(f"      📊 Expected teams per match: {original_size/len(df_transformed) if len(df_transformed) > 0 else 0:.1f}")
        
        if len(df_transformed) == 0:
            print(f"   ❌ No valid team matches created!")
            return pd.DataFrame()
        
        # Continue with validation on transformed data
        df = df_transformed
        original_size = len(df)
        
        # 1. Handle missing critical columns with flexible naming
        critical_columns = ['league', 'result', 'team']
        
        # Check for missing critical data
        available_critical = [col for col in critical_columns if col in df.columns]
        missing_critical = df[available_critical].isnull().any(axis=1)
        
        if missing_critical.sum() > 0:
            print(f"   🔧 Removing {missing_critical.sum():,} matches with missing critical data")
            df = df[~missing_critical].copy()
        
        # 2. Ensure result column is binary (0/1)
        if 'result' in df.columns:
            df['result'] = pd.to_numeric(df['result'], errors='coerce')
            df = df[df['result'].isin([0, 1])].copy()
        
        # 3. Handle dates
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            # Remove matches with invalid dates
            df = df[df['date'].notna()].copy()
        
        # 4. Remove duplicates CAREFULLY
        before_dedup = len(df)
        # Only remove exact duplicates across all columns
        df = df.drop_duplicates(subset=['gameid', 'team'], keep='first')  # Keep first occurrence per game-team
        after_dedup = len(df)
        if before_dedup != after_dedup:
            print(f"   🔄 Removed {before_dedup - after_dedup:,} duplicate game-team combinations")
        
        # 5. Validate champion columns but don't remove data
        champion_cols = ['top_champion', 'jng_champion', 'mid_champion', 'bot_champion', 'sup_champion']
        print(f"   🏆 Champion column validation:")
        
        for col in champion_cols:
            if col in df.columns:
                unknown_count = (df[col] == 'Unknown').sum()
                valid_count = len(df) - unknown_count
                print(f"      {col}: {valid_count:,} valid, {unknown_count:,} unknown")
        
        # 6. Data quality statistics
        print(f"   📊 Data validation:")
        print(f"      Unique teams: {df['team'].nunique():,}")
        print(f"      Unique leagues: {df['league'].nunique()}")
        if 'date' in df.columns:
            print(f"      Date range: {df['date'].min()} to {df['date'].max()}")
        if 'year' in df.columns:
            years = sorted(df['year'].unique())
            print(f"      Years covered: {years}")
        
        # 7. Final league distribution
        print(f"\n   🏆 Final league distribution:")
        league_dist = df['league'].value_counts()
        total_matches = len(df)
        for league, count in league_dist.items():
            percentage = (count / total_matches) * 100
            print(f"      {league}: {count:,} matches ({percentage:.1f}%)")
        
        # 8. Sample the transformed data structure
        print(f"\n   📋 Sample team-match structure:")
        sample_cols = ['gameid', 'team', 'league', 'result', 'top_champion', 'jng_champion', 'mid_champion']
        available_sample_cols = [col for col in sample_cols if col in df.columns]
        if len(df) >= 3:
            print(df[available_sample_cols].head(3).to_string())
        
        final_size = len(df)
        self.stats['final_matches'] = final_size
        
        print(f"\n   ✅ Final dataset: {final_size:,} team matches")
        
        # 🔥 Data quality assessment
        if final_size > 40000:
            print(f"   🎉 EXCELLENT: {final_size:,} matches > 40K target!")
        elif final_size > 30000:
            print(f"   ✅ GOOD: {final_size:,} matches > 30K")
        else:
            print(f"   ⚠️ BELOW EXPECTATION: {final_size:,} matches < 30K")
        
        return df
    
    def save_dataset(self, df, output_path="Data/complete_target_leagues_dataset.csv"):
        """Save the complete dataset with metadata."""
        print(f"\n💾 SAVING COMPLETE DATASET")
        print("=" * 40)
        
        # Save main dataset
        df.to_csv(output_path, index=False)
        file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        
        print(f"   ✅ Saved: {output_path}")
        print(f"   📊 Size: {file_size_mb:.1f}MB")
        print(f"   📈 Rows: {len(df):,}")
        print(f"   📊 Columns: {len(df.columns)}")
        
        # Create metadata file
        metadata = {
            'creation_date': pd.Timestamp.now().isoformat(),
            'total_matches': len(df),
            'leagues_included': list(df['league'].unique()),
            'years_covered': sorted(df['year'].unique()),
            'date_range': {
                'start': df['date'].min().isoformat(),
                'end': df['date'].max().isoformat()
            },
            'processing_stats': self.stats,
            'league_distribution': df['league'].value_counts().to_dict(),
            'columns': list(df.columns)
        }
        
        metadata_path = output_path.replace('.csv', '_metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"   📋 Metadata: {metadata_path}")
        
        return output_path, metadata_path
    
    def create_complete_dataset(self, output_path="Data/complete_target_leagues_dataset.csv"):
        """Main method to create the complete target leagues dataset."""
        
        # Combine all years
        combined_df = self.combine_all_years()
        
        # Clean and validate
        clean_df = self.clean_and_validate(combined_df)
        
        # Save dataset
        dataset_path, metadata_path = self.save_dataset(clean_df, output_path)
        
        # Final summary
        print(f"\n🎉 COMPLETE TARGET DATASET CREATION FINISHED!")
        print("=" * 60)
        print(f"📊 Final Statistics:")
        print(f"   📁 Files processed: {self.stats['files_processed']}")
        print(f"   📈 Total rows read: {self.stats['total_rows_read']:,}")
        print(f"   🎯 Target matches extracted: {self.stats['target_league_matches']:,}")
        print(f"   ✅ Final clean matches: {self.stats['final_matches']:,}")
        print(f"   🏆 Leagues: {', '.join(sorted(clean_df['league'].unique()))}")
        print(f"   📅 Years: {min(self.stats['years_processed'])}-{max(self.stats['years_processed'])}")
        
        extraction_rate = (self.stats['final_matches'] / self.stats['total_rows_read']) * 100
        print(f"   📊 Overall extraction rate: {extraction_rate:.1f}%")
        
        return clean_df, dataset_path, metadata_path

def main():
    """Main execution function."""
    print("🔧 COMPLETE TARGET LEAGUES DATA EXTRACTION - PRE-MATCH PREDICTION")
    print("🚨 CRITICAL: Only pre-match information (picks & bans) - NO DATA LEAKAGE!")
    print("Creating comprehensive dataset from ALL available years")
    print("=" * 80)
    
    # Create extractor
    extractor = CompleteTargetDatasetCreator()
    
    # Create complete dataset
    df, dataset_path, metadata_path = extractor.create_complete_dataset()
    
    # Verify against current dataset
    current_path = "Data/target_leagues_dataset.csv"
    if os.path.exists(current_path):
        current_df = pd.read_csv(current_path)
        print(f"\n📊 COMPARISON WITH CURRENT DATASET:")
        print(f"   📉 Current: {len(current_df):,} matches")
        print(f"   📈 New: {len(df):,} matches")
        print(f"   🚀 Improvement: {len(df) - len(current_df):+,} matches ({((len(df)/len(current_df))-1)*100:.1f}% increase)")
        print(f"   🔒 Data Safety: Pre-match information ONLY (no leakage)")
    
    return df, extractor

if __name__ == "__main__":
    df, extractor = main() 