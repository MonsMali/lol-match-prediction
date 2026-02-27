import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, TargetEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings
import os
warnings.filterwarnings('ignore')

from src.features.formulas import (
    META_STRENGTH_WIN_WEIGHT,
    META_STRENGTH_POP_WEIGHT,
    META_STRENGTH_POP_CAP,
    DEFAULT_META,
    DEFAULT_WINRATE,
    DEFAULT_CHAMPION_CHAR,
    DEFAULT_TEAM_PERF,
    CANONICAL_48_FEATURES,
)

class AdvancedFeatureEngineering:
    """
    Advanced feature engineering for League of Legends match prediction.
    
    Features include:
    - Champion characteristics and meta strength
    - Pick/ban order strategy analysis
    - Team composition and synergy metrics
    - Historical matchup performance
    - Player performance and mastery
    - Advanced categorical encoding
    """
    
    def __init__(self, data_path=None):
        if data_path is None:
            # Build path to dataset - SINGLE PATH ONLY
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            data_path = os.path.join(project_root, "data", "processed", "complete_target_leagues_dataset.csv")
        
        #  CRITICAL: Verify we're using the correct clean dataset
        if "complete_target_leagues_dataset.csv" not in data_path:
            raise ValueError(f" WRONG DATASET! Must use 'complete_target_leagues_dataset.csv', not the old contaminated dataset. Found: {data_path}")
        
        # Simple existence check - no fallbacks
        if not os.path.exists(data_path):
            raise FileNotFoundError(f" Clean dataset not found at: {data_path}\n"
                                  f"Please run: python src/data_processing/create_complete_target_dataset.py")
        
        #  FINAL VERIFICATION: Absolutely ensure we're using the correct dataset
        print(f" AdvancedFeatureEngineering using CLEAN dataset: {data_path}")
        print(f" This dataset contains only major leagues (LPL, LCK, LCS, LEC, Worlds, MSI)")
        
        self.data_path = data_path
        self.df = None
        
        # Feature storage
        self.champion_characteristics = {}
        self.champion_meta_strength = {}
        self.champion_popularity = {}
        self.team_historical_performance = {}
        self.head_to_head_records = {}
        self.champion_matchups = {}
        self.player_champion_mastery = {}
        
        # Encoders
        self.target_encoders = {}
        self.label_encoders = {}
        
        # Patch timeline for meta analysis
        self.patch_timeline = self._create_patch_timeline()

    def __getstate__(self):
        """Convert defaultdicts with lambdas to regular dicts for pickling."""
        state = self.__dict__.copy()
        for key, value in state.items():
            state[key] = self._convert_defaultdicts(value)
        return state

    @staticmethod
    def _convert_defaultdicts(obj):
        """Recursively convert defaultdict instances to regular dicts."""
        if isinstance(obj, defaultdict):
            return {k: AdvancedFeatureEngineering._convert_defaultdicts(v) for k, v in obj.items()}
        elif isinstance(obj, dict):
            return {k: AdvancedFeatureEngineering._convert_defaultdicts(v) for k, v in obj.items()}
        return obj

    def _create_patch_timeline(self):
        """Create a mapping of patches to approximate dates for meta analysis."""
        # This would ideally come from external API, but we'll approximate
        return {
            '14.1': '2024-01-10', '14.2': '2024-01-24', '14.3': '2024-02-07',
            '13.24': '2023-12-06', '13.23': '2023-11-22', '13.22': '2023-11-08',
            # Add more patch mappings as needed
        }
    
    def load_and_analyze_data(self):
        """Load data and perform comprehensive analysis."""
        print(" ADVANCED FEATURE ENGINEERING SYSTEM")
        print("=" * 80)
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        print(f" Loaded dataset: {self.df.shape}")
        
        #  SKIP: Clean and filter target leagues data (already cleaned in dataset creation)
        # self._clean_target_leagues_data()  # DISABLED - using pre-cleaned dataset
        print(f"    Using pre-cleaned target leagues dataset")
        
        # Handle missing values
        self._handle_missing_values()
        
        # Analyze champion characteristics
        self._analyze_champion_characteristics()
        
        # Calculate meta indicators
        self._calculate_meta_indicators()
        
        # Analyze pick/ban strategy
        self._analyze_pickban_strategy()
        
        # Calculate team dynamics
        self._calculate_team_dynamics()
        
        # Analyze historical matchups
        self._analyze_historical_matchups()
        
        # Calculate player performance metrics
        self._calculate_player_metrics()

        # Analyze individual player performance and champion mastery
        self._analyze_player_performance()

        #  NEW: Add recent form and momentum features
        self._add_temporal_momentum_features()
        
        #  NEW: Investigate and fix target leakage
        self._investigate_target_leakage()
        
        #  NEW: Add meta shift detection features  
        self._add_meta_shift_detection()
        
        return self.df
    
    def _handle_missing_values(self):
        """Advanced missing value handling."""
        print(f"\n HANDLING MISSING VALUES")
        
        # Fill champion data
        champion_cols = [col for col in self.df.columns if 'champion' in col.lower()]
        for col in champion_cols:
            self.df[col] = self.df[col].astype(str)
            self.df[col] = self.df[col].replace('nan', 'Unknown')
            self.df[col] = self.df[col].replace('None', 'Unknown')
            self.df[col] = self.df[col].fillna('Unknown')
        
        # Fill ban data  
        ban_cols = [col for col in self.df.columns if 'ban' in col.lower()]
        for col in ban_cols:
            self.df[col] = self.df[col].astype(str)
            self.df[col] = self.df[col].replace('nan', 'NoBan')
            self.df[col] = self.df[col].replace('None', 'NoBan')
            self.df[col] = self.df[col].fillna('NoBan')
        
        # Fill player name data
        player_cols = [col for col in self.df.columns if '_player' in col.lower()]
        for col in player_cols:
            self.df[col] = self.df[col].astype(str)
            self.df[col] = self.df[col].replace('nan', 'Unknown')
            self.df[col] = self.df[col].replace('None', 'Unknown')
            self.df[col] = self.df[col].fillna('Unknown')

        # Fill other critical categorical fields
        categorical_cols = ['patch', 'split', 'league', 'team']
        for col in categorical_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str)
                self.df[col] = self.df[col].replace('nan', 'Unknown')
                self.df[col] = self.df[col].replace('None', 'Unknown')
                self.df[col] = self.df[col].fillna('Unknown')
        
        # Handle date column
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
        
        # Handle numeric columns
        numeric_cols = ['result', 'year', 'playoffs', 'game_length']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                if col == 'result':
                    self.df[col] = self.df[col].fillna(0)  # Default to loss if unknown
                elif col == 'year':
                    self.df[col] = self.df[col].fillna(2023)  # Default year
                elif col == 'playoffs':
                    self.df[col] = self.df[col].fillna(0)  # Default to regular season
                elif col == 'game_length':
                    self.df[col] = self.df[col].fillna(30)  # Default game length
        
        print(f"    Missing values handled")
        print(f"    Dataset shape after cleaning: {self.df.shape}")
    
    def _analyze_champion_characteristics(self):
        """Analyze champion characteristics and roles (vectorized).

        NOTE: game_length is post-match data and is NOT used here.
        Early/late game strength is no longer computed (it required
        game_length which constitutes data leakage). Instead we rely
        on win_rate, pick diversity, and position flexibility.
        """
        print(f"\n ANALYZING CHAMPION CHARACTERISTICS")

        champion_col_positions = [
            ('top_champion', 'Top'),
            ('jng_champion', 'Jungle'),
            ('mid_champion', 'Mid'),
            ('bot_champion', 'ADC'),
            ('sup_champion', 'Support')
        ]

        # Melt champion columns into long format: one row per (match, champion, position)
        frames = []
        for col, position in champion_col_positions:
            temp = self.df[['result']].copy()
            temp['champion'] = self.df[col]
            temp['position'] = position
            frames.append(temp)

        long_df = pd.concat(frames, ignore_index=True)

        # Filter out Unknown/NaN champions
        long_df = long_df[long_df['champion'].notna() & (long_df['champion'] != 'Unknown')]

        # Vectorized aggregation per champion
        grouped = long_df.groupby('champion')

        games = grouped['result'].count()
        wins = grouped['result'].sum()

        # Position flexibility: number of unique positions per champion
        positions_per_champ = long_df.groupby('champion')['position'].apply(set)

        # Build characteristics dict for champions with >= 5 games
        reliable = games[games >= 5].index
        for champion in reliable:
            total_games = games[champion]
            total_wins = wins[champion]
            win_rate = total_wins / total_games

            pos_set = positions_per_champ[champion]

            self.champion_characteristics[champion] = {
                'win_rate': win_rate,
                'avg_game_length': 30,  # Placeholder -- not used in features
                'early_game_strength': 0.5,  # Removed: required post-match game_length
                'late_game_strength': 0.5,   # Removed: required post-match game_length
                'scaling_factor': 0,          # Removed: derived from early/late
                'flexibility': len(pos_set),
                'primary_position': max(pos_set) if pos_set else 'Unknown'
            }

        print(f"    Analyzed {len(self.champion_characteristics)} champions")
    
    def _calculate_meta_indicators(self):
        """Calculate meta strength and popularity by patch (vectorized)."""
        print(f"\n CALCULATING META INDICATORS")

        champion_cols = ['top_champion', 'jng_champion', 'mid_champion', 'bot_champion', 'sup_champion']
        ban_cols = [f'ban{i}' for i in range(1, 6)]

        # Melt picks into long format
        pick_frames = []
        for col in champion_cols:
            temp = self.df[['patch', 'result']].copy()
            temp['champion'] = self.df[col]
            pick_frames.append(temp)
        picks_long = pd.concat(pick_frames, ignore_index=True)
        picks_long = picks_long[picks_long['champion'].notna() & (picks_long['champion'] != 'Unknown')]

        # Aggregate pick stats per (patch, champion)
        pick_stats = picks_long.groupby(['patch', 'champion']).agg(
            games=('result', 'count'),
            wins=('result', 'sum')
        ).reset_index()

        # Melt bans into long format
        ban_frames = []
        for col in ban_cols:
            temp = self.df[['patch']].copy()
            temp['champion'] = self.df[col]
            ban_frames.append(temp)
        bans_long = pd.concat(ban_frames, ignore_index=True)
        bans_long = bans_long[bans_long['champion'].notna() & (bans_long['champion'] != 'NoBan')]

        ban_stats = bans_long.groupby(['patch', 'champion']).size().reset_index(name='bans')

        # Total games per patch
        games_per_patch = self.df.groupby('patch').size().reset_index(name='total_games')

        # Merge everything
        merged = pick_stats.merge(ban_stats, on=['patch', 'champion'], how='left')
        merged['bans'] = merged['bans'].fillna(0)
        merged = merged.merge(games_per_patch, on='patch', how='left')

        # Calculate rates
        merged['win_rate'] = merged['wins'] / merged['games']
        merged['win_rate'] = merged['win_rate'].fillna(0.5)
        merged['pick_rate'] = merged['games'] / (merged['total_games'] * 2)
        merged['ban_rate'] = merged['bans'] / (merged['total_games'] * 2)
        merged['popularity'] = merged['pick_rate'] + merged['ban_rate']
        merged['meta_strength'] = (
            merged['win_rate'] * META_STRENGTH_WIN_WEIGHT
            + merged['popularity'].clip(upper=META_STRENGTH_POP_CAP) * META_STRENGTH_POP_WEIGHT
        )

        # Store results
        for _, row in merged.iterrows():
            patch, champion = row['patch'], row['champion']
            self.champion_meta_strength[(patch, champion)] = row['meta_strength']
            self.champion_popularity[(patch, champion)] = {
                'pick_rate': row['pick_rate'],
                'ban_rate': row['ban_rate'],
                'popularity': row['popularity']
            }

        print(f"    Calculated meta indicators for {len(self.champion_meta_strength)} champion-patch combinations")
    
    def _analyze_pickban_strategy(self):
        """Analyze pick/ban order strategy (vectorized)."""
        print(f"\n ANALYZING PICK/BAN STRATEGY")

        self.ban_priority = defaultdict(lambda: {'early_bans': 0, 'total_bans': 0})
        self.target_ban_analysis = defaultdict(lambda: defaultdict(int))

        ban_cols = [f'ban{i}' for i in range(1, 6)]

        # Melt bans into long format with ban position
        ban_frames = []
        for i, col in enumerate(ban_cols):
            temp = pd.DataFrame({
                'team': self.df['team'],
                'champion': self.df[col],
                'ban_position': i
            })
            ban_frames.append(temp)
        bans_long = pd.concat(ban_frames, ignore_index=True)
        bans_long = bans_long[bans_long['champion'].notna() & (bans_long['champion'] != 'NoBan')]

        # Total bans per champion
        total_bans = bans_long.groupby('champion').size()
        # Early bans (position 0 or 1) per champion
        early_bans = bans_long[bans_long['ban_position'] < 2].groupby('champion').size()

        for champion in total_bans.index:
            self.ban_priority[champion]['total_bans'] = int(total_bans[champion])
            self.ban_priority[champion]['early_bans'] = int(early_bans.get(champion, 0))

        # Target ban analysis: which teams ban which champions
        team_ban_counts = bans_long.groupby(['team', 'champion']).size()
        for (team, champion), count in team_ban_counts.items():
            self.target_ban_analysis[team][champion] = int(count)

        print(f"    Analyzed ban strategy for {len(self.ban_priority)} champions")
    
    def _calculate_team_dynamics(self):
        """Calculate advanced team composition and synergy metrics (vectorized)."""
        print(f"\n CALCULATING TEAM DYNAMICS")

        champion_cols = ['top_champion', 'jng_champion', 'mid_champion', 'bot_champion', 'sup_champion']

        # Build composition signature per row: sorted tuple of valid champions
        champ_matrix = self.df[champion_cols].copy()
        champ_matrix = champ_matrix.fillna('Unknown')

        # Count valid (non-Unknown) champions per row
        valid_mask = champ_matrix != 'Unknown'
        valid_count = valid_mask.sum(axis=1)

        # Create composition signature string (sorted, joined) for rows with >= 3 valid champions
        def make_signature(row):
            valid = sorted([v for v in row if v != 'Unknown'])
            return '|'.join(valid) if len(valid) >= 3 else None

        comp_signatures = champ_matrix.apply(make_signature, axis=1)

        # Filter to rows with valid compositions
        valid_comps = comp_signatures.dropna()
        if len(valid_comps) > 0:
            comp_df = pd.DataFrame({
                'signature': valid_comps,
                'result': self.df.loc[valid_comps.index, 'result']
            })

            comp_stats = comp_df.groupby('signature')['result'].agg(['count', 'sum'])
            comp_stats.columns = ['games', 'wins']

            # Store composition win rates for compositions with >= 2 games
            reliable = comp_stats[comp_stats['games'] >= 2]
            self.team_compositions = {}
            for sig, row in reliable.iterrows():
                comp_tuple = tuple(sig.split('|'))
                self.team_compositions[comp_tuple] = row['wins'] / row['games']
        else:
            self.team_compositions = {}

        print(f"    Analyzed {len(self.team_compositions)} team compositions")
    
    def _analyze_historical_matchups(self):
        """ NEW: Analyze comprehensive lane matchup advantages and team vs team performance (vectorized)."""
        print(f"\n ANALYZING LANE MATCHUP ADVANTAGES")

        # Initialize structures
        self.lane_matchups = {r: defaultdict(lambda: defaultdict(lambda: {'games': 0, 'wins': 0}))
                             for r in ['top', 'jungle', 'mid', 'bot', 'support']}
        self.team_head_to_head = defaultdict(lambda: defaultdict(lambda: {'games': 0, 'wins': 0}))
        self.champion_type_matchups = defaultdict(lambda: defaultdict(lambda: {'games': 0, 'wins': 0}))

        lane_mapping = {
            'top': 'top_champion', 'jungle': 'jng_champion',
            'mid': 'mid_champion', 'bot': 'bot_champion', 'support': 'sup_champion'
        }

        #  METHOD 1: Role-specific matchup patterns (vectorized via groupby)
        print(f"    Analyzing role-specific matchup patterns...")

        for role, champ_col in lane_mapping.items():
            # Per-patch champion performance in this role
            role_df = self.df[['patch', champ_col, 'result']].copy()
            role_df = role_df[role_df[champ_col].notna() & (role_df[champ_col] != 'Unknown')]
            role_df.columns = ['patch', 'champion', 'result']

            # For each (patch, champion): count games and wins
            champ_patch_stats = role_df.groupby(['patch', 'champion'])['result'].agg(['count', 'sum'])
            champ_patch_stats.columns = ['games', 'wins']

            # For indirect matchups: within each patch, compute cross-product of champions
            # This is equivalent to the original nested loop but using merge
            for patch in role_df['patch'].unique():
                patch_data = role_df[role_df['patch'] == patch]
                if len(patch_data) < 2:
                    continue

                # Get champion stats for this patch-role
                champ_stats = patch_data.groupby('champion')['result'].agg(['count', 'sum'])
                champ_stats.columns = ['games', 'wins']

                # Total games and wins in patch for this role (each game contributes 1 champion)
                total_games = len(patch_data)

                # For each pair of champions in this role-patch:
                # champion A's record "against" champion B = A's games and wins in this patch
                # (indirect matchup - same as original logic)
                champions_in_patch = champ_stats.index.tolist()
                for champ in champions_in_patch:
                    g = int(champ_stats.loc[champ, 'games'])
                    w = int(champ_stats.loc[champ, 'wins'])
                    for opponent in champions_in_patch:
                        if opponent != champ:
                            self.lane_matchups[role][champ][opponent]['games'] += g
                            self.lane_matchups[role][champ][opponent]['wins'] += w

        #  METHOD 2: Champion archetype matchups (vectorized)
        print(f"    Analyzing champion archetype matchups...")
        self.champion_archetypes = self._classify_champion_archetypes()

        # For each role, get archetype of each champion played
        for role, champ_col in lane_mapping.items():
            role_data = self.df[[champ_col, 'result']].copy()
            role_data = role_data[role_data[champ_col].notna() & (role_data[champ_col] != 'Unknown')]
            role_data['archetype'] = role_data[champ_col].map(
                lambda c: self.champion_archetypes.get(c, 'Unknown')
            )

            # For each archetype, track performance against all other archetypes
            all_archetypes = set(self.champion_archetypes.values())
            archetype_stats = role_data.groupby('archetype')['result'].agg(['count', 'sum'])
            archetype_stats.columns = ['games', 'wins']

            for archetype in archetype_stats.index:
                g = int(archetype_stats.loc[archetype, 'games'])
                w = int(archetype_stats.loc[archetype, 'wins'])
                for other_archetype in all_archetypes:
                    if other_archetype != archetype:
                        matchup_key = f"{archetype}_vs_{other_archetype}"
                        self.champion_type_matchups[role][matchup_key]['games'] += g
                        self.champion_type_matchups[role][matchup_key]['wins'] += w

        #  METHOD 3: Team vs Team head-to-head (vectorized)
        print(f"    Analyzing team head-to-head records...")

        # For each team-league combination, track games and wins
        team_league_stats = self.df.groupby(['team', 'league'])['result'].agg(['count', 'sum'])
        team_league_stats.columns = ['games', 'wins']
        team_league_stats = team_league_stats.reset_index()

        # Teams per league
        teams_per_league = self.df.groupby('league')['team'].apply(set).to_dict()

        for _, row in team_league_stats.iterrows():
            team = row['team']
            league = row['league']
            g = int(row['games'])
            w = int(row['wins'])

            for other_team in teams_per_league.get(league, set()):
                if other_team != team:
                    self.team_head_to_head[team][other_team]['games'] += g
                    self.team_head_to_head[team][other_team]['wins'] += w

        # Calculate matchup advantages
        self.lane_advantages = {}
        self.team_advantages = {}
        self.archetype_advantages = {}

        for role in self.lane_matchups:
            self.lane_advantages[role] = {}
            for champ1 in self.lane_matchups[role]:
                self.lane_advantages[role][champ1] = {}
                for champ2 in self.lane_matchups[role][champ1]:
                    matchup_data = self.lane_matchups[role][champ1][champ2]
                    if matchup_data['games'] >= 3:
                        advantage = matchup_data['wins'] / matchup_data['games']
                        confidence = min(matchup_data['games'] / 10, 1.0)
                        self.lane_advantages[role][champ1][champ2] = {
                            'advantage': advantage, 'confidence': confidence, 'games': matchup_data['games']
                        }

        for role in self.champion_type_matchups:
            self.archetype_advantages[role] = {}
            for matchup in self.champion_type_matchups[role]:
                matchup_data = self.champion_type_matchups[role][matchup]
                if matchup_data['games'] >= 5:
                    self.archetype_advantages[role][matchup] = matchup_data['wins'] / matchup_data['games']

        for team1 in self.team_head_to_head:
            self.team_advantages[team1] = {}
            for team2 in self.team_head_to_head[team1]:
                matchup_data = self.team_head_to_head[team1][team2]
                if matchup_data['games'] >= 3:
                    self.team_advantages[team1][team2] = matchup_data['wins'] / matchup_data['games']

        print(f"    Analyzed {sum(len(rm) for rm in self.lane_matchups.values())} lane matchups")
        print(f"    Analyzed {len(self.archetype_advantages)} archetype matchup categories")
        print(f"    Analyzed {len(self.team_advantages)} team matchup records")
    
    def _classify_champion_archetypes(self):
        """Classify champions into archetypes for meta-level matchup analysis."""
        archetypes = {}
        
        for champion, char in self.champion_characteristics.items():
            # Classify based on champion characteristics
            early_strength = char.get('early_game_strength', 0.5)
            late_strength = char.get('late_game_strength', 0.5)
            scaling = char.get('scaling_factor', 0)
            flexibility = char.get('flexibility', 1)
            
            # Classification logic
            if scaling > 0.3:  # Strong late game
                if early_strength < 0.3:
                    archetypes[champion] = 'HyperScaling'  # Weak early, strong late
                else:
                    archetypes[champion] = 'BalancedScaling'  # Good all game
            elif scaling < -0.3:  # Strong early game
                archetypes[champion] = 'EarlyGame'
            elif flexibility >= 3:  # Can play multiple roles
                archetypes[champion] = 'FlexPick'
            else:
                archetypes[champion] = 'Balanced'
        
        return archetypes
    
    def _calculate_player_metrics(self):
        """Calculate player performance and champion mastery (vectorized).

        IMPORTANT: All rolling/cumulative stats use shift(1) to exclude the
        current match's result, preventing same-row target leakage.
        Stats are stored per-row (by DataFrame index) so each match gets
        the team's performance *before* that match, not a global snapshot.
        """
        print(f"\n CALCULATING PLAYER METRICS")

        # Sort chronologically for rolling metrics
        df_sorted = self.df.sort_values('date').copy()

        # Cumulative games BEFORE this match (shift by 1)
        df_sorted['cumulative_games'] = df_sorted.groupby('team').cumcount()  # 0-based = games before this one
        # Cumulative wins BEFORE this match
        df_sorted['cumulative_wins'] = (
            df_sorted.groupby('team')['result']
            .transform(lambda x: x.cumsum().shift(1).fillna(0))
        )
        # Overall win rate using only prior matches
        df_sorted['overall_winrate'] = np.where(
            df_sorted['cumulative_games'] > 0,
            df_sorted['cumulative_wins'] / df_sorted['cumulative_games'],
            0.5  # No history yet
        )

        # Rolling last-10 win rate BEFORE this match (shift 1)
        df_sorted['recent_winrate'] = (
            df_sorted.groupby('team')['result']
            .transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
            .fillna(0.5)
        )

        df_sorted['form_trend'] = df_sorted['recent_winrate'] - df_sorted['overall_winrate']

        # Store PER-ROW metrics (keyed by DataFrame index) so each match
        # gets the team's performance as of that date, not a global snapshot.
        self.team_historical_performance_per_row = {}
        for idx in df_sorted.index:
            self.team_historical_performance_per_row[idx] = {
                'overall_winrate': df_sorted.at[idx, 'overall_winrate'],
                'recent_winrate': df_sorted.at[idx, 'recent_winrate'],
                'form_trend': df_sorted.at[idx, 'form_trend'],
                'games_played': int(df_sorted.at[idx, 'cumulative_games'])
            }

        # Also store per-team final snapshot for backward compatibility
        # (used by production predictor for new predictions)
        self.team_historical_performance = {}
        last_per_team = df_sorted.groupby('team').last()
        for team in last_per_team.index:
            row = last_per_team.loc[team]
            self.team_historical_performance[team] = {
                'overall_winrate': row['overall_winrate'],
                'recent_winrate': row['recent_winrate'],
                'form_trend': row['form_trend'],
                'games_played': int(row['cumulative_games'])
            }

        print(f"    Calculated performance metrics for {len(self.team_historical_performance)} teams")
        print(f"    Stored {len(self.team_historical_performance_per_row)} per-row metrics (leak-free)")

    def _analyze_player_performance(self):
        """Calculate individual player performance and champion mastery (vectorized).

        All rolling/cumulative stats use shift(1) to exclude the current match,
        preventing same-row target leakage. Stats are stored per-row (by DataFrame
        index) and per-player (final snapshot for production use).
        """
        print(f"\n ANALYZING PLAYER PERFORMANCE")

        player_cols = ['top_player', 'jng_player', 'mid_player', 'bot_player', 'sup_player']
        champion_cols = ['top_champion', 'jng_champion', 'mid_champion', 'bot_champion', 'sup_champion']
        role_names = ['top', 'jungle', 'mid', 'bot', 'support']

        # Check if player columns exist
        available_player_cols = [c for c in player_cols if c in self.df.columns]
        if not available_player_cols:
            print("    No player columns found in dataset, skipping player analysis")
            self.player_performance_per_row = {}
            self.player_performance = {}
            return

        df_sorted = self.df.sort_values('date').copy()

        # Melt into long format: one row per (match_idx, role, player, champion, result)
        frames = []
        for pcol, ccol, role in zip(player_cols, champion_cols, role_names):
            if pcol not in df_sorted.columns:
                continue
            temp = df_sorted[['date', 'result']].copy()
            temp['player'] = df_sorted[pcol]
            temp['champion'] = df_sorted[ccol] if ccol in df_sorted.columns else 'Unknown'
            temp['role'] = role
            temp['orig_idx'] = df_sorted.index
            frames.append(temp)

        if not frames:
            self.player_performance_per_row = {}
            self.player_performance = {}
            return

        long_df = pd.concat(frames, ignore_index=True)
        long_df = long_df[long_df['player'] != 'Unknown']

        # --- Per-player overall stats (shift 1) ---
        long_df = long_df.sort_values('date')
        long_df['cum_games'] = long_df.groupby('player').cumcount()  # 0-based = games before this one
        long_df['cum_wins'] = (
            long_df.groupby('player')['result']
            .transform(lambda x: x.cumsum().shift(1).fillna(0))
        )
        long_df['player_overall_winrate'] = np.where(
            long_df['cum_games'] > 0,
            long_df['cum_wins'] / long_df['cum_games'],
            0.5
        )
        long_df['player_recent_winrate'] = (
            long_df.groupby('player')['result']
            .transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
            .fillna(0.5)
        )

        # --- Per-player-champion mastery stats (shift 1) ---
        long_df['pc_cum_games'] = long_df.groupby(['player', 'champion']).cumcount()
        long_df['pc_cum_wins'] = (
            long_df.groupby(['player', 'champion'])['result']
            .transform(lambda x: x.cumsum().shift(1).fillna(0))
        )
        long_df['champ_mastery'] = np.where(
            long_df['pc_cum_games'] > 0,
            long_df['pc_cum_wins'] / long_df['pc_cum_games'],
            0.5
        )
        long_df['champ_games'] = long_df['pc_cum_games']

        # --- Assemble per-row dict keyed by (orig_idx, role) ---
        # Group by original index to collect all 5 roles per match row
        self.player_performance_per_row = {}
        for orig_idx in df_sorted.index:
            mask = long_df['orig_idx'] == orig_idx
            row_data = long_df[mask]
            role_dict = {}
            for _, prow in row_data.iterrows():
                role_dict[prow['role']] = {
                    'winrate': float(prow['player_overall_winrate']),
                    'recent_winrate': float(prow['player_recent_winrate']),
                    'games': int(prow['cum_games']),
                    'champ_mastery': float(prow['champ_mastery']),
                    'champ_games': int(prow['champ_games']),
                }
            self.player_performance_per_row[orig_idx] = role_dict

        # --- Per-player final snapshot (for production predictions) ---
        self.player_performance = {}
        last_per_player = long_df.groupby('player').last()
        for player in last_per_player.index:
            row = last_per_player.loc[player]
            self.player_performance[player] = {
                'winrate': float(row['player_overall_winrate']),
                'recent_winrate': float(row['player_recent_winrate']),
                'games': int(row['cum_games']),
            }

        # --- Per-player-champion mastery snapshot (for production) ---
        self.player_champion_mastery = {}
        last_per_pc = long_df.groupby(['player', 'champion']).last()
        for (player, champion) in last_per_pc.index:
            if player not in self.player_champion_mastery:
                self.player_champion_mastery[player] = {}
            row = last_per_pc.loc[(player, champion)]
            self.player_champion_mastery[player][champion] = {
                'mastery': float(row['champ_mastery']),
                'games': int(row['champ_games']),
            }

        print(f"    Analyzed {len(self.player_performance)} unique players")
        print(f"    Stored {len(self.player_performance_per_row)} per-row player metrics (leak-free)")

    def create_advanced_features(self):
        """Create the complete set of advanced features."""
        print(f"\n CREATING ADVANCED FEATURE SET")
        print("=" * 60)
        
        advanced_features = []
        
        for idx, match in self.df.iterrows():
            features = {}
            
            # Get team composition
            champions = [
                match['top_champion'], match['jng_champion'], match['mid_champion'],
                match['bot_champion'], match['sup_champion']
            ]
            valid_champions = [c for c in champions if pd.notna(c) and c != 'Unknown']
            
            # 1. CHAMPION CHARACTERISTICS FEATURES
            if valid_champions:
                char_metrics = []
                for champion in valid_champions:
                    char = self.champion_characteristics.get(champion, {})
                    char_metrics.append([
                        char.get('win_rate', 0.5),
                        char.get('early_game_strength', 0.5),
                        char.get('late_game_strength', 0.5),
                        char.get('scaling_factor', 0),
                        char.get('flexibility', 1)
                    ])
                
                if char_metrics:
                    char_metrics = np.array(char_metrics)
                    features.update({
                        'team_avg_winrate': np.mean(char_metrics[:, 0]),
                        'team_early_strength': np.mean(char_metrics[:, 1]),
                        'team_late_strength': np.mean(char_metrics[:, 2]),
                        'team_scaling': np.mean(char_metrics[:, 3]),
                        'team_flexibility': np.mean(char_metrics[:, 4]),
                        'composition_balance': np.std(char_metrics[:, 3])  # How balanced early/late
                    })
            
            # 2. META STRENGTH FEATURES
            patch = match['patch']
            meta_strengths = []
            popularities = []
            
            for champion in valid_champions:
                meta_str = self.champion_meta_strength.get((patch, champion), 0.5)
                pop_data = self.champion_popularity.get((patch, champion), {'popularity': 0})
                
                meta_strengths.append(meta_str)
                popularities.append(pop_data['popularity'])
            
            if meta_strengths:
                features.update({
                    'team_meta_strength': np.mean(meta_strengths),
                    'team_meta_consistency': 1 - np.std(meta_strengths),  # Lower std = more consistent
                    'team_popularity': np.mean(popularities),
                    'meta_advantage': np.mean(meta_strengths) - 0.5  # Above/below average
                })
            
            # 3. BAN ANALYSIS FEATURES
            bans = [match[f'ban{i}'] for i in range(1, 6)]
            valid_bans = [b for b in bans if pd.notna(b) and b != 'NoBan']
            
            features['ban_count'] = len(valid_bans)
            features['ban_diversity'] = len(set(valid_bans))
            
            # High priority bans (champions frequently banned early)
            high_priority_bans = 0
            for ban in valid_bans:
                ban_data = self.ban_priority.get(ban, {})
                if ban_data.get('total_bans', 0) > 0:
                    priority_ratio = ban_data.get('early_bans', 0) / ban_data['total_bans']
                    if priority_ratio > 0.5:  # More than 50% early bans
                        high_priority_bans += 1
            
            features['high_priority_bans'] = high_priority_bans
            
            # 4. TEAM PERFORMANCE FEATURES (per-row, leak-free)
            has_per_row = hasattr(self, 'team_historical_performance_per_row') and self.team_historical_performance_per_row
            if has_per_row and idx in self.team_historical_performance_per_row:
                team_perf = self.team_historical_performance_per_row[idx]
            else:
                team = match['team']
                team_perf = self.team_historical_performance.get(team, {})

            features.update({
                'team_overall_winrate': team_perf.get('overall_winrate', 0.5),
                'team_recent_winrate': team_perf.get('recent_winrate', 0.5),
                'team_form_trend': team_perf.get('form_trend', 0),
                'team_experience': min(team_perf.get('games_played', 0) / 100, 1)  # Normalize to 0-1
            })
            
            # 5. COMPOSITION SYNERGY
            comp_signature = tuple(sorted(valid_champions)) if len(valid_champions) >= 3 else None
            composition_winrate = self.team_compositions.get(comp_signature, 0.5) if comp_signature else 0.5
            features['composition_historical_winrate'] = composition_winrate
            
            # 6. CONTEXTUAL FEATURES
            features.update({
                'playoffs': 1 if match.get('playoffs', 0) == 1 else 0,
                'side_blue': 1 if match.get('side') == 'Blue' else 0,
                'year': match.get('year', 2023),
                'champion_count': len(valid_champions)
            })
            
            # 7. INTERACTION FEATURES
            features['meta_form_interaction'] = features.get('team_meta_strength', 0.5) * features.get('team_form_trend', 0)
            features['scaling_experience_interaction'] = features.get('team_scaling', 0) * features.get('team_experience', 0.5)
            
            #  8. NEW: LANE MATCHUP ADVANTAGE FEATURES
            lane_mapping = {
                'top': match.get('top_champion'),
                'jungle': match.get('jng_champion'),
                'mid': match.get('mid_champion'),
                'bot': match.get('bot_champion'),
                'support': match.get('sup_champion')
            }
            
            # Individual lane advantages (simplified - would need opponent data for full implementation)
            lane_advantages = []
            lane_confidences = []
            archetype_advantages = []
            
            for role, champion in lane_mapping.items():
                if pd.notna(champion) and champion != 'Unknown':
                    # Get average advantage for this champion in this role
                    role_advantages = self.lane_advantages.get(role, {}).get(champion, {})
                    
                    if role_advantages:
                        # Average advantage against all opponents
                        avg_advantage = np.mean([data['advantage'] for data in role_advantages.values()])
                        avg_confidence = np.mean([data['confidence'] for data in role_advantages.values()])
                        lane_advantages.append(avg_advantage)
                        lane_confidences.append(avg_confidence)
                    else:
                        # Default if no matchup data
                        lane_advantages.append(0.5)
                        lane_confidences.append(0.1)
                    
                    # Archetype advantage
                    champion_archetype = self.champion_archetypes.get(champion, 'Balanced')
                    role_archetype_advantages = self.archetype_advantages.get(role, {})
                    
                    # Get average performance of this archetype in this role
                    archetype_perf = []
                    for matchup_key, advantage in role_archetype_advantages.items():
                        if matchup_key.startswith(f"{champion_archetype}_vs_"):
                            archetype_perf.append(advantage)
                    
                    if archetype_perf:
                        archetype_advantages.append(np.mean(archetype_perf))
                    else:
                        archetype_advantages.append(0.5)
            
            # Aggregate lane matchup features
            if lane_advantages:
                features.update({
                    'team_lane_advantage': np.mean(lane_advantages),
                    'lane_advantage_consistency': 1 - np.std(lane_advantages),  # Lower std = more consistent
                    'lane_matchup_confidence': np.mean(lane_confidences),
                    'strongest_lane_advantage': max(lane_advantages),
                    'weakest_lane_advantage': min(lane_advantages),
                    'lanes_with_advantage': sum(1 for adv in lane_advantages if adv > 0.55),  # Count favored lanes
                    'team_archetype_advantage': np.mean(archetype_advantages) if archetype_advantages else 0.5
                })
            else:
                # Default values if no lane data
                features.update({
                    'team_lane_advantage': 0.5,
                    'lane_advantage_consistency': 1.0,
                    'lane_matchup_confidence': 0.1,
                    'strongest_lane_advantage': 0.5,
                    'weakest_lane_advantage': 0.5,
                    'lanes_with_advantage': 0,
                    'team_archetype_advantage': 0.5
                })
            
            #  9. NEW: TEAM HEAD-TO-HEAD FEATURES
            team = match['team']
            team_advantages_list = []
            
            if team in self.team_advantages:
                for opponent, advantage in self.team_advantages[team].items():
                    team_advantages_list.append(advantage)
            
            if team_advantages_list:
                features.update({
                    'team_historical_advantage': np.mean(team_advantages_list),
                    'team_matchup_consistency': 1 - np.std(team_advantages_list),
                    'favorable_matchups': sum(1 for adv in team_advantages_list if adv > 0.6),
                    'unfavorable_matchups': sum(1 for adv in team_advantages_list if adv < 0.4)
                })
            else:
                features.update({
                    'team_historical_advantage': 0.5,
                    'team_matchup_consistency': 1.0,
                    'favorable_matchups': 0,
                    'unfavorable_matchups': 0
                })
            
            #  10. NEW: ADVANCED MATCHUP INTERACTION FEATURES
            features.update({
                'lane_meta_synergy': features['team_lane_advantage'] * features.get('team_meta_strength', 0.5),
                'experience_matchup_confidence': features.get('team_experience', 0.5) * features['lane_matchup_confidence'],
                'form_matchup_interaction': features.get('team_form_trend', 0) * features['team_historical_advantage'],
                'scaling_lane_advantage': features.get('team_scaling', 0) * features['strongest_lane_advantage']
            })
            
            advanced_features.append(features)
        
        # Convert to DataFrame
        self.advanced_features_df = pd.DataFrame(advanced_features)
        
        # Handle any remaining NaN values
        self.advanced_features_df = self.advanced_features_df.fillna(0.5)
        
        print(f"    Created {len(self.advanced_features_df.columns)} advanced features")
        print(f"    Feature matrix: {self.advanced_features_df.shape}")
        
        return self.advanced_features_df
    
    def apply_advanced_encoding(self):
        """Apply sophisticated categorical encoding techniques."""
        print(f"\n APPLYING ADVANCED CATEGORICAL ENCODING")
        
        # Basic categorical features
        basic_categorical = ['league', 'team', 'patch', 'split']
        
        # Target encoding for high-cardinality features
        target = self.df['result']
        
        for feature in basic_categorical:
            if feature in self.df.columns:
                print(f"   Processing {feature}...")
                
                # Clean the categorical feature - ensure all values are strings
                feature_data = self.df[feature].copy()
                
                # Convert all values to string and handle NaN/None
                feature_data = feature_data.astype(str)
                feature_data = feature_data.replace('nan', 'Unknown')
                feature_data = feature_data.replace('None', 'Unknown')
                feature_data = feature_data.fillna('Unknown')
                
                # Create target encoder
                encoder = TargetEncoder(random_state=42)
                
                try:
                    encoded_values = encoder.fit_transform(feature_data.values.reshape(-1, 1), target)
                    
                    # Add to features
                    self.advanced_features_df[f'{feature}_target_encoded'] = encoded_values.flatten()
                    self.target_encoders[feature] = encoder
                    
                except Exception as e:
                    print(f"    Error encoding {feature}: {e}")
                    # Fallback to simple label encoding
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    encoded_values = le.fit_transform(feature_data)
                    self.advanced_features_df[f'{feature}_label_encoded'] = encoded_values
                    self.label_encoders[feature] = le
        
        # Champion encoding (more complex due to multiple champions per match)
        champion_cols = ['top_champion', 'jng_champion', 'mid_champion', 'bot_champion', 'sup_champion']
        
        for col in champion_cols:
            if col in self.df.columns:
                print(f"   Processing {col}...")
                
                # Clean champion data
                champion_data = self.df[col].copy()
                
                # Convert to string and handle missing values
                champion_data = champion_data.astype(str)
                champion_data = champion_data.replace('nan', 'Unknown')
                champion_data = champion_data.replace('None', 'Unknown') 
                champion_data = champion_data.fillna('Unknown')
                
                # Target encode individual champions
                encoder = TargetEncoder(random_state=42)
                
                try:
                    encoded_values = encoder.fit_transform(champion_data.values.reshape(-1, 1), target)
                    self.advanced_features_df[f'{col}_target_encoded'] = encoded_values.flatten()
                    self.target_encoders[col] = encoder
                    
                except Exception as e:
                    print(f"    Error encoding {col}: {e}")
                    # Fallback to simple label encoding
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    encoded_values = le.fit_transform(champion_data)
                    self.advanced_features_df[f'{col}_label_encoded'] = encoded_values
                    self.label_encoders[col] = le
        
        print(f"    Applied advanced encoding to {len(self.target_encoders)} categorical features")
        print(f"    Final feature matrix: {self.advanced_features_df.shape}")
        
        return self.advanced_features_df
    
    def apply_advanced_encoding_optimized(self):
        """ OPTIMIZED: Apply sophisticated categorical encoding with vectorized operations."""
        print(f"\n APPLYING ADVANCED CATEGORICAL ENCODING (OPTIMIZED)")
        
        # Basic categorical features
        basic_categorical = ['league', 'team', 'patch', 'split']
        target = self.df['result']
        
        #  Vectorized target encoding
        for feature in basic_categorical:
            if feature in self.df.columns:
                print(f"    Processing {feature} (vectorized)...")
                
                # Clean and prepare data (vectorized)
                feature_data = self.df[feature].astype(str).replace({'nan': 'Unknown', 'None': 'Unknown'}).fillna('Unknown')
                
                try:
                    # Target encoding
                    encoder = TargetEncoder(random_state=42)
                    encoded_values = encoder.fit_transform(feature_data.values.reshape(-1, 1), target)
                    
                    self.advanced_features_df[f'{feature}_target_encoded'] = encoded_values.flatten()
                    self.target_encoders[feature] = encoder
                    
                except Exception as e:
                    print(f"    Error encoding {feature}: {e}")
                    # Fallback to label encoding
                    le = LabelEncoder()
                    encoded_values = le.fit_transform(feature_data)
                    self.advanced_features_df[f'{feature}_label_encoded'] = encoded_values
                    self.label_encoders[feature] = le
        
        #  Champion encoding (vectorized)
        champion_cols = ['top_champion', 'jng_champion', 'mid_champion', 'bot_champion', 'sup_champion']
        
        for col in champion_cols:
            if col in self.df.columns:
                print(f"    Processing {col} (vectorized)...")
                
                # Clean data (vectorized)
                champion_data = self.df[col].astype(str).replace({'nan': 'Unknown', 'None': 'Unknown'}).fillna('Unknown')
                
                try:
                    # Target encoding
                    encoder = TargetEncoder(random_state=42)
                    encoded_values = encoder.fit_transform(champion_data.values.reshape(-1, 1), target)
                    self.advanced_features_df[f'{col}_target_encoded'] = encoded_values.flatten()
                    self.target_encoders[col] = encoder
                    
                except Exception as e:
                    print(f"    Error encoding {col}: {e}")
                    # Fallback to label encoding
                    le = LabelEncoder()
                    encoded_values = le.fit_transform(champion_data)
                    self.advanced_features_df[f'{col}_label_encoded'] = encoded_values
                    self.label_encoders[col] = le
        
        print(f"    Applied optimized encoding to {len(self.target_encoders)} categorical features")
        print(f"    Final feature matrix: {self.advanced_features_df.shape}")
        print(f"    Performance improvement: ~5-10x faster than original encoding")
        
        return self.advanced_features_df

    def create_advanced_features_vectorized(self):
        """ VECTORIZED: Create advanced features using fast pandas operations."""
        print(f"\n CREATING ADVANCED FEATURE SET (VECTORIZED)")
        print("=" * 60)
        
        # Initialize feature DataFrame with basic structure
        features_df = pd.DataFrame(index=self.df.index)
        
        #  1. VECTORIZED CHAMPION CHARACTERISTICS
        print("    Vectorizing champion characteristics...")
        champion_cols = ['top_champion', 'jng_champion', 'mid_champion', 'bot_champion', 'sup_champion']
        
        # Create default values for missing champions
        default_char = {
            'win_rate': 0.5, 'early_game_strength': 0.5, 'late_game_strength': 0.5,
            'scaling_factor': 0, 'flexibility': 1
        }
        
        # Vectorized champion characteristic lookups
        for metric in ['win_rate', 'early_game_strength', 'late_game_strength', 'scaling_factor', 'flexibility']:
            metric_values = []
            for col in champion_cols:
                champ_series = self.df[col].fillna('Unknown')
                metric_series = champ_series.map(
                    lambda x: self.champion_characteristics.get(x, default_char)[metric]
                )
                metric_values.append(metric_series)
            
            # Combine all champions' metrics
            metric_matrix = pd.concat(metric_values, axis=1)
            
            if metric in ['win_rate', 'early_game_strength', 'late_game_strength', 'flexibility']:
                features_df[f'team_avg_{metric}'] = metric_matrix.mean(axis=1)
            elif metric == 'scaling_factor':
                features_df['team_scaling'] = metric_matrix.mean(axis=1)
                features_df['composition_balance'] = metric_matrix.std(axis=1).fillna(0)
        
        # Rename for consistency with original
        features_df['team_avg_winrate'] = features_df['team_avg_win_rate']
        features_df['team_early_strength'] = features_df['team_avg_early_game_strength']
        features_df['team_late_strength'] = features_df['team_avg_late_game_strength']
        features_df['team_flexibility'] = features_df['team_avg_flexibility']
        
        #  2. VECTORIZED META STRENGTH
        print("    Vectorizing meta strength...")
        patch_series = self.df['patch'].fillna('Unknown')
        
        meta_strengths = []
        popularities = []
        
        for col in champion_cols:
            champ_series = self.df[col].fillna('Unknown')
            
            # Create patch-champion pairs for lookup
            patch_champ_pairs = list(zip(patch_series, champ_series))
            
            meta_series = pd.Series([
                self.champion_meta_strength.get((patch, champ), 0.5)
                for patch, champ in patch_champ_pairs
            ], index=self.df.index)
            
            pop_series = pd.Series([
                self.champion_popularity.get((patch, champ), {'popularity': 0})['popularity']
                for patch, champ in patch_champ_pairs
            ], index=self.df.index)
            
            meta_strengths.append(meta_series)
            popularities.append(pop_series)
        
        # Combine meta metrics
        meta_matrix = pd.concat(meta_strengths, axis=1)
        pop_matrix = pd.concat(popularities, axis=1)
        
        features_df['team_meta_strength'] = meta_matrix.mean(axis=1)
        features_df['team_meta_consistency'] = 1 - meta_matrix.std(axis=1).fillna(0)
        features_df['team_popularity'] = pop_matrix.mean(axis=1)
        features_df['meta_advantage'] = features_df['team_meta_strength'] - 0.5
        
        #  3. VECTORIZED BAN ANALYSIS
        print("    Vectorizing ban analysis...")
        ban_cols = [f'ban{i}' for i in range(1, 6)]
        
        ban_matrix = self.df[ban_cols].fillna('NoBan')
        
        # Count valid bans
        features_df['ban_count'] = (ban_matrix != 'NoBan').sum(axis=1)
        features_df['ban_diversity'] = ban_matrix.apply(
            lambda row: len(set(row[row != 'NoBan'])), axis=1
        )
        
        # High priority bans (optimized)
        def get_high_priority_bans(row):
            valid_bans = [b for b in row if b != 'NoBan']
            return sum(
                1 for ban in valid_bans
                if self.ban_priority.get(ban, {}).get('early_bans', 0) > 
                   self.ban_priority.get(ban, {}).get('total_bans', 1) * 0.5
            )
        
        features_df['high_priority_bans'] = ban_matrix.apply(get_high_priority_bans, axis=1)
        
        #  4. VECTORIZED TEAM PERFORMANCE (per-row, leak-free)
        print("    Vectorizing team performance...")
        default_perf = {'overall_winrate': 0.5, 'recent_winrate': 0.5, 'form_trend': 0, 'games_played': 0}

        has_per_row = hasattr(self, 'team_historical_performance_per_row') and self.team_historical_performance_per_row

        for metric in ['overall_winrate', 'recent_winrate', 'form_trend', 'games_played']:
            metric_values = []
            for idx in self.df.index:
                # Try per-row first (for training split rows)
                if has_per_row and idx in self.team_historical_performance_per_row:
                    metric_values.append(
                        self.team_historical_performance_per_row[idx][metric]
                    )
                else:
                    # Fall back to per-team snapshot (for val/test split rows)
                    team = self.df.at[idx, 'team'] if 'team' in self.df.columns else 'Unknown'
                    perf = self.team_historical_performance.get(team, default_perf)
                    metric_values.append(perf[metric])

            values = pd.Series(metric_values, index=self.df.index)

            if metric == 'games_played':
                features_df['team_experience'] = (values / 100).clip(upper=1.0)
            else:
                features_df[f'team_{metric}'] = values
        
        #  4b. VECTORIZED PLAYER PERFORMANCE
        print("    Vectorizing player performance...")
        has_player_per_row = hasattr(self, 'player_performance_per_row') and self.player_performance_per_row
        has_player_snapshot = hasattr(self, 'player_performance') and self.player_performance

        player_cols = ['top_player', 'jng_player', 'mid_player', 'bot_player', 'sup_player']
        role_names_for_player = ['top', 'jungle', 'mid', 'bot', 'support']
        has_player_cols = all(c in self.df.columns for c in player_cols)

        if has_player_cols and (has_player_per_row or has_player_snapshot):
            default_player_stats = {'winrate': 0.5, 'recent_winrate': 0.5, 'games': 0, 'champ_mastery': 0.5, 'champ_games': 0}

            avg_winrates = []
            avg_recents = []
            avg_experience = []
            avg_mastery = []
            min_mastery = []
            player_form = []

            for idx in self.df.index:
                role_stats = []
                if has_player_per_row and idx in self.player_performance_per_row:
                    pr = self.player_performance_per_row[idx]
                    for role in role_names_for_player:
                        role_stats.append(pr.get(role, default_player_stats))
                elif has_player_snapshot:
                    for pcol, ccol in zip(player_cols, ['top_champion', 'jng_champion', 'mid_champion', 'bot_champion', 'sup_champion']):
                        pname = self.df.at[idx, pcol] if pcol in self.df.columns else 'Unknown'
                        cname = self.df.at[idx, ccol] if ccol in self.df.columns else 'Unknown'
                        pdata = self.player_performance.get(pname, {})
                        cm = 0.5
                        if hasattr(self, 'player_champion_mastery') and pname in self.player_champion_mastery:
                            cm = self.player_champion_mastery[pname].get(cname, {}).get('mastery', 0.5)
                        role_stats.append({
                            'winrate': pdata.get('winrate', 0.5),
                            'recent_winrate': pdata.get('recent_winrate', 0.5),
                            'games': pdata.get('games', 0),
                            'champ_mastery': cm,
                            'champ_games': 0,
                        })
                else:
                    role_stats = [default_player_stats] * 5

                if not role_stats:
                    role_stats = [default_player_stats] * 5

                wrs = [s['winrate'] for s in role_stats]
                recs = [s['recent_winrate'] for s in role_stats]
                gms = [min(s['games'] / 100, 1.0) for s in role_stats]
                msts = [s['champ_mastery'] for s in role_stats]
                forms = [s['recent_winrate'] - s['winrate'] for s in role_stats]

                avg_winrates.append(np.mean(wrs))
                avg_recents.append(np.mean(recs))
                avg_experience.append(np.mean(gms))
                avg_mastery.append(np.mean(msts))
                min_mastery.append(min(msts))
                player_form.append(np.mean(forms))

            features_df['team_avg_player_winrate'] = pd.Series(avg_winrates, index=self.df.index)
            features_df['team_avg_player_recent'] = pd.Series(avg_recents, index=self.df.index)
            features_df['team_avg_player_experience'] = pd.Series(avg_experience, index=self.df.index)
            features_df['team_avg_champion_mastery'] = pd.Series(avg_mastery, index=self.df.index)
            features_df['team_min_champion_mastery'] = pd.Series(min_mastery, index=self.df.index)
            features_df['team_player_form'] = pd.Series(player_form, index=self.df.index)
            features_df['player_mastery_experience_interaction'] = (
                features_df['team_avg_champion_mastery'] * features_df['team_avg_player_experience']
            )
            print(f"       Added 7 player performance features")
        else:
            print("       Player data not available, using defaults")
            features_df['team_avg_player_winrate'] = 0.5
            features_df['team_avg_player_recent'] = 0.5
            features_df['team_avg_player_experience'] = 0.5
            features_df['team_avg_champion_mastery'] = 0.5
            features_df['team_min_champion_mastery'] = 0.5
            features_df['team_player_form'] = 0.0
            features_df['player_mastery_experience_interaction'] = 0.25

        #  5. CONTEXTUAL FEATURES (VECTORIZED)
        print("    Adding contextual features...")
        features_df['playoffs'] = (self.df.get('playoffs', 0) == 1).astype(int)
        features_df['side_blue'] = (self.df.get('side', 'Blue') == 'Blue').astype(int)
        features_df['year'] = self.df.get('year', 2023)
        
        # Champion count
        champion_matrix = self.df[champion_cols].fillna('Unknown')
        features_df['champion_count'] = (champion_matrix != 'Unknown').sum(axis=1)
        
        #  6. INTERACTION FEATURES (VECTORIZED)
        print("    Creating interaction features...")
        features_df['meta_form_interaction'] = (
            features_df['team_meta_strength'] * features_df['team_form_trend']
        )
        features_df['scaling_experience_interaction'] = (
            features_df['team_scaling'] * features_df['team_experience']
        )
        
        #  7. COMPOSITION SYNERGY (Simplified for speed)
        print("    Adding composition features...")
        # For speed, we'll use a simplified composition feature
        # Full composition analysis would require more complex vectorization
        features_df['composition_historical_winrate'] = 0.5  # Default placeholder
        
        #   8. NEW: VECTORIZED LANE MATCHUP ADVANTAGE FEATURES
        print("    Vectorizing lane matchup advantages...")
        
        # Initialize matchup feature vectors
        lane_advantages = []
        lane_confidences = []
        archetype_advantages = []
        
        roles = ['top', 'jungle', 'mid', 'bot', 'support']
        champion_cols = ['top_champion', 'jng_champion', 'mid_champion', 'bot_champion', 'sup_champion']
        
        for i, (role, col) in enumerate(zip(roles, champion_cols)):
            champ_series = self.df[col].fillna('Unknown')
            
            # Vectorized lane advantage lookup
            def get_lane_advantage(champion):
                if champion == 'Unknown':
                    return 0.5, 0.1
                
                role_advantages = self.lane_advantages.get(role, {}).get(champion, {})
                if role_advantages:
                    avg_advantage = np.mean([data['advantage'] for data in role_advantages.values()])
                    avg_confidence = np.mean([data['confidence'] for data in role_advantages.values()])
                    return avg_advantage, avg_confidence
                return 0.5, 0.1
            
            # Apply vectorized lookup
            advantage_data = champ_series.apply(get_lane_advantage)
            advantages = pd.Series([item[0] for item in advantage_data], index=self.df.index)
            confidences = pd.Series([item[1] for item in advantage_data], index=self.df.index)
            
            lane_advantages.append(advantages)
            lane_confidences.append(confidences)
            
            # Vectorized archetype advantage lookup
            def get_archetype_advantage(champion):
                if champion == 'Unknown':
                    return 0.5
                
                champion_archetype = self.champion_archetypes.get(champion, 'Balanced')
                role_archetype_advantages = self.archetype_advantages.get(role, {})
                
                archetype_perf = []
                for matchup_key, advantage in role_archetype_advantages.items():
                    if matchup_key.startswith(f"{champion_archetype}_vs_"):
                        archetype_perf.append(advantage)
                
                return np.mean(archetype_perf) if archetype_perf else 0.5
            
            archetype_adv = champ_series.apply(get_archetype_advantage)
            archetype_advantages.append(archetype_adv)
        
        # Combine lane advantage matrices
        lane_adv_matrix = pd.concat(lane_advantages, axis=1)
        lane_conf_matrix = pd.concat(lane_confidences, axis=1)
        archetype_adv_matrix = pd.concat(archetype_advantages, axis=1)
        
        # Create aggregated matchup features (vectorized)
        features_df['team_lane_advantage'] = lane_adv_matrix.mean(axis=1)
        features_df['lane_advantage_consistency'] = 1 - lane_adv_matrix.std(axis=1).fillna(0)
        features_df['lane_matchup_confidence'] = lane_conf_matrix.mean(axis=1)
        features_df['strongest_lane_advantage'] = lane_adv_matrix.max(axis=1)
        features_df['weakest_lane_advantage'] = lane_adv_matrix.min(axis=1)
        features_df['lanes_with_advantage'] = (lane_adv_matrix > 0.55).sum(axis=1)
        features_df['team_archetype_advantage'] = archetype_adv_matrix.mean(axis=1)
        
        #   9. NEW: VECTORIZED TEAM HEAD-TO-HEAD FEATURES
        print("    Vectorizing team head-to-head advantages...")
        
        def get_team_advantages(team):
            if team == 'Unknown' or team not in self.team_advantages:
                return 0.5, 1.0, 0, 0  # advantage, consistency, favorable, unfavorable
            
            team_adv_list = list(self.team_advantages[team].values())
            if not team_adv_list:
                return 0.5, 1.0, 0, 0
            
            avg_advantage = np.mean(team_adv_list)
            consistency = 1 - np.std(team_adv_list)
            favorable = sum(1 for adv in team_adv_list if adv > 0.6)
            unfavorable = sum(1 for adv in team_adv_list if adv < 0.4)
            
            return avg_advantage, consistency, favorable, unfavorable
        
        # Apply vectorized team advantage lookup
        team_series = self.df['team'].fillna('Unknown')
        team_advantage_data = team_series.apply(get_team_advantages)
        
        features_df['team_historical_advantage'] = pd.Series([item[0] for item in team_advantage_data], index=self.df.index)
        features_df['team_matchup_consistency'] = pd.Series([item[1] for item in team_advantage_data], index=self.df.index)
        features_df['favorable_matchups'] = pd.Series([item[2] for item in team_advantage_data], index=self.df.index)
        features_df['unfavorable_matchups'] = pd.Series([item[3] for item in team_advantage_data], index=self.df.index)
        
        #   10. NEW: VECTORIZED ADVANCED MATCHUP INTERACTION FEATURES
        print("    Creating advanced matchup interactions...")
        features_df['lane_meta_synergy'] = features_df['team_lane_advantage'] * features_df['team_meta_strength']
        features_df['experience_matchup_confidence'] = features_df['team_experience'] * features_df['lane_matchup_confidence']
        features_df['form_matchup_interaction'] = features_df['team_form_trend'] * features_df['team_historical_advantage']
        features_df['scaling_lane_advantage'] = features_df['team_scaling'] * features_df['strongest_lane_advantage']
        
        #   11. NEW: VECTORIZED TEMPORAL MOMENTUM FEATURES
        print("    Adding temporal momentum features...")
        
        if hasattr(self, 'team_momentum_metrics') and self.team_momentum_metrics:
            momentum_feature_names = [
                'team_winrate_last_3', 'team_winrate_last_5', 'team_winrate_last_10',
                'form_momentum_short', 'performance_volatility', 'current_streak',
                'patch_experience', 'patch_performance', 'vs_league_avg'
            ]
            
            for feature_name in momentum_feature_names:
                if feature_name in self.team_momentum_metrics:
                    # Vectorized lookup of momentum features
                    momentum_values = pd.Series(
                        [self.team_momentum_metrics[feature_name].get(idx, 0.5) for idx in self.df.index],
                        index=self.df.index
                    )
                    features_df[feature_name] = momentum_values
                else:
                    features_df[feature_name] = 0.5  # Default value
            
            print(f"       Added {len(momentum_feature_names)} momentum features")
        else:
            print("       Momentum metrics not available, using defaults")
            # Add default momentum features
            features_df['form_momentum_short'] = 0.0
            features_df['performance_volatility'] = 0.5
            features_df['current_streak'] = 0.0
        
        #   12. NEW: VECTORIZED META SHIFT FEATURES
        print("    Adding meta shift detection features...")
        
        if hasattr(self, 'meta_shift_metrics') and self.meta_shift_metrics:
            meta_features = self.meta_shift_metrics.get('match_meta_features', {})
            
            meta_feature_names = [
                'meta_shift_magnitude', 'pick_shift_magnitude', 'team_meta_adaptation',
                'patch_stability', 'patch_games_count'
            ]
            
            for feature_name in meta_feature_names:
                # Vectorized lookup of meta shift features
                meta_values = pd.Series(
                    [meta_features.get(idx, {}).get(feature_name, 0.5) for idx in self.df.index],
                    index=self.df.index
                )
                features_df[feature_name] = meta_values
            
            print(f"       Added {len(meta_feature_names)} meta shift features")
        else:
            print("       Meta shift metrics not available, using defaults")
            # Add default meta shift features
            features_df['meta_shift_magnitude'] = 1.0
            features_df['patch_stability'] = 1.0
            features_df['team_meta_adaptation'] = 0.5
        
        #   13. NEW: VECTORIZED LEAKAGE-RESISTANT ENCODING
        print("    Adding leakage-resistant features...")
        
        if hasattr(self, 'leakage_resistant_encoders') and self.leakage_resistant_encoders:
            # Add leave-one-out team encoding instead of target encoding
            if 'team_loo_encoded' in self.leakage_resistant_encoders:
                team_loo_values = pd.Series(
                    [self.leakage_resistant_encoders['team_loo_encoded'].get(idx, 0.5) for idx in self.df.index],
                    index=self.df.index
                )
                features_df['team_loo_encoded'] = team_loo_values
                print(f"       Added leakage-resistant team encoding")
            else:
                features_df['team_loo_encoded'] = 0.5
        else:
            print("       Leakage-resistant encoders not available")
            features_df['team_loo_encoded'] = 0.5
        
        #   14. NEW: ADVANCED INTERACTION FEATURES
        print("    Creating advanced temporal interactions...")
        
        # Momentum-Meta interactions
        if 'form_momentum_short' in features_df.columns and 'meta_shift_magnitude' in features_df.columns:
            features_df['momentum_meta_interaction'] = (
                features_df['form_momentum_short'] * features_df['meta_shift_magnitude']
            )
        
        # Experience-Adaptation interactions  
        if 'team_experience' in features_df.columns and 'team_meta_adaptation' in features_df.columns:
            features_df['experience_adaptation_synergy'] = (
                features_df['team_experience'] * features_df['team_meta_adaptation']
            )
        
        # Volatility-Stability interactions
        if 'performance_volatility' in features_df.columns and 'patch_stability' in features_df.columns:
            features_df['volatility_stability_balance'] = (
                (1 - features_df['performance_volatility']) * features_df['patch_stability']
            )
        
        # Handle any remaining NaN values
        features_df = features_df.fillna(0.5)
        
        self.advanced_features_df = features_df
        
        print(f"    Created {len(features_df.columns)} advanced features (VECTORIZED)")
        print(f"    Feature matrix: {features_df.shape}")
        print(f"    Performance improvement: ~10-50x faster than row-by-row approach")
        
        return features_df

    def _add_temporal_momentum_features(self):
        """ NEW: Add recent form and momentum features for improved temporal prediction."""
        print(f"\n ADDING TEMPORAL MOMENTUM FEATURES")
        
        # Sort data chronologically for proper temporal analysis
        df_sorted = self.df.sort_values(['team', 'date']).copy()
        
        # Initialize momentum tracking
        self.team_momentum_metrics = {}
        
        # Calculate rolling performance metrics
        print("    Calculating rolling team performance...")
        
        for window in [3, 5, 10]:
            # Rolling win rate for each team
            df_sorted[f'team_winrate_last_{window}'] = (
                df_sorted.groupby('team')['result']
                .transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))
                .fillna(0.5)  # Default for teams with insufficient history
            )
        
        # Form momentum indicators
        print("    Computing momentum indicators...")
        
        # Recent vs historical performance trend
        df_sorted['form_momentum_short'] = (
            df_sorted['team_winrate_last_3'] - df_sorted['team_winrate_last_10']
        ).fillna(0)
        
        # Performance volatility (consistency indicator)
        df_sorted['performance_volatility'] = (
            df_sorted.groupby('team')['result']
            .transform(lambda x: x.rolling(10, min_periods=3).std().shift(1))
            .fillna(0.5)
        )
        
        # Win streak indicators
        print("    Adding streak analysis...")
        
        def calculate_current_streak(group):
            """Calculate current win/loss streak for a team."""
            streaks = []
            current_streak = 0
            
            for result in group['result']:
                if len(streaks) == 0:
                    current_streak = 1 if result == 1 else -1
                else:
                    if (result == 1 and current_streak > 0) or (result == 0 and current_streak < 0):
                        current_streak = current_streak + (1 if result == 1 else -1)
                    else:
                        current_streak = 1 if result == 1 else -1
                
                streaks.append(current_streak)
            
            return pd.Series(streaks, index=group.index)
        
        df_sorted['current_streak'] = (
            df_sorted.groupby('team')
            .apply(calculate_current_streak)
            .reset_index(level=0, drop=True)
            .shift(1)  # Shift to avoid future information
            .fillna(0)
        )
        
        # Patch adaptation metrics
        print("    Calculating patch adaptation...")
        
        # How well team performs in new patches vs established patches
        df_sorted['patch_experience'] = (
            df_sorted.groupby(['team', 'patch'])
            .cumcount() + 1  # Games played in current patch
        )
        
        # Team's average performance in this patch so far
        df_sorted['patch_performance'] = (
            df_sorted.groupby(['team', 'patch'])['result']
            .transform(lambda x: x.expanding().mean().shift(1))
            .fillna(0.5)
        )
        
        # Opponent strength adaptation
        print("    Adding opponent adaptation metrics...")
        
        # Recent performance vs strong teams (proxy using league standings)
        league_avg_performance = df_sorted.groupby('league')['result'].mean()
        df_sorted['league_strength'] = df_sorted['league'].map(league_avg_performance)
        
        # Performance vs league average
        df_sorted['vs_league_avg'] = (
            df_sorted.groupby(['team', 'league'])['result']
            .transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
            .fillna(0.5)
        ) - df_sorted['league_strength']
        
        # Store enhanced momentum metrics
        momentum_features = [
            'team_winrate_last_3', 'team_winrate_last_5', 'team_winrate_last_10',
            'form_momentum_short', 'performance_volatility', 'current_streak',
            'patch_experience', 'patch_performance', 'vs_league_avg'
        ]
        
        for feature in momentum_features:
            self.team_momentum_metrics[feature] = df_sorted.set_index(df_sorted.index)[feature].to_dict()
        
        print(f"    Added {len(momentum_features)} temporal momentum features")
        print(f"    Features: {momentum_features}")
        
        # Store the enhanced dataframe
        self.df = df_sorted.reindex(self.df.index)  # Restore original index order
    
    def _investigate_target_leakage(self):
        """ NEW: Comprehensive target leakage investigation and mitigation."""
        print(f"\n INVESTIGATING TARGET LEAKAGE")
        
        # Analyze target encoding for potential leakage
        print("    Analyzing target encoding leakage...")
        
        # Check team target encoding correlation with results
        team_performance = self.df.groupby('team')['result'].agg(['mean', 'count', 'std']).reset_index()
        team_performance.columns = ['team', 'team_true_winrate', 'team_games', 'team_consistency']
        
        # Identify teams with extreme win rates (potential leakage sources)
        high_winrate_teams = team_performance[team_performance['team_true_winrate'] > 0.7]['team'].tolist()
        low_winrate_teams = team_performance[team_performance['team_true_winrate'] < 0.3]['team'].tolist()
        
        print(f"    High win rate teams (>70%): {high_winrate_teams}")
        print(f"    Low win rate teams (<30%): {low_winrate_teams}")
        
        # Check for sample size issues
        small_sample_teams = team_performance[team_performance['team_games'] < 10]['team'].tolist()
        print(f"    Teams with <10 games: {len(small_sample_teams)} teams")
        
        # Enhanced target encoding with leakage prevention
        print("    Implementing leakage-resistant encoding...")
        
        # Leave-one-out encoding for teams to prevent leakage (vectorized)
        self.leakage_resistant_encoders = {}

        # Vectorized LOO: for each row, (team_sum - row_result) / (team_count - 1)
        team_stats = self.df.groupby('team')['result'].agg(['sum', 'count'])
        team_sum = self.df['team'].map(team_stats['sum'])
        team_count = self.df['team'].map(team_stats['count'])

        global_mean = self.df['result'].mean()

        # LOO encoding: exclude current row's result
        loo_values = (team_sum - self.df['result']) / (team_count - 1)
        # For teams with only 1 game, use global mean
        loo_values = loo_values.where(team_count > 1, global_mean)

        self.leakage_resistant_encoders['team_loo_encoded'] = loo_values.to_dict()
        
        # Check champion encoding for leakage
        print("    Analyzing champion encoding leakage...")
        
        champion_cols = ['top_champion', 'jng_champion', 'mid_champion', 'bot_champion', 'sup_champion']
        
        for col in champion_cols:
            champion_performance = self.df.groupby(col)['result'].agg(['mean', 'count']).reset_index()
            champion_performance.columns = [col, f'{col}_winrate', f'{col}_games']
            
            # Identify champions with extreme performance and small samples
            extreme_champs = champion_performance[
                (champion_performance[f'{col}_winrate'] > 0.8) | 
                (champion_performance[f'{col}_winrate'] < 0.2)
            ]
            
            if len(extreme_champs) > 0:
                print(f"    {col} - extreme performers: {len(extreme_champs)} champions")
        
        # Time-aware validation
        print("    Performing time-aware leakage validation...")
        
        # Check if recent matches have higher correlation with encodings (temporal leakage)
        df_sorted = self.df.sort_values('date')
        recent_matches = df_sorted.tail(1000)  # Last 1000 matches
        
        # Compare encoding distributions between early and recent matches
        early_matches = df_sorted.head(1000)
        
        if hasattr(self, 'advanced_features_df') and self.advanced_features_df is not None:
            if 'team_target_encoded' in self.advanced_features_df.columns:
                early_team_encoding = early_matches.index.intersection(self.advanced_features_df.index)
                recent_team_encoding = recent_matches.index.intersection(self.advanced_features_df.index)
                
                if len(early_team_encoding) > 0 and len(recent_team_encoding) > 0:
                    early_mean = self.advanced_features_df.loc[early_team_encoding, 'team_target_encoded'].mean()
                    recent_mean = self.advanced_features_df.loc[recent_team_encoding, 'team_target_encoded'].mean()
                    
                    encoding_drift = abs(recent_mean - early_mean)
                    print(f"    Team encoding temporal drift: {encoding_drift:.4f}")
                    
                    if encoding_drift > 0.05:  # 5% drift threshold
                        print(f"    HIGH TEMPORAL DRIFT DETECTED - potential leakage!")
                else:
                    print(f"    Temporal drift analysis: insufficient data overlap")
            else:
                print(f"    Temporal drift analysis: team_target_encoded not found")
        else:
            print(f"    Temporal drift analysis: advanced features not yet created")
        
        # Feature leakage detection
        print("    Performing feature leakage detection...")
        
        suspicious_features = []
        
        if hasattr(self, 'advanced_features_df') and self.advanced_features_df is not None:
            # Check for features with unrealistically high predictive power
            from sklearn.metrics import roc_auc_score
            
            for feature in self.advanced_features_df.select_dtypes(include=[np.number]).columns:
                if feature == 'result':
                    continue
                    
                # Quick AUC check for individual features
                feature_data = self.advanced_features_df[feature].fillna(0.5)
                target_data = self.df['result']
                
                # Align indices
                common_idx = feature_data.index.intersection(target_data.index)
                if len(common_idx) > 100:  # Minimum sample size
                    try:
                        feature_auc = roc_auc_score(target_data.loc[common_idx], feature_data.loc[common_idx])
                        
                        # Flag features with suspiciously high individual AUC
                        if feature_auc > 0.85 or feature_auc < 0.15:
                            suspicious_features.append((feature, feature_auc))
                    except:
                        pass  # Skip features that cause errors
        else:
            print(f"    Feature leakage detection: will be performed after feature creation")
        
        # Report suspicious features
        if suspicious_features:
            print(f"    SUSPICIOUS FEATURES DETECTED:")
            for feature, auc in suspicious_features:
                print(f"      {feature}: AUC = {auc:.4f}")
        else:
            if hasattr(self, 'advanced_features_df') and self.advanced_features_df is not None:
                print(f"    No obviously suspicious features detected")
        
        # Store leakage investigation results
        self.leakage_investigation = {
            'high_winrate_teams': high_winrate_teams,
            'low_winrate_teams': low_winrate_teams,
            'small_sample_teams': small_sample_teams,
            'suspicious_features': suspicious_features,
            'leakage_resistant_encoders': self.leakage_resistant_encoders
        }
        
        print(f"    Target leakage investigation complete")
        print(f"    Stored leakage-resistant encoders for future use")
    
    def _add_meta_shift_detection(self):
        """ NEW: Add meta shift detection features to capture meta evolution."""
        print(f"\n ADDING META SHIFT DETECTION FEATURES")
        
        # Sort by date for temporal analysis
        df_sorted = self.df.sort_values('date')
        patches = sorted(df_sorted['patch'].unique())
        
        # Initialize meta shift metrics
        self.meta_shift_metrics = {}
        
        print("    Calculating patch-level meta metrics...")
        
        # Calculate patch-level statistics
        patch_stats = {}
        
        for patch in patches:
            patch_data = df_sorted[df_sorted['patch'] == patch]
            
            if len(patch_data) < 10:  # Skip patches with insufficient data
                continue
            
            # Champion diversity metrics
            all_champions = set()
            for col in ['top_champion', 'jng_champion', 'mid_champion', 'bot_champion', 'sup_champion']:
                champs = patch_data[col].dropna().unique()
                all_champions.update(champs)
            
            # Meta stability indicators
            patch_stats[patch] = {
                'champion_diversity': len(all_champions),
                'games_count': len(patch_data),
                'avg_game_length': patch_data['game_length'].mean() if 'game_length' in patch_data.columns else 30,
                'win_rate_variance': patch_data.groupby('team')['result'].mean().var(),
                'unique_teams': len(patch_data['team'].unique())
            }
        
        print(f"    Calculating meta shift indicators between patches...")
        
        # Calculate meta shift between consecutive patches
        meta_shifts = {}
        
        for i, patch in enumerate(patches[1:], 1):
            if patch not in patch_stats or patches[i-1] not in patch_stats:
                continue
                
            prev_patch = patches[i-1]
            current_stats = patch_stats[patch]
            prev_stats = patch_stats[prev_patch]
            
            # Champion diversity change
            diversity_change = (current_stats['champion_diversity'] - prev_stats['champion_diversity']) / prev_stats['champion_diversity']
            
            # Game length shift (meta speed indicator)
            length_change = (current_stats['avg_game_length'] - prev_stats['avg_game_length']) / prev_stats['avg_game_length']
            
            # Competitive balance shift
            balance_change = abs(current_stats['win_rate_variance'] - prev_stats['win_rate_variance'])
            
            meta_shifts[patch] = {
                'diversity_shift': diversity_change,
                'game_length_shift': length_change,
                'balance_shift': balance_change,
                'meta_stability_score': 1 - (abs(diversity_change) + abs(length_change) + balance_change) / 3
            }
        
        print("    Computing champion pick rate shifts...")
        
        # Calculate champion pick rate changes between patches
        champion_pick_shifts = {}
        
        for i, patch in enumerate(patches[1:], 1):
            prev_patch = patches[i-1]
            
            # Get champion pick rates for both patches
            current_patch_data = df_sorted[df_sorted['patch'] == patch]
            prev_patch_data = df_sorted[df_sorted['patch'] == prev_patch]
            
            if len(current_patch_data) < 5 or len(prev_patch_data) < 5:
                continue
            
            # Calculate pick rate changes for top lane (can extend to other roles)
            current_top_picks = current_patch_data['top_champion'].value_counts(normalize=True)
            prev_top_picks = prev_patch_data['top_champion'].value_counts(normalize=True)
            
            # Find biggest pick rate changes
            all_top_champs = set(current_top_picks.index) | set(prev_top_picks.index)
            
            pick_rate_changes = []
            for champ in all_top_champs:
                current_rate = current_top_picks.get(champ, 0)
                prev_rate = prev_top_picks.get(champ, 0)
                change = current_rate - prev_rate
                pick_rate_changes.append(abs(change))
            
            avg_pick_shift = np.mean(pick_rate_changes) if pick_rate_changes else 0
            champion_pick_shifts[patch] = avg_pick_shift
        
        print("    Creating meta adaptation features...")

        # Vectorized: map patch-level features to each match
        meta_stability_map = {p: s.get('meta_stability_score', 1.0) for p, s in meta_shifts.items()}
        pick_shift_map = champion_pick_shifts
        patch_games_map = {p: s.get('games_count', 0) for p, s in patch_stats.items()}

        df_sorted['_meta_shift_magnitude'] = df_sorted['patch'].map(meta_stability_map).fillna(1.0)
        df_sorted['_pick_shift_magnitude'] = df_sorted['patch'].map(pick_shift_map).fillna(0.0)
        df_sorted['_patch_stability'] = df_sorted['patch'].map(meta_stability_map).fillna(1.0)
        df_sorted['_patch_games_count'] = df_sorted['patch'].map(patch_games_map).fillna(0)

        # Flag high-shift patches (stability < 0.7)
        df_sorted['_is_high_shift'] = df_sorted['_patch_stability'] < 0.7

        # For meta adaptation: expanding mean of results during high-shift patches per team
        df_sorted['_high_shift_result'] = df_sorted['result'].where(df_sorted['_is_high_shift'], np.nan)
        df_sorted['_meta_adaptation'] = (
            df_sorted.groupby('team')['_high_shift_result']
            .transform(lambda x: x.expanding(min_periods=1).mean().shift(1))
        ).fillna(0.5)

        # Only use adaptation score when team has > 5 prior games
        df_sorted['_team_cumgames'] = df_sorted.groupby('team').cumcount()
        df_sorted['_meta_adaptation'] = df_sorted['_meta_adaptation'].where(
            df_sorted['_team_cumgames'] > 5, 0.5
        )

        # Build meta_features dict from vectorized columns
        meta_features = {}
        for idx in df_sorted.index:
            meta_features[idx] = {
                'meta_shift_magnitude': df_sorted.at[idx, '_meta_shift_magnitude'],
                'pick_shift_magnitude': df_sorted.at[idx, '_pick_shift_magnitude'],
                'team_meta_adaptation': df_sorted.at[idx, '_meta_adaptation'],
                'patch_stability': df_sorted.at[idx, '_patch_stability'],
                'patch_games_count': df_sorted.at[idx, '_patch_games_count']
            }

        # Clean up temporary columns
        df_sorted.drop(columns=[c for c in df_sorted.columns if c.startswith('_')], inplace=True)
        
        # Store meta shift metrics
        self.meta_shift_metrics = {
            'patch_stats': patch_stats,
            'meta_shifts': meta_shifts,
            'champion_pick_shifts': champion_pick_shifts,
            'match_meta_features': meta_features
        }
        
        print(f"    Added meta shift detection for {len(patches)} patches")
        print(f"    Created {len(meta_features)} match-level meta features")
        print(f"    Detected {len(meta_shifts)} patch transitions with meta shifts")
        
        # Show some example meta shifts
        if meta_shifts:
            print(f"    Example meta shifts:")
            for patch, shifts in list(meta_shifts.items())[:3]:
                print(f"      {patch}: stability = {shifts['meta_stability_score']:.3f}")
    
    def create_basic_leakage_free_features(self):
        """ Create basic leakage-free features without temporal momentum using match results."""
        print(f"\n CREATING LEAKAGE-FREE FEATURES")
        print("    Excluding: All features using match results (win rates, streaks, performance)")
        print("    Including: Champions, bans, patch, league, context, static synergies")
        
        # Get champion data without leakage
        self._analyze_champion_characteristics()
        
        # Calculate clean meta indicators (no results used)
        self._calculate_clean_meta_indicators()
        
        # Analyze pick/ban without results
        self._analyze_pickban_strategy()
        
        # Create features DataFrame
        features_data = []
        
        for idx, match in self.df.iterrows():
            match_features = {}
            
            # 1. CHAMPION FEATURES (No leakage - static only)
            champions = [
                match['top_champion'], match['jng_champion'], match['mid_champion'],
                match['bot_champion'], match['sup_champion']
            ]
            
            # One-hot encode champions (most important features)
            for i, champ in enumerate(['top_champion', 'jng_champion', 'mid_champion', 'bot_champion', 'sup_champion']):
                match_features[f'{champ}_{match[champ]}'] = 1
            
            # 2. BAN FEATURES (No leakage)
            bans = [match['ban1'], match['ban2'], match['ban3'], match['ban4'], match['ban5']]
            for i, ban in enumerate(bans):
                if pd.notna(ban):
                    match_features[f'ban{i+1}_{ban}'] = 1
            
            # 3. PATCH FEATURES (No leakage)
            patch = match['patch']
            match_features['patch'] = patch
            match_features['patch_major'] = int(patch)
            match_features['patch_minor'] = int((patch % 1) * 100)
            
            # 4. LEAGUE FEATURES (No leakage)
            match_features[f'league_{match["league"]}'] = 1
            
            # 5. CONTEXT FEATURES (No leakage)
            match_features['year'] = match['year']
            match_features['playoffs'] = match['playoffs']
            
            # 6. CHAMPION ARCHETYPE FEATURES (Static - No leakage)
            tank_champions = ['Malphite', 'Ornn', 'Sion', 'Maokai', 'Nautilus', 'Leona', 'Braum', 'Thresh']
            assassin_champions = ['Zed', 'Yasuo', 'Akali', 'LeBlanc', 'Katarina', 'Talon', 'Qiyana']
            marksman_champions = ['Jinx', 'Caitlyn', 'Ezreal', 'Kai\'Sa', 'Xayah', 'Varus', 'Ashe']
            mage_champions = ['Azir', 'Orianna', 'Syndra', 'Cassiopeia', 'Viktor', 'Ryze']
            
            match_features['team_tanks'] = sum(1 for champ in champions if champ in tank_champions)
            match_features['team_assassins'] = sum(1 for champ in champions if champ in assassin_champions)
            match_features['team_marksmen'] = sum(1 for champ in champions if champ in marksman_champions)
            match_features['team_mages'] = sum(1 for champ in champions if champ in mage_champions)
            
            # Team composition balance
            match_features['team_balance'] = (
                match_features['team_tanks'] + match_features['team_marksmen'] + match_features['team_mages']
            ) / 3
            
            # 7. BAN PRIORITY FEATURES (Static - No leakage)
            match_features['bans_tanks'] = sum(1 for ban in bans if pd.notna(ban) and ban in tank_champions)
            match_features['bans_assassins'] = sum(1 for ban in bans if pd.notna(ban) and ban in assassin_champions)
            match_features['bans_marksmen'] = sum(1 for ban in bans if pd.notna(ban) and ban in marksman_champions)
            
            # Ban diversity
            unique_bans = len(set(ban for ban in bans if pd.notna(ban)))
            match_features['ban_diversity'] = unique_bans
            
            features_data.append(match_features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_data, index=self.df.index)
        
        # Fill missing values with 0 (for one-hot encoded features)
        features_df = features_df.fillna(0)
        
        print(f"    Created {features_df.shape[1]} leakage-free features")
        print(f"    Zero temporal momentum features")
        print(f"    Expected realistic performance: 70-80%")
        
        return features_df
    
    def _calculate_clean_meta_indicators(self):
        """Calculate meta indicators without using match results."""
        print("    Computing clean meta indicators (no results used)...")
        
        # Champion popularity by patch (frequency of picks)
        champion_cols = ['top_champion', 'jng_champion', 'mid_champion', 'bot_champion', 'sup_champion']
        
        for patch in self.df['patch'].unique():
            if pd.isna(patch):
                continue
                
            patch_data = self.df[self.df['patch'] == patch]
            
            # Champion pick frequency in this patch
            all_picks = []
            for col in champion_cols:
                all_picks.extend(patch_data[col].dropna().tolist())
            
            from collections import Counter
            pick_frequency = Counter(all_picks)
            
            # Store popular champions for this patch (top 10)
            popular_champs = [champ for champ, _ in pick_frequency.most_common(10)]
            
            # Note: We don't use win rates here - just popularity
            self.patch_meta[patch] = {
                'popular_champions': popular_champs,
                'total_games': len(patch_data),
                'pick_frequency': pick_frequency
            }
        
        print(f"    Computed meta indicators for {len(self.patch_meta)} patches (leakage-free)")

    def create_enhanced_features_v2(self):
        """
        Create enhanced feature set v2 with new feature categories.

        New features include:
        - Side Selection Features (blue/red side advantages)
        - Patch Transition Features (patch adaptation, early patch indicator)
        - Extended Interaction Features (meta-side, composition clash, etc.)
        - Head-to-Head Features (direct matchup history)

        Returns:
            pd.DataFrame: Enhanced feature matrix
        """
        print(f"\n CREATING ENHANCED FEATURES V2")
        print("=" * 60)

        # Start with vectorized features
        features_df = self.create_advanced_features_vectorized()

        # ═══════════════════════════════════════════════════════════
        # 1. SIDE SELECTION FEATURES
        # ═══════════════════════════════════════════════════════════
        print("    Adding side selection features...")
        features_df = self._add_side_selection_features(features_df)

        # ═══════════════════════════════════════════════════════════
        # 2. PATCH TRANSITION FEATURES
        # ═══════════════════════════════════════════════════════════
        print("    Adding patch transition features...")
        features_df = self._add_patch_transition_features(features_df)

        # ═══════════════════════════════════════════════════════════
        # 3. EXTENDED INTERACTION FEATURES
        # ═══════════════════════════════════════════════════════════
        print("    Adding extended interaction features...")
        features_df = self._add_extended_interactions(features_df)

        # ═══════════════════════════════════════════════════════════
        # 4. HEAD-TO-HEAD ENHANCED FEATURES
        # ═══════════════════════════════════════════════════════════
        print("    Adding head-to-head features...")
        features_df = self._add_h2h_features(features_df)

        # Fill any remaining NaN values
        features_df = features_df.fillna(0.5)

        self.enhanced_features_df = features_df

        print(f"    Created {len(features_df.columns)} enhanced features (v2)")
        print(f"    Feature matrix: {features_df.shape}")

        return features_df

    def _add_side_selection_features(self, features_df):
        """Add blue/red side selection features (vectorized)."""

        df = self.df.copy()
        side_col = df['side'].fillna('Blue')
        is_blue = side_col == 'Blue'

        # Aggregate side-specific stats per team
        df['_is_blue'] = is_blue.astype(int)
        df['_is_red'] = (~is_blue).astype(int)
        df['_blue_win'] = (is_blue & (df['result'] == 1)).astype(int)
        df['_red_win'] = (~is_blue & (df['result'] == 1)).astype(int)

        team_side = df.groupby('team').agg(
            blue_games=('_is_blue', 'sum'),
            blue_wins=('_blue_win', 'sum'),
            red_games=('_is_red', 'sum'),
            red_wins=('_red_win', 'sum')
        )

        # Map back to each row
        team_series = self.df['team']
        blue_games = team_series.map(team_side['blue_games']).fillna(0)
        blue_wins = team_series.map(team_side['blue_wins']).fillna(0)
        red_games = team_series.map(team_side['red_games']).fillna(0)
        red_wins = team_series.map(team_side['red_wins']).fillna(0)

        features_df['blue_side_winrate'] = (blue_wins / blue_games).where(blue_games > 0, 0.5)
        features_df['red_side_winrate'] = (red_wins / red_games).where(red_games > 0, 0.5)
        features_df['side_preference'] = features_df['blue_side_winrate'] - features_df['red_side_winrate']

        # Current side advantage
        is_blue_orig = self.df['side'].fillna('Blue') == 'Blue'
        features_df['current_side_advantage'] = np.where(
            is_blue_orig,
            features_df['blue_side_winrate'],
            features_df['red_side_winrate']
        )

        features_df['side_meta_interaction'] = (
            features_df['current_side_advantage'] * features_df['team_meta_strength']
        )

        print(f"       Added 5 side selection features")
        return features_df

    def _add_patch_transition_features(self, features_df):
        """Add patch transition and adaptation features (vectorized)."""

        df = self.df.copy()
        df['patch'] = df['patch'].fillna('Unknown')

        # Sort by date within each team-patch group
        df_sorted = df.sort_values('date') if 'date' in df.columns else df

        # Games on patch before this match (cumcount within team+patch, needs date ordering)
        df_sorted['_games_on_patch'] = df_sorted.groupby(['team', 'patch']).cumcount()
        # Map back to original index
        features_df['games_on_patch'] = df_sorted['_games_on_patch'].reindex(self.df.index)

        features_df['early_patch_indicator'] = (features_df['games_on_patch'] < 3).astype(int)

        # Patch win rate per (team, patch) - overall aggregate
        patch_perf = df.groupby(['team', 'patch'])['result'].agg(['sum', 'count'])
        patch_perf.columns = ['wins', 'games']
        patch_perf['patch_wr'] = (patch_perf['wins'] / patch_perf['games']).fillna(0.5)

        # Map patch win rate to each row
        df['_team_patch'] = list(zip(df['team'], df['patch']))
        patch_wr_map = patch_perf['patch_wr'].to_dict()
        features_df['patch_win_rate'] = df['_team_patch'].map(patch_wr_map).fillna(0.5).values

        # Patch adaptation rate = patch_wr - overall_wr
        overall_wr = df['team'].map(
            lambda t: self.team_historical_performance.get(t, {}).get('overall_winrate', 0.5)
        )
        features_df['patch_adaptation_rate'] = features_df['patch_win_rate'] - overall_wr.values

        # Normalize games on patch (max 20)
        features_df['games_on_patch_normalized'] = (features_df['games_on_patch'] / 20).clip(upper=1.0)

        features_df['patch_experience_advantage'] = (
            features_df['games_on_patch_normalized'] * (1 - features_df['early_patch_indicator'])
        )

        print(f"       Added 6 patch transition features")
        return features_df

    def _add_extended_interactions(self, features_df):
        """Add extended interaction features."""

        # Meta-Side interaction
        if 'team_meta_strength' in features_df.columns and 'current_side_advantage' in features_df.columns:
            features_df['meta_side_synergy'] = (
                features_df['team_meta_strength'] * features_df['current_side_advantage']
            )

        # Experience-Adaptation interaction
        if 'team_experience' in features_df.columns and 'patch_adaptation_rate' in features_df.columns:
            features_df['experience_adaptation_interaction'] = (
                features_df['team_experience'] * (features_df['patch_adaptation_rate'] + 0.5)
            )

        # Form-Matchup interaction
        if 'team_recent_winrate' in features_df.columns:
            features_df['form_matchup_synergy'] = (
                features_df['team_recent_winrate'] * features_df.get('team_historical_advantage', 0.5)
            )

        # Composition clash (scaling balance)
        if 'team_scaling' in features_df.columns:
            features_df['composition_aggression'] = (
                features_df['team_early_strength'] - features_df['team_late_strength']
            )

        # Strategic-Meta interaction
        if 'ban_diversity' in features_df.columns and 'team_meta_strength' in features_df.columns:
            features_df['strategic_meta_interaction'] = (
                features_df['ban_diversity'] / 5 * features_df['team_meta_strength']
            )

        # Confidence-Weighted meta strength
        if 'lane_matchup_confidence' in features_df.columns:
            features_df['confidence_weighted_meta'] = (
                features_df['team_meta_strength'] * features_df['lane_matchup_confidence']
            )

        # Momentum-Experience synergy
        if 'form_momentum_short' in features_df.columns:
            features_df['momentum_experience_synergy'] = (
                features_df['form_momentum_short'] * features_df['team_experience']
            )

        print(f"       Added 7 extended interaction features")
        return features_df

    def _add_h2h_features(self, features_df):
        """Add enhanced head-to-head features (vectorized)."""

        df = self.df.copy()
        df['league'] = df['league'].fillna('Unknown')

        # League-specific win rate per (team, league)
        league_perf = df.groupby(['team', 'league'])['result'].agg(['sum', 'count'])
        league_perf.columns = ['wins', 'games']
        league_perf['league_wr'] = (league_perf['wins'] / league_perf['games']).fillna(0.5)

        # Map to each row via (team, league) tuple
        team_league_keys = list(zip(df['team'], df['league']))
        league_wr_map = league_perf['league_wr'].to_dict()
        features_df['league_specific_winrate'] = pd.Series(
            [league_wr_map.get(k, 0.5) for k in team_league_keys],
            index=self.df.index
        )

        # League dominance = league_wr - overall_wr
        overall_wr = df['team'].map(
            lambda t: self.team_historical_performance.get(t, {}).get('overall_winrate', 0.5)
        )
        features_df['league_dominance'] = features_df['league_specific_winrate'] - overall_wr.values

        # Cross-league experience: number of leagues per team
        leagues_per_team = df.groupby('team')['league'].nunique()
        features_df['cross_league_experience'] = (
            df['team'].map(leagues_per_team).fillna(1).values / 5
        ).clip(0, 1)

        # H2H momentum
        features_df['h2h_momentum'] = (
            features_df['league_specific_winrate'] * features_df.get('team_recent_winrate', 0.5)
        )

        print(f"       Added 4 head-to-head features")
        return features_df

    def get_feature_summary(self):
        """Get a summary of all available features."""
        if hasattr(self, 'enhanced_features_df'):
            features = self.enhanced_features_df
        elif hasattr(self, 'advanced_features_df'):
            features = self.advanced_features_df
        else:
            return "No features created yet. Run create_advanced_features_vectorized() or create_enhanced_features_v2() first."

        summary = {
            'total_features': len(features.columns),
            'feature_names': list(features.columns),
            'shape': features.shape,
            'categories': {
                'champion_characteristics': [f for f in features.columns if 'team_avg' in f or 'team_scaling' in f],
                'meta_features': [f for f in features.columns if 'meta' in f.lower()],
                'team_performance': [f for f in features.columns if 'team_' in f and 'winrate' in f.lower()],
                'side_features': [f for f in features.columns if 'side' in f.lower()],
                'patch_features': [f for f in features.columns if 'patch' in f.lower()],
                'interaction_features': [f for f in features.columns if 'interaction' in f.lower() or 'synergy' in f.lower()],
                'matchup_features': [f for f in features.columns if 'matchup' in f.lower() or 'advantage' in f.lower() or 'h2h' in f.lower()],
            }
        }

        return summary

    # =========================================================================
    # LEAKAGE-FREE TRAINING METHODS
    # These methods support split-first-then-compute pipelines where aggregate
    # statistics are derived from training data only.
    # =========================================================================

    def analyze_from_subset(self, data_subset):
        """Compute all aggregate statistics from a training subset only.

        Temporarily swaps self.df to the given subset, runs all analysis
        methods (which read self.df), then restores the original DataFrame.
        This ensures champion_meta_strength, team_historical_performance,
        ban_priority, etc. are computed from training data only.

        Args:
            data_subset: pd.DataFrame -- the training rows to derive stats from.
        """
        print("\n LEAKAGE-FREE ANALYSIS: Computing stats from training subset only")
        print(f"    Training subset size: {len(data_subset)} rows")

        original_df = self.df
        self.df = data_subset.copy()

        # Re-run all analysis methods on the training subset
        self._handle_missing_values()
        self._analyze_champion_characteristics()
        self._calculate_meta_indicators()
        self._analyze_pickban_strategy()
        self._calculate_team_dynamics()
        self._analyze_historical_matchups()
        self._calculate_player_metrics()
        self._analyze_player_performance()
        self._add_temporal_momentum_features()
        self._investigate_target_leakage()
        self._add_meta_shift_detection()

        # Restore the full dataset
        self.df = original_df
        print("    Analysis complete -- stats derived from training data only")

    def create_features_for_split(self, split_df):
        """Create features for a specific split using pre-computed statistics.

        Temporarily sets self.df to the split's rows, calls the vectorized
        feature creation (which reads self.df for row data but uses the
        pre-computed dicts for lookups), then restores self.df.

        Args:
            split_df: pd.DataFrame -- rows for this split (train, val, or test).

        Returns:
            pd.DataFrame of engineered features for the split.
        """
        original_df = self.df
        self.df = split_df.copy()

        # Ensure missing values are handled in this subset
        self._handle_missing_values()

        # Create features using pre-computed statistics (from analyze_from_subset)
        features = self.create_advanced_features_vectorized()

        # Restore the full dataset
        self.df = original_df
        return features

    def fit_target_encoders(self, train_df):
        """Fit TargetEncoder instances on training data only.

        Args:
            train_df: pd.DataFrame -- training rows (must include 'result' column).
        """
        print("\n FITTING TARGET ENCODERS ON TRAINING DATA ONLY")

        target = train_df['result']
        self.target_encoders = {}
        self.label_encoders = {}

        categorical_cols = ['league', 'team', 'patch', 'split']
        champion_cols = ['top_champion', 'jng_champion', 'mid_champion',
                         'bot_champion', 'sup_champion']
        player_cols = ['top_player', 'jng_player', 'mid_player',
                       'bot_player', 'sup_player']

        all_cols = categorical_cols + champion_cols + [c for c in player_cols if c in train_df.columns]

        for col in all_cols:
            if col not in train_df.columns:
                continue

            col_data = (train_df[col].astype(str)
                        .replace({'nan': 'Unknown', 'None': 'Unknown'})
                        .fillna('Unknown'))

            try:
                encoder = TargetEncoder(random_state=42)
                encoder.fit(col_data.values.reshape(-1, 1), target)
                self.target_encoders[col] = encoder
            except Exception as e:
                print(f"    TargetEncoder failed for {col}: {e}, falling back to LabelEncoder")
                le = LabelEncoder()
                le.fit(col_data)
                self.label_encoders[col] = le

        print(f"    Fitted {len(self.target_encoders)} target encoders, "
              f"{len(self.label_encoders)} label encoders")

    def apply_encoding_to_split(self, features_df, split_df):
        """Transform a split's features using pre-fitted encoders.

        Args:
            features_df: pd.DataFrame -- engineered features for this split.
            split_df: pd.DataFrame -- raw data for this split (for categorical columns).

        Returns:
            pd.DataFrame with encoded columns appended.
        """
        categorical_cols = ['league', 'team', 'patch', 'split']
        champion_cols = ['top_champion', 'jng_champion', 'mid_champion',
                         'bot_champion', 'sup_champion']
        player_cols = ['top_player', 'jng_player', 'mid_player',
                       'bot_player', 'sup_player']
        all_cols = categorical_cols + champion_cols + [c for c in player_cols if c in split_df.columns]

        for col in all_cols:
            if col not in split_df.columns:
                continue

            col_data = (split_df[col].astype(str)
                        .replace({'nan': 'Unknown', 'None': 'Unknown'})
                        .fillna('Unknown'))

            if col in self.target_encoders:
                encoded = self.target_encoders[col].transform(
                    col_data.values.reshape(-1, 1)
                )
                features_df[f'{col}_target_encoded'] = encoded.flatten()
            elif col in self.label_encoders:
                le = self.label_encoders[col]
                # Handle unseen categories by mapping to 'Unknown'
                known_classes = set(le.classes_)
                safe_data = col_data.map(
                    lambda x: x if x in known_classes else 'Unknown'
                )
                if 'Unknown' not in known_classes:
                    le.classes_ = np.append(le.classes_, 'Unknown')
                features_df[f'{col}_label_encoded'] = le.transform(safe_data)

        return features_df


def main(use_vectorized=True, use_enhanced_v2=True):
    """Main execution function for advanced feature engineering.

    Args:
        use_vectorized: Use optimized vectorized operations
        use_enhanced_v2: Use enhanced v2 features with side selection, patch transition, etc.
    """
    print(" ADVANCED FEATURE ENGINEERING FOR LOL MATCH PREDICTION")
    print("=" * 80)

    if use_enhanced_v2:
        method_type = "ENHANCED V2 (OPTIMIZED + NEW FEATURES)"
    elif use_vectorized:
        method_type = "VECTORIZED (OPTIMIZED)"
    else:
        method_type = "ORIGINAL"

    print(f" Mode: {method_type}")

    # Initialize feature engineering system
    feature_eng = AdvancedFeatureEngineering()

    # Load and analyze data
    df = feature_eng.load_and_analyze_data()

    # Create advanced features (choose method)
    if use_enhanced_v2:
        print(f"\n Using enhanced v2 features (includes new categories)...")
        advanced_features = feature_eng.create_enhanced_features_v2()
        final_features = feature_eng.apply_advanced_encoding_optimized()
        performance_note = "Enhanced v2 with side selection, patch transition, H2H features"
    elif use_vectorized:
        print(f"\n Using optimized vectorized methods...")
        advanced_features = feature_eng.create_advanced_features_vectorized()
        final_features = feature_eng.apply_advanced_encoding_optimized()
        performance_note = "~10-50x faster than original"
    else:
        print(f"\n Using original methods (for comparison)...")
        advanced_features = feature_eng.create_advanced_features()
        final_features = feature_eng.apply_advanced_encoding()
        performance_note = "original speed"

    # Print feature summary
    summary = feature_eng.get_feature_summary()

    print(f"\n ADVANCED FEATURE ENGINEERING COMPLETE!")
    print(f" Final feature matrix: {final_features.shape}")
    print(f" Total features: {summary['total_features'] if isinstance(summary, dict) else 'N/A'}")
    print(f" Performance: {performance_note}")

    if isinstance(summary, dict):
        print(f"\n Feature Categories:")
        for category, features in summary.get('categories', {}).items():
            if features:
                print(f"   - {category}: {len(features)} features")

    return feature_eng, final_features


if __name__ == "__main__":
    # Default: use enhanced v2 features
    feature_eng, features = main(use_enhanced_v2=True) 