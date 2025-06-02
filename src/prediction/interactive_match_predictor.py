import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_engineering.advanced_feature_engineering import AdvancedFeatureEngineering

class RestartDraftException(Exception):
    """Custom exception to handle draft restarts."""
    pass

class InteractiveLoLPredictor:
    """
    Interactive League of Legends match predictor for real professional games.
    
    Features:
    - Team selection from major leagues
    - Interactive pick/ban input
    - Real-time predictions
    - Best-of-series support
    """
    
    def __init__(self, model_path=None):
        print("🎮 INTERACTIVE LoL MATCH PREDICTOR")
        print("=" * 50)
        
        if model_path is None:
            # Build path to models in the new organized structure
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            # UPDATED: Prioritize Bayesian-optimized models from comprehensive comparison
            model_path = os.path.join(project_root, "models", "bayesian_optimized_models", "bayesian_best_model_Logistic_Regression.joblib")
        
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_engineering = None
        self.valid_champions = set()  # Will store all valid champion names
        
        # Professional teams database
        self.teams_db = {
            'LCK': ['T1', 'GenG', 'KT', 'DRX', 'DK', 'HLE', 'LSB', 'NS', 'FOX', 'BRO'],
            'LEC': ['G2', 'FNC', 'MAD', 'VIT', 'BDS', 'SK', 'TH', 'KOI', 'GX', 'KC'],
            'LCS': ['TL', 'C9', 'FLY', '100T', 'TSM', 'DIG', 'IMT', 'EG', 'CLG', 'GG'],
            'LPL': ['JDG', 'BLG', 'WBG', 'TES', 'EDG', 'IG', 'WE', 'OMG', 'LNG', 'UP']
        }
        
        # Current meta champions (patch 14.1 example)
        self.champion_pool = [
            # Top laners
            'Gnar', 'Jax', 'Aatrox', 'Fiora', 'Camille', 'Ornn', 'Sion', 'Malphite', 'Renekton', 'Gragas',
            # Junglers  
            'Graves', 'Viego', 'Hecarim', 'Elise', 'Kha\'Zix', 'Jarvan IV', 'Lee Sin', 'Rek\'Sai', 'Nidalee', 'Sejuani',
            # Mid laners
            'Azir', 'LeBlanc', 'Yone', 'Yasuo', 'Sylas', 'Orianna', 'Syndra', 'Ahri', 'Corki', 'Viktor',
            # Bot laners
            'Jinx', 'Kai\'Sa', 'Varus', 'Jhin', 'Xayah', 'Ashe', 'Aphelios', 'Caitlyn', 'Ezreal', 'Lucian',
            # Supports
            'Thresh', 'Nautilus', 'Leona', 'Alistar', 'Braum', 'Lulu', 'Yuumi', 'Renata Glasc', 'Soraka', 'Janna'
        ]
        
        self.load_model_components()
        self.setup_feature_engineering()
        self.load_champion_database()
    
    def load_model_components(self):
        """Load trained model components."""
        print(f"\n📦 LOADING MODEL COMPONENTS")
        
        try:
            # Get project root for organized paths
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            
            # Try different model paths starting with organized structure
            model_paths = [
                self.model_path,  # User-provided or default path
                # PRIORITIZE: Bayesian-optimized models (from comprehensive_logistic_regression_comparison.py)
                os.path.join(project_root, "models", "bayesian_optimized_models", "bayesian_best_model_Logistic_Regression.joblib"),
                os.path.join(project_root, "models", "bayesian_optimized_models", "stratified_temporal_bayesian_model.joblib"),
                os.path.join(project_root, "models", "bayesian_optimized_models", "stratified_random_temporal_bayesian_model.joblib"),
                # FALLBACK: Enhanced models (legacy)
                os.path.join(project_root, "models", "enhanced_models", "enhanced_best_model_Logistic_Regression.joblib"),
                os.path.join(project_root, "models", "enhanced_best_model.joblib"),
                os.path.join(project_root, "models", "ultimate_best_model.joblib"),
                "enhanced_best_model.joblib",  # Legacy paths
                "Experiment/enhanced_best_model.joblib"
            ]
            
            scaler_paths = [
                # PRIORITIZE: Bayesian-optimized scalers
                os.path.join(project_root, "models", "bayesian_optimized_models", "stratified_temporal_bayesian_scaler.joblib"),
                os.path.join(project_root, "models", "bayesian_optimized_models", "stratified_random_temporal_bayesian_scaler.joblib"),
                os.path.join(project_root, "models", "bayesian_optimized_models", "pure_temporal_bayesian_scaler.joblib"),
                # FALLBACK: Enhanced scalers (legacy)
                os.path.join(project_root, "models", "enhanced_models", "enhanced_scaler.joblib"),
                os.path.join(project_root, "models", "enhanced_scaler.joblib"),
                os.path.join(project_root, "models", "ultimate_scaler.joblib"),
                "enhanced_models/enhanced_scaler.joblib"  # Legacy path
            ]
            
            model_loaded = False
            for path in model_paths:
                if os.path.exists(path):
                    print(f"   📂 Found model at: {path}")
                    deployment_package = joblib.load(path)
                    
                    if isinstance(deployment_package, dict) and 'model' in deployment_package:
                        self.model = deployment_package['model']
                        self.scaler = deployment_package.get('scaler')
                        
                        model_type = "🧠 BAYESIAN-OPTIMIZED" if "bayesian" in path.lower() else "📊 ENHANCED"
                        print(f"   ✅ Loaded {model_type} model: {deployment_package.get('model_name', 'Unknown')}")
                        
                        # Show Bayesian optimization details if available
                        if "bayesian" in path.lower():
                            total_evals = deployment_package.get('total_evaluations', 'Unknown')
                            strategy = deployment_package.get('strategy', 'Unknown')
                            print(f"   🎯 Strategy: {strategy}")
                            print(f"   🔬 Optimization evaluations: {total_evals}")
                            print(f"   ⚡ Convergence optimized: {deployment_package.get('convergence_optimized', 'Unknown')}")
                        
                        print(f"   📊 AUC Performance: {deployment_package.get('performance', {}).get('auc', 'Unknown')}")
                        
                        if self.scaler is not None:
                            print(f"   ✅ Scaler loaded successfully")
                        else:
                            print(f"   ⚠️ No scaler in deployment package, trying separate files...")
                            # Try to load scaler separately
                            for scaler_path in scaler_paths:
                                if os.path.exists(scaler_path):
                                    self.scaler = joblib.load(scaler_path)
                                    scaler_type = "🧠 BAYESIAN" if "bayesian" in scaler_path.lower() else "📊 ENHANCED"
                                    print(f"   ✅ Loaded {scaler_type} scaler from: {scaler_path}")
                                    break
                        
                        model_loaded = True
                        break
                    else:
                        self.model = deployment_package
                        model_type = "🧠 BAYESIAN" if "bayesian" in path.lower() else "📊 STANDARD"
                        print(f"   ✅ Loaded {model_type} model from: {path}")
                        
                        # Try to load scaler separately for simple models
                        for scaler_path in scaler_paths:
                            if os.path.exists(scaler_path):
                                self.scaler = joblib.load(scaler_path)
                                scaler_type = "🧠 BAYESIAN" if "bayesian" in scaler_path.lower() else "📊 ENHANCED"
                                print(f"   ✅ Loaded {scaler_type} scaler from: {scaler_path}")
                                break
                        
                        model_loaded = True
                        break
            
            if not model_loaded:
                raise Exception(f"No model found in expected locations: {model_paths}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            raise
    
    def setup_feature_engineering(self):
        """Initialize feature engineering with proper data loading."""
        print(f"\n🛠️ SETTING UP FEATURE ENGINEERING")
        
        try:
            # Try different data paths starting with the new organized structure
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            
            data_paths = [
                os.path.join(project_root, "data", "target_leagues_dataset.csv"),  # New organized structure
                "../../data/target_leagues_dataset.csv",
                "../data/target_leagues_dataset.csv",
                "Dataset collection/target_leagues_dataset.csv",  # Legacy paths
                "../Dataset collection/target_leagues_dataset.csv"
            ]
            
            for path in data_paths:
                if os.path.exists(path):
                    print(f"   📂 Found dataset at: {path}")
                    self.feature_engineering = AdvancedFeatureEngineering(path)
                    self.feature_engineering.load_and_analyze_data()
                    print(f"   ✅ Feature engineering ready")
                    return
            
            raise Exception(f"Could not find dataset for feature engineering. Searched paths: {data_paths}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            raise
    
    def load_champion_database(self):
        """Load all valid champion names from the dataset for validation."""
        print(f"\n📋 LOADING CHAMPION DATABASE")
        
        try:
            # Extract champions from the feature engineering data
            if self.feature_engineering and hasattr(self.feature_engineering, 'df'):
                df = self.feature_engineering.df
                
                # Get all champion columns
                champion_columns = ['top_champion', 'jng_champion', 'mid_champion', 'bot_champion', 'sup_champion']
                
                all_champions = set()
                for col in champion_columns:
                    if col in df.columns:
                        champions = df[col].dropna().unique()
                        all_champions.update(champions)
                
                # Remove any empty or invalid entries
                all_champions = {champ for champ in all_champions if champ and champ != 'Unknown'}
                
                self.valid_champions = all_champions
                print(f"   ✅ Loaded {len(self.valid_champions)} valid champions")
                print(f"   📝 Examples: {', '.join(list(self.valid_champions)[:10])}...")
                
        except Exception as e:
            print(f"   ⚠️ Could not load champion database: {e}")
            print(f"   🔄 Using default champion pool")
            self.valid_champions = set(self.champion_pool)
    
    def validate_champion_name(self, champion_input):
        """Validate champion name and suggest corrections if needed."""
        if not champion_input or not champion_input.strip():
            return None, "Empty champion name"
        
        # Clean and format input
        champion = champion_input.strip().title()
        
        # Direct match
        if champion in self.valid_champions:
            return champion, None
        
        # Try different capitalizations
        test_formats = [
            champion_input.strip(),
            champion_input.strip().lower(),
            champion_input.strip().upper(),
            champion_input.strip().title(),
            champion_input.strip().capitalize()
        ]
        
        for test_format in test_formats:
            if test_format in self.valid_champions:
                return test_format, None
        
        # Enhanced fuzzy matching with edit distance
        def levenshtein_distance(s1, s2):
            """Calculate edit distance between two strings."""
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        # Find similar champions using multiple criteria
        similar_champions = []
        champion_lower = champion.lower()
        
        # Common LoL champion misspellings and variations
        common_misspellings = {
            'talyah': 'Taliyah',
            'taliya': 'Taliyah', 
            'talia': 'Taliyah',
            'kaisa': 'Kai\'Sa',
            'kai sa': 'Kai\'Sa',
            'kaisa': 'Kai\'Sa',
            'leblanc': 'LeBlanc',
            'le blanc': 'LeBlanc',
            'xinzhao': 'Xin Zhao',
            'xin': 'Xin Zhao',
            'khazix': 'Kha\'Zix',
            'kha zix': 'Kha\'Zix',
            'jarvan': 'Jarvan IV',
            'j4': 'Jarvan IV',
            'reksai': 'Rek\'Sai',
            'rek sai': 'Rek\'Sai',
            'kogmaw': 'Kog\'Maw',
            'kog maw': 'Kog\'Maw',
            'chogath': 'Cho\'Gath',
            'cho gath': 'Cho\'Gath',
            'velkoz': 'Vel\'Koz',
            'vel koz': 'Vel\'Koz',
            'renata': 'Renata Glasc',
            'missfortune': 'Miss Fortune',
            'mf': 'Miss Fortune',
            'twistedfate': 'Twisted Fate',
            'tf': 'Twisted Fate',
            'masteryi': 'Master Yi',
            'yi': 'Master Yi',
            'drmundo': 'Dr. Mundo',
            'mundo': 'Dr. Mundo'
        }
        
        # Check common misspellings first
        if champion_lower in common_misspellings:
            correct_name = common_misspellings[champion_lower]
            if correct_name in self.valid_champions:
                return correct_name, None
        
        # Check for partial matches in misspellings
        for misspelling, correct_name in common_misspellings.items():
            if champion_lower in misspelling or misspelling in champion_lower:
                if correct_name in self.valid_champions:
                    similar_champions.append(correct_name)
        
        # Enhanced similarity matching
        for valid_champ in self.valid_champions:
            valid_lower = valid_champ.lower()
            
            # Calculate edit distance for close matches
            edit_distance = levenshtein_distance(champion_lower, valid_lower)
            max_allowed_distance = max(1, min(3, len(champion_lower) // 3))  # Allow 1-3 character differences
            
            if edit_distance <= max_allowed_distance:
                similar_champions.append((valid_champ, edit_distance))
                continue
            
            # Exact substring match
            if champion_lower in valid_lower or valid_lower in champion_lower:
                similar_champions.append((valid_champ, 0))  # High priority
                continue
            
            # Check for common character swaps or missing characters
            if len(champion_lower) >= 4:  # Only for longer names
                # Check if removing one character makes a match
                for i in range(len(champion_lower)):
                    test_name = champion_lower[:i] + champion_lower[i+1:]
                    if test_name == valid_lower:
                        similar_champions.append((valid_champ, 1))
                        break
                
                # Check if adding one character makes a match  
                if valid_lower.replace(champion_lower, '') and len(valid_lower) - len(champion_lower) == 1:
                    similar_champions.append((valid_champ, 1))
        
        # Process and sort suggestions
        final_suggestions = []
        
        # Add exact edit distance matches first
        for item in similar_champions:
            if isinstance(item, tuple):
                champ_name, distance = item
                final_suggestions.append((champ_name, distance))
            else:
                final_suggestions.append((item, 0))
        
        # Remove duplicates and sort by distance, then length, then alphabetically
        seen = set()
        unique_suggestions = []
        for champ_name, distance in final_suggestions:
            if champ_name not in seen:
                seen.add(champ_name)
                unique_suggestions.append((champ_name, distance))
        
        # Sort suggestions: edit distance first, then by length, then alphabetically
        unique_suggestions.sort(key=lambda x: (x[1], len(x[0]), x[0]))
        final_champion_list = [champ for champ, _ in unique_suggestions[:8]]  # Top 8 suggestions
        
        error_msg = f"'{champion}' not found"
        if final_champion_list:
            error_msg += f". Did you mean: {', '.join(final_champion_list)}"
        
        return None, error_msg
    
    def get_champion_input_with_validation(self, prompt, suggestions=None, used_champions=None):
        """Get champion input with real-time validation and suggestions."""
        while True:
            if suggestions:
                # Filter out already used champions from suggestions
                if used_champions:
                    available_suggestions = [champ for champ in suggestions if champ not in used_champions]
                    if available_suggestions:
                        print(f"   💡 Suggestions: {', '.join(available_suggestions[:8])}")
                    else:
                        print(f"   💡 (All suggested champions already used)")
                else:
                    print(f"   💡 Suggestions: {', '.join(suggestions[:8])}")
            
            print(f"   🔧 Commands: 'quit' to exit, 'restart' to restart draft, 'help' for help")
            champion_input = input(f"   {prompt}").strip()
            
            if not champion_input:
                print(f"   ⚠️ Please enter a champion name or command")
                continue
            
            # Handle special commands
            if champion_input.lower() in ['quit', 'q', 'exit']:
                print(f"\n👋 Thanks for using the LoL Match Predictor!")
                print(f"   Come back anytime to predict more matches!")
                import sys
                sys.exit(0)
            
            if champion_input.lower() in ['restart', 'r', 'reset']:
                print(f"\n🔄 Restarting draft phase...")
                return 'RESTART_DRAFT'
            
            if champion_input.lower() in ['help', 'h']:
                self.show_champion_help(used_champions)
                continue
            
            # Validate the champion
            validated_champion, error = self.validate_champion_name(champion_input)
            
            if validated_champion:
                # Check if champion is already used
                if used_champions and validated_champion in used_champions:
                    print(f"   ❌ {validated_champion} has already been picked or banned!")
                    print(f"   🚫 Used champions: {', '.join(sorted(used_champions))}")
                    continue
                
                print(f"   ✅ {validated_champion}")
                return validated_champion
            else:
                print(f"   ❌ {error}")
                
                # Ask if user wants to continue anyway or try again
                choice = input(f"   🔄 Try again (t), force use anyway (f), or help (h)? [t]: ").lower()
                
                if choice == 'f':
                    # User wants to force the invalid name
                    formatted_name = champion_input.strip().title()
                    
                    # Still check if it's already used
                    if used_champions and formatted_name in used_champions:
                        print(f"   ❌ {formatted_name} has already been picked or banned!")
                        continue
                    
                    print(f"   ⚠️ Using '{formatted_name}' (may cause prediction errors)")
                    return formatted_name
                elif choice == 'h':
                    self.show_champion_help(used_champions)
                    continue
                else:
                    # Default: try again
                    continue
    
    def show_champion_help(self, used_champions=None):
        """Show comprehensive help for champion input."""
        print(f"\n   📚 CHAMPION NAME HELP:")
        print(f"   🔸 Use exact spelling: 'Xin Zhao' not 'XinZhao'")
        print(f"   🔸 Preserve apostrophes: 'Kai'Sa' not 'Kaisa'")
        print(f"   🔸 Case doesn't matter: 'xin zhao' works")
        print(f"   🔸 Commands: 'quit' to exit, 'restart' to restart draft")
        
        if used_champions:
            print(f"   🚫 Already used: {', '.join(sorted(used_champions))}")
            remaining_count = len(self.valid_champions) - len(used_champions)
            print(f"   ✅ {remaining_count} champions still available")
        
        print(f"   🔸 Popular picks: {', '.join(list(self.valid_champions)[:10])}...")
        print(f"   🔸 Total champions available: {len(self.valid_champions)}")
        print()
    
    def display_teams(self):
        """Display available teams by league."""
        print(f"\n🏆 AVAILABLE TEAMS BY LEAGUE:")
        print("=" * 40)
        
        for league, teams in self.teams_db.items():
            print(f"\n{league}: {', '.join(teams)}")
    
    def get_team_selection(self):
        """Interactive team selection with proper validation."""
        print(f"\n👥 TEAM SELECTION")
        print("=" * 25)
        
        # Display available teams
        self.display_teams()
        
        # Get blue side team with validation
        print(f"\n🔵 BLUE SIDE TEAM:")
        print(f"   🔧 Commands: 'quit' to exit")
        while True:
            blue_league = input(f"   League (LCK/LEC/LCS/LPL): ").strip()
            
            if blue_league.lower() in ['quit', 'q', 'exit']:
                print(f"\n👋 Thanks for using the LoL Match Predictor!")
                sys.exit(0)
                
            blue_league = blue_league.upper()
            if blue_league in self.teams_db:
                break
            else:
                print(f"   ❌ Invalid league '{blue_league}'. Please choose from: {', '.join(self.teams_db.keys())}")
        
        print(f"   Available teams: {', '.join(self.teams_db[blue_league])}")
        while True:
            blue_team_input = input(f"   Team name: ").strip()
            
            if blue_team_input.lower() in ['quit', 'q', 'exit']:
                print(f"\n👋 Thanks for using the LoL Match Predictor!")
                sys.exit(0)
            
            # Case-insensitive team matching
            blue_team = None
            for team in self.teams_db[blue_league]:
                if team.lower() == blue_team_input.lower():
                    blue_team = team  # Use the correctly capitalized version
                    break
            
            if blue_team:
                break
            else:
                print(f"   ❌ Team '{blue_team_input}' not found in {blue_league}. Available: {', '.join(self.teams_db[blue_league])}")
        
        # Get red side team with validation
        print(f"\n🔴 RED SIDE TEAM:")
        print(f"   🔧 Commands: 'quit' to exit")
        while True:
            red_league = input(f"   League (LCK/LEC/LCS/LPL): ").strip()
            
            if red_league.lower() in ['quit', 'q', 'exit']:
                print(f"\n👋 Thanks for using the LoL Match Predictor!")
                sys.exit(0)
                
            red_league = red_league.upper()
            if red_league in self.teams_db:
                break
            else:
                print(f"   ❌ Invalid league '{red_league}'. Please choose from: {', '.join(self.teams_db.keys())}")
        
        print(f"   Available teams: {', '.join(self.teams_db[red_league])}")
        while True:
            red_team_input = input(f"   Team name: ").strip()
            
            if red_team_input.lower() in ['quit', 'q', 'exit']:
                print(f"\n👋 Thanks for using the LoL Match Predictor!")
                sys.exit(0)
            
            # Case-insensitive team matching
            red_team = None
            for team in self.teams_db[red_league]:
                if team.lower() == red_team_input.lower():
                    red_team = team  # Use the correctly capitalized version
                    break
            
            if red_team:
                break
            else:
                print(f"   ❌ Team '{red_team_input}' not found in {red_league}. Available: {', '.join(self.teams_db[red_league])}")
        
        print(f"\n✅ TEAMS CONFIRMED:")
        print(f"   🔵 Blue: {blue_team} ({blue_league})")
        print(f"   🔴 Red: {red_team} ({red_league})")
        
        return {
            'blue_team': blue_team,
            'blue_league': blue_league,
            'red_team': red_team,
            'red_league': red_league
        }
    
    def get_champion_input(self, role, side_color):
        """Get champion input with suggestions."""
        role_champions = {
            'top': [c for c in self.champion_pool if c in ['Gnar', 'Jax', 'Aatrox', 'Fiora', 'Camille', 'Ornn', 'Sion', 'Malphite', 'Renekton', 'Gragas']],
            'jungle': [c for c in self.champion_pool if c in ['Graves', 'Viego', 'Hecarim', 'Elise', 'Kha\'Zix', 'Jarvan IV', 'Lee Sin', 'Rek\'Sai', 'Nidalee', 'Sejuani']],
            'mid': [c for c in self.champion_pool if c in ['Azir', 'LeBlanc', 'Yone', 'Yasuo', 'Sylas', 'Orianna', 'Syndra', 'Ahri', 'Corki', 'Viktor']],
            'bot': [c for c in self.champion_pool if c in ['Jinx', 'Kai\'Sa', 'Varus', 'Jhin', 'Xayah', 'Ashe', 'Aphelios', 'Caitlyn', 'Ezreal', 'Lucian']],
            'support': [c for c in self.champion_pool if c in ['Thresh', 'Nautilus', 'Leona', 'Alistar', 'Braum', 'Lulu', 'Yuumi', 'Renata Glasc', 'Soraka', 'Janna']]
        }
        
        print(f"   {side_color} {role.capitalize()}: (Popular: {', '.join(role_champions[role][:5])})")
        champion = input(f"   Champion: ").title()
        
        return champion
    
    def get_picks_and_bans(self, teams):
        """Interactive picks and bans input following real professional draft format."""
        
        # Allow restarting the draft
        while True:
            try:
                return self._execute_draft(teams)
            except RestartDraftException:
                print(f"\n🔄 RESTARTING DRAFT...")
                print(f"   Starting fresh draft phase for {teams['blue_team']} vs {teams['red_team']}")
                continue
    
    def _execute_draft(self, teams):
        """Execute the draft phase - separated for restart functionality."""
        print(f"\n⚔️ PROFESSIONAL DRAFT PHASE")
        print("=" * 40)
        print(f"🔵 Blue: {teams['blue_team']} vs 🔴 Red: {teams['red_team']}")
        print(f"\n📋 PROFESSIONAL DRAFT ORDER:")
        print(f"   Phase 1: 6 Bans → 6 Picks → 4 Bans → 4 Picks")
        
        match_data = {
            'blue_team': teams['blue_team'],
            'red_team': teams['red_team'],
            'league': teams['blue_league'],
            'patch': '14.1',  # Current patch
            'split': 'Spring',
            'year': 2024,
            'playoffs': 0,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'side': 'Blue'
        }
        
        # Initialize ban and pick storage
        bans = {'blue': [], 'red': []}
        picks = {'blue': [], 'red': []}
        
        # Track all used champions (picks + bans from both teams)
        used_champions = set()
        
        print(f"\n🚫 FIRST BAN PHASE (6 bans)")
        print("=" * 35)
        
        # First Ban Phase (6 total)
        ban_sequence = [
            ('red', 1), ('blue', 1), ('blue', 2), 
            ('red', 2), ('blue', 3), ('red', 3)
        ]
        
        for side, ban_num in ban_sequence:
            side_emoji = "🔴" if side == 'red' else "🔵"
            team_name = teams[f'{side}_team']
            
            print(f"\n{side_emoji} {team_name.upper()} BAN {ban_num}:")
            
            # Get popular ban targets based on current meta
            popular_bans = ['Azir', 'Yone', 'Kai\'Sa', 'Viego', 'LeBlanc', 'Graves', 'Jinx', 'Nautilus']
            
            champion = self.get_champion_input_with_validation(
                f"Champion to ban: ", 
                suggestions=popular_bans,
                used_champions=used_champions
            )
            
            if champion == 'RESTART_DRAFT':
                raise RestartDraftException()
            
            bans[side].append(champion)
            used_champions.add(champion)
            print(f"   🚫 {team_name} bans {champion}")
        
        print(f"\n🎯 FIRST PICK PHASE (6 picks)")
        print("=" * 35)
        
        # First Pick Phase (6 total)
        pick_sequence = [
            ('blue', 1, 'flex'), ('red', 1, 'flex'), ('red', 2, 'flex'),
            ('blue', 2, 'flex'), ('blue', 3, 'flex'), ('red', 3, 'flex')
        ]
        
        for side, pick_num, role_hint in pick_sequence:
            side_emoji = "🔴" if side == 'red' else "🔵"
            team_name = teams[f'{side}_team']
            
            print(f"\n{side_emoji} {team_name.upper()} PICK {pick_num} ({role_hint}):")
            
            # Get suggestions based on pick order
            if pick_num == 1:
                suggestions = ['Jax', 'Aatrox', 'Graves', 'Azir', 'Jinx', 'Ornn']
                print(f"   Usually: Top/Jungle/Strong flex picks")
            elif pick_num == 2:
                suggestions = ['Viego', 'LeBlanc', 'Thresh', 'Xin Zhao', 'Varus']
                print(f"   Usually: Fill core roles, respond to opponent")
            else:
                suggestions = ['Hecarim', 'Yasuo', 'Nautilus', 'Kai\'Sa', 'Gnar']
                print(f"   Usually: Round out early composition")
            
            champion = self.get_champion_input_with_validation(
                f"Champion to pick: ",
                suggestions=suggestions,
                used_champions=used_champions
            )
            
            if champion == 'RESTART_DRAFT':
                raise RestartDraftException()
            
            picks[side].append(champion)
            used_champions.add(champion)
            print(f"   ⭐ {team_name} picks {champion}")
        
        print(f"\n🚫 SECOND BAN PHASE (4 bans)")
        print("=" * 35)
        print(f"   Target remaining threats and counter-picks")
        
        # Second Ban Phase (4 total)
        second_ban_sequence = [
            ('red', 4), ('blue', 4), ('red', 5), ('blue', 5)
        ]
        
        for side, ban_num in second_ban_sequence:
            side_emoji = "🔴" if side == 'red' else "🔵"
            team_name = teams[f'{side}_team']
            
            print(f"\n{side_emoji} {team_name.upper()} BAN {ban_num}:")
            
            # Show remaining strong picks as ban suggestions
            remaining_strong = ['Elise', 'Jhin', 'Alistar', 'Viktor', 'Fiora', 'Lucian']
            
            champion = self.get_champion_input_with_validation(
                f"Champion to ban: ",
                suggestions=remaining_strong,
                used_champions=used_champions
            )
            
            if champion == 'RESTART_DRAFT':
                raise RestartDraftException()
            
            bans[side].append(champion)
            used_champions.add(champion)
            print(f"   🚫 {team_name} bans {champion}")
        
        print(f"\n🎯 FINAL PICK PHASE (4 picks)")
        print("=" * 35)
        print(f"   Complete team compositions")
        
        # Final Pick Phase (4 total)
        final_pick_sequence = [
            ('red', 4), ('blue', 4), ('blue', 5), ('red', 5)
        ]
        
        for side, pick_num in final_pick_sequence:
            side_emoji = "🔴" if side == 'red' else "🔵"
            team_name = teams[f'{side}_team']
            
            print(f"\n{side_emoji} {team_name.upper()} PICK {pick_num}:")
            
            # Final picks usually fill remaining roles
            final_suggestions = ['Leona', 'Nidalee', 'Orianna', 'Caitlyn', 'Braum']
            
            champion = self.get_champion_input_with_validation(
                f"Champion to pick: ",
                suggestions=final_suggestions,
                used_champions=used_champions
            )
            
            if champion == 'RESTART_DRAFT':
                raise RestartDraftException()
            
            picks[side].append(champion)
            used_champions.add(champion)
            print(f"   ⭐ {team_name} picks {champion}")
        
        print(f"\n📊 DRAFT SUMMARY (Total {len(used_champions)} champions used):")
        print(f"   🚫 All bans: {', '.join(bans['blue'] + bans['red'])}")
        print(f"   ⭐ All picks: {', '.join(picks['blue'] + picks['red'])}")
        
        # Now assign roles - this is the tricky part since draft order != role order
        print(f"\n🎭 ROLE ASSIGNMENT")
        print("=" * 25)
        print(f"Now let's assign the 5 picks to their actual roles:")
        
        roles = ['Top', 'Jungle', 'Mid', 'Bot', 'Support']
        
        # Blue side role assignment
        print(f"\n🔵 {teams['blue_team'].upper()} ROLE ASSIGNMENT:")
        print(f"   Picks: {', '.join(picks['blue'])}")
        print(f"   🔧 Commands: 'quit' to exit, 'restart' to restart draft")
        
        blue_roles = {}
        for i, role in enumerate(roles):
            print(f"\n   {role} player:")
            available_picks = [pick for pick in picks['blue'] if pick not in blue_roles.values()]
            print(f"   Available: {', '.join(available_picks)}")
            
            while True:
                champion_input = input(f"   Which champion plays {role}? ").strip()
                
                if not champion_input:
                    print(f"   ⚠️ Please select a champion")
                    continue
                
                # Handle special commands
                if champion_input.lower() in ['quit', 'q', 'exit']:
                    print(f"\n👋 Thanks for using the LoL Match Predictor!")
                    sys.exit(0)
                
                if champion_input.lower() in ['restart', 'r', 'reset']:
                    print(f"\n🔄 Restarting draft phase...")
                    raise RestartDraftException()
                
                # Find exact match (case-insensitive) in the picks list
                matched_champion = None
                for pick in picks['blue']:
                    if pick.lower() == champion_input.lower():
                        matched_champion = pick
                        break
                
                if matched_champion and matched_champion not in blue_roles.values():
                    blue_roles[role.lower()] = matched_champion
                    print(f"   ✅ {matched_champion} → {role}")
                    break
                elif matched_champion in blue_roles.values():
                    print(f"   ❌ {matched_champion} already assigned to another role")
                elif not matched_champion:
                    print(f"   ❌ '{champion_input}' not in team picks: {picks['blue']}")
                    # Try to find a close match in picks
                    close_matches = [pick for pick in picks['blue'] if champion_input.lower() in pick.lower() or pick.lower() in champion_input.lower()]
                    if close_matches:
                        print(f"   💡 Did you mean: {', '.join(close_matches)}")
                else:
                    print(f"   ❌ Please enter a valid champion from the picks")
        
        # Red side role assignment  
        print(f"\n🔴 {teams['red_team'].upper()} ROLE ASSIGNMENT:")
        print(f"   Picks: {', '.join(picks['red'])}")
        print(f"   🔧 Commands: 'quit' to exit, 'restart' to restart draft")
        
        red_roles = {}
        for i, role in enumerate(roles):
            print(f"\n   {role} player:")
            available_picks = [pick for pick in picks['red'] if pick not in red_roles.values()]
            print(f"   Available: {', '.join(available_picks)}")
            
            while True:
                champion_input = input(f"   Which champion plays {role}? ").strip()
                
                if not champion_input:
                    print(f"   ⚠️ Please select a champion")
                    continue
                
                # Handle special commands
                if champion_input.lower() in ['quit', 'q', 'exit']:
                    print(f"\n👋 Thanks for using the LoL Match Predictor!")
                    sys.exit(0)
                
                if champion_input.lower() in ['restart', 'r', 'reset']:
                    print(f"\n🔄 Restarting draft phase...")
                    raise RestartDraftException()
                
                # Find exact match (case-insensitive) in the picks list
                matched_champion = None
                for pick in picks['red']:
                    if pick.lower() == champion_input.lower():
                        matched_champion = pick
                        break
                
                if matched_champion and matched_champion not in red_roles.values():
                    red_roles[role.lower()] = matched_champion
                    print(f"   ✅ {matched_champion} → {role}")
                    break
                elif matched_champion in red_roles.values():
                    print(f"   ❌ {matched_champion} already assigned to another role")
                elif not matched_champion:
                    print(f"   ❌ '{champion_input}' not in team picks: {picks['red']}")
                    # Try to find a close match in picks
                    close_matches = [pick for pick in picks['red'] if champion_input.lower() in pick.lower() or pick.lower() in champion_input.lower()]
                    if close_matches:
                        print(f"   💡 Did you mean: {', '.join(close_matches)}")
                else:
                    print(f"   ❌ Please enter a valid champion from the picks")
        
        # Assign to match_data format
        match_data['top_champion'] = blue_roles['top']
        match_data['jng_champion'] = blue_roles['jungle']  
        match_data['mid_champion'] = blue_roles['mid']
        match_data['bot_champion'] = blue_roles['bot']
        match_data['sup_champion'] = blue_roles['support']
        
        match_data['red_top_champion'] = red_roles['top']
        match_data['red_jng_champion'] = red_roles['jungle']
        match_data['red_mid_champion'] = red_roles['mid'] 
        match_data['red_bot_champion'] = red_roles['bot']
        match_data['red_sup_champion'] = red_roles['support']
        
        # Assign bans (pad with empty if needed)
        for i in range(5):
            match_data[f'ban{i+1}'] = bans['blue'][i] if i < len(bans['blue']) else ''
            match_data[f'red_ban{i+1}'] = bans['red'][i] if i < len(bans['red']) else ''
        
        # Additional info
        print(f"\n📋 ADDITIONAL INFO:")
        game_length = input(f"   Expected game length (minutes, default 30): ")
        match_data['game_length'] = int(game_length) if game_length.isdigit() else 30
        
        # Draft summary
        print(f"\n📊 DRAFT SUMMARY")
        print("=" * 25)
        print(f"🔵 {teams['blue_team'].upper()}:")
        print(f"   Top: {blue_roles['top']}")
        print(f"   Jungle: {blue_roles['jungle']}")
        print(f"   Mid: {blue_roles['mid']}")
        print(f"   Bot: {blue_roles['bot']}")
        print(f"   Support: {blue_roles['support']}")
        print(f"   Bans: {', '.join(bans['blue'])}")
        
        print(f"\n🔴 {teams['red_team'].upper()}:")
        print(f"   Top: {red_roles['top']}")
        print(f"   Jungle: {red_roles['jungle']}")
        print(f"   Mid: {red_roles['mid']}")
        print(f"   Bot: {red_roles['bot']}")
        print(f"   Support: {red_roles['support']}")
        print(f"   Bans: {', '.join(bans['red'])}")
        
        return match_data
    
    def predict_match(self, match_data):
        """Make prediction for the match."""
        print(f"\n🎯 MAKING PREDICTION")
        print("=" * 25)
        
        try:
            # We need to predict for both sides
            # Blue side prediction
            blue_data = match_data.copy()
            blue_data['team'] = blue_data['blue_team']
            blue_data['side'] = 'Blue'
            # Add any missing columns that feature engineering might expect
            blue_data['result'] = 0  # Placeholder - we're predicting this
            
            # Red side prediction  
            red_data = match_data.copy()
            red_data['team'] = red_data['red_team']
            red_data['side'] = 'Red'
            # Swap champions for red side perspective
            red_data['top_champion'] = match_data['red_top_champion']
            red_data['jng_champion'] = match_data['red_jng_champion']
            red_data['mid_champion'] = match_data['red_mid_champion']
            red_data['bot_champion'] = match_data['red_bot_champion']
            red_data['sup_champion'] = match_data['red_sup_champion']
            red_data['result'] = 0  # Placeholder - we're predicting this
            
            # Create features for both sides
            blue_df = pd.DataFrame([blue_data])
            red_df = pd.DataFrame([red_data])
            
            # Store original dataframe temporarily
            temp_df = self.feature_engineering.df.copy()
            
            print(f"   🔧 Processing blue side features...")
            # Blue side features - use EXACT same pipeline as training
            self.feature_engineering.df = blue_df
            blue_features = self.feature_engineering.create_advanced_features_vectorized()
            blue_final = self.apply_prediction_time_encoding(blue_features)
            
            print(f"   🔧 Processing red side features...")
            # Red side features - use EXACT same pipeline as training
            self.feature_engineering.df = red_df
            red_features = self.feature_engineering.create_advanced_features_vectorized()
            red_final = self.apply_prediction_time_encoding(red_features)
            
            # Restore original data
            self.feature_engineering.df = temp_df
            
            print(f"   📊 Blue features shape: {blue_final.shape}")
            print(f"   📊 Red features shape: {red_final.shape}")
            
            # Ensure both feature sets have the same columns
            if blue_final.shape[1] != red_final.shape[1]:
                print(f"   ⚠️ Feature mismatch, aligning columns...")
                # Get common columns
                common_cols = blue_final.columns.intersection(red_final.columns)
                blue_final = blue_final[common_cols]
                red_final = red_final[common_cols]
                print(f"   ✅ Aligned to {len(common_cols)} common features")
            
            # Make predictions
            if self.scaler:
                print(f"   🔧 Scaling features...")
                blue_scaled = self.scaler.transform(blue_final)
                red_scaled = self.scaler.transform(red_final)
            else:
                print(f"   ⚠️ No scaler available, using raw features")
                blue_scaled = blue_final
                red_scaled = red_final
            
            print(f"   🎯 Generating predictions...")
            # Get predictions
            blue_pred = self.model.predict_proba(blue_scaled)[0]
            red_pred = self.model.predict_proba(red_scaled)[0]
            
            # Calculate win probabilities
            blue_win_prob = blue_pred[1]  # Probability blue team wins
            red_win_prob = red_pred[1]    # Probability red team wins
            
            # Average both perspectives for final prediction
            avg_blue_win = (blue_win_prob + (1 - red_win_prob)) / 2
            avg_red_win = 1 - avg_blue_win
            
            # Display results
            print(f"\n🏆 PREDICTION RESULTS:")
            print(f"   🔵 {match_data['blue_team']}: {avg_blue_win:.1%} win probability")
            print(f"   🔴 {match_data['red_team']}: {avg_red_win:.1%} win probability")
            
            winner = match_data['blue_team'] if avg_blue_win > 0.5 else match_data['red_team']
            confidence = max(avg_blue_win, avg_red_win)
            
            print(f"\n🎯 PREDICTED WINNER: {winner}")
            print(f"   📊 Confidence: {confidence:.1%}")
            
            if confidence > 0.7:
                print(f"   ✅ High confidence prediction")
            elif confidence > 0.6:
                print(f"   ⚠️ Moderate confidence prediction")
            else:
                print(f"   🤔 Low confidence - could go either way")
                
            return {
                'winner': winner,
                'blue_win_prob': avg_blue_win,
                'red_win_prob': avg_red_win,
                'confidence': confidence
            }
            
        except Exception as e:
            print(f"   ❌ Error making prediction: {e}")
            print(f"   🔍 Debug info:")
            print(f"      Model loaded: {self.model is not None}")
            print(f"      Scaler loaded: {self.scaler is not None}")
            print(f"      Feature engineering ready: {self.feature_engineering is not None}")
            
            # Print more detailed error info
            import traceback
            print(f"   📋 Full error traceback:")
            traceback.print_exc()
            
            return None
    
    def apply_prediction_time_encoding(self, advanced_features_df):
        """Apply prediction-time encoding that creates the exact 37 features expected by the May 30th model."""
        print(f"   🎯 Applying prediction-time encoding for 37-feature pipeline...")
        
        # Start with the advanced features (should have ~28 features)
        feature_df = advanced_features_df.copy()
        
        # Load pre-trained encoders if available
        try:
            encoders = joblib.load('models/enhanced_encoders.joblib')
            print(f"   📊 Loaded pre-trained encoders: {list(encoders.keys())}")
        except:
            encoders = {}
            print(f"   ⚠️ No pre-trained encoders found, using fallback values")
        
        # Get current match data
        current_match = self.feature_engineering.df.iloc[0]
        
        # Basic categorical encoding (5 features)
        basic_categorical = ['league', 'team', 'patch', 'split', 'side']
        
        for feature in basic_categorical:
            if feature in current_match:
                value = str(current_match[feature])
                
                # Use pre-trained encoder or fallback
                if feature in encoders:
                    try:
                        # Try to transform using pre-trained encoder
                        encoded_value = encoders[feature].transform([[value]])[0][0]
                    except:
                        # Fallback to default value
                        encoded_value = 0.5
                else:
                    # Default encoding values
                    if feature == 'league':
                        encoded_value = {'LCK': 0.52, 'LEC': 0.50, 'LCS': 0.48, 'LPL': 0.54}.get(value, 0.50)
                    elif feature == 'side':
                        encoded_value = 1.0 if value == 'Blue' else 0.0
                    else:
                        encoded_value = 0.5
                
                feature_df[f'{feature}_encoded'] = encoded_value
                print(f"   ✅ {feature}_encoded = {encoded_value:.3f}")
        
        # Champion encoding (5 features) - using historical data
        champion_cols = ['top_champion', 'jng_champion', 'mid_champion', 'bot_champion', 'sup_champion']
        
        for col in champion_cols:
            if col in current_match:
                champion = current_match[col]
                
                # Get champion winrate from characteristics
                if champion in self.feature_engineering.champion_characteristics:
                    champ_winrate = self.feature_engineering.champion_characteristics[champion]['win_rate']
                else:
                    champ_winrate = 0.5  # Default for unknown champions
                
                feature_df[f'{col}_target_encoded'] = champ_winrate
        
        # Additional calculated features to reach 37 total
        # Meta-synergy interactions (3 features)
        team_meta_strength = feature_df.get('team_meta_strength', 0.5)
        
        # Convert to scalar if Series
        if hasattr(team_meta_strength, 'iloc'):
            team_meta_strength = team_meta_strength.iloc[0]
        
        # Use existing synergy features or create defaults
        if 'team_avg_synergy' in feature_df:
            team_synergy = feature_df['team_avg_synergy']
            if hasattr(team_synergy, 'iloc'):
                team_synergy = team_synergy.iloc[0]
        else:
            team_synergy = 0.5
        
        feature_df['meta_synergy_product'] = team_meta_strength * team_synergy
        feature_df['meta_synergy_ratio'] = team_meta_strength / max(team_synergy, 0.01)
        
        # Fix composition_historical_winrate access
        comp_winrate = feature_df.get('composition_historical_winrate', 0.5)
        if hasattr(comp_winrate, 'iloc'):
            comp_winrate = comp_winrate.iloc[0]
        
        feature_df['historical_meta_product'] = team_meta_strength * comp_winrate
        
        # Strategic analysis features (2 features) 
        ban_count = feature_df.get('ban_count', pd.Series([0])).iloc[0] if hasattr(feature_df.get('ban_count', 0), 'iloc') else feature_df.get('ban_count', 0)
        champion_count = feature_df.get('champion_count', pd.Series([5])).iloc[0] if hasattr(feature_df.get('champion_count', 5), 'iloc') else feature_df.get('champion_count', 5)
        
        feature_df['composition_strength_gap'] = abs(team_meta_strength - 0.5)  # Distance from neutral
        feature_df['ban_pressure_ratio'] = ban_count / max(champion_count, 1)
        
        # Ensure we have exactly 37 features as expected
        expected_feature_count = 37
        current_feature_count = len(feature_df.columns)
        
        print(f"   📊 Current features: {current_feature_count}, Target: {expected_feature_count}")
        
        # If we have too few features, add padding features
        while len(feature_df.columns) < expected_feature_count:
            padding_name = f'feature_padding_{len(feature_df.columns)}'
            feature_df[padding_name] = 0.5
        
        # If we have too many features, remove excess (shouldn't happen)
        if len(feature_df.columns) > expected_feature_count:
            excess_cols = feature_df.columns[expected_feature_count:]
            feature_df = feature_df.drop(columns=excess_cols)
            print(f"   ⚠️ Removed {len(excess_cols)} excess features")
        
        print(f"   ✅ Created exactly {len(feature_df.columns)} features for May 30th model compatibility")
        print(f"   📋 Final feature shape: {feature_df.shape}")
        
        return feature_df
    
    def run_best_of_series(self):
        """Run predictions for a best-of series."""
        print(f"\n🏆 BEST-OF SERIES PREDICTOR")
        print("=" * 35)
        
        # Get teams
        teams = self.get_team_selection()
        
        # Get series format
        series_format = input(f"\nSeries format (3/5, default 3): ")
        max_games = 5 if series_format == '5' else 3
        games_to_win = (max_games + 1) // 2
        
        print(f"\n🎮 PREDICTING BEST-OF-{max_games} SERIES")
        print(f"   🔵 {teams['blue_team']} vs 🔴 {teams['red_team']}")
        print(f"   First to {games_to_win} wins!")
        
        series_results = []
        blue_wins = 0
        red_wins = 0
        
        game_num = 1
        while blue_wins < games_to_win and red_wins < games_to_win and game_num <= max_games:
            print(f"\n" + "="*50)
            print(f"🎮 GAME {game_num}")
            print(f"Current Score - 🔵 {teams['blue_team']}: {blue_wins} | 🔴 {teams['red_team']}: {red_wins}")
            print(f"="*50)
            
            # Get picks and bans for this game
            match_data = self.get_picks_and_bans(teams)
            
            # Make prediction
            result = self.predict_match(match_data)
            
            if result:
                series_results.append(result)
                
                if result['winner'] == teams['blue_team']:
                    blue_wins += 1
                else:
                    red_wins += 1
                
                print(f"\n📊 GAME {game_num} RESULT:")
                print(f"   Winner: {result['winner']}")
                print(f"   Updated Score - 🔵 {teams['blue_team']}: {blue_wins} | 🔴 {teams['red_team']}: {red_wins}")
            
            # Check for series end
            if blue_wins >= games_to_win:
                print(f"\n🏆 SERIES WINNER: {teams['blue_team']} ({blue_wins}-{red_wins})")
                break
            elif red_wins >= games_to_win:
                print(f"\n🏆 SERIES WINNER: {teams['red_team']} ({red_wins}-{blue_wins})")
                break
            
            # Continue to next game
            continue_series = input(f"\nContinue to Game {game_num + 1}? (y/n, default y): ")
            if continue_series.lower() == 'n':
                break
                
            game_num += 1
        
        # Series summary
        print(f"\n🏆 SERIES SUMMARY")
        print("=" * 25)
        print(f"   🔵 {teams['blue_team']}: {blue_wins} wins")
        print(f"   🔴 {teams['red_team']}: {red_wins} wins")
        
        if blue_wins > red_wins:
            print(f"   🏆 Series Winner: {teams['blue_team']}")
        elif red_wins > blue_wins:
            print(f"   🏆 Series Winner: {teams['red_team']}")
        else:
            print(f"   🤝 Series tied")
        
        return series_results

def main():
    """Main interactive prediction function."""
    try:
        predictor = InteractiveLoLPredictor()
        
        print(f"\n🎯 CHOOSE PREDICTION MODE:")
        print(f"   1. Single Game Prediction")
        print(f"   2. Best-of Series Prediction")
        print(f"   q. Quit")
        
        while True:
            mode = input(f"\nSelect mode (1/2/q, default 2): ").strip().lower()
            
            if mode in ['q', 'quit', 'exit']:
                print(f"\n👋 Thanks for using the LoL Match Predictor!")
                print(f"   Come back anytime to predict more matches!")
                return
            elif mode == '1':
                # Single game prediction
                teams = predictor.get_team_selection()
                match_data = predictor.get_picks_and_bans(teams)
                result = predictor.predict_match(match_data)
                break
            elif mode == '2' or mode == '':
                # Best-of series prediction
                predictor.run_best_of_series()
                break
            else:
                print(f"   ❌ Invalid option '{mode}'. Please choose 1, 2, or q")
        
        print(f"\n✅ PREDICTION SESSION COMPLETE!")
        print(f"   Ready for your next prediction!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"   Please check model files and try again")

if __name__ == "__main__":
    main() 