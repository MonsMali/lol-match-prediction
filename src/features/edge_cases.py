"""
Edge Case Detection and Handling Module

Detects and handles problematic inputs for League of Legends match prediction.

Key Features:
- Detects: new champions, new teams, new patches, small sample sizes
- Fallback strategies: role-average, league-average, nearest-patch
- Confidence penalty multiplier for uncertain predictions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


class EdgeCaseHandler:
    """
    Detects and handles edge cases in match prediction inputs.

    Edge cases include:
    - New champions not seen during training
    - New teams with insufficient historical data
    - New patches with potentially different meta
    - Champion/team combinations with very small sample sizes

    Each edge case type has an associated confidence penalty that
    can be used to adjust prediction confidence.

    Attributes:
        known_champions: Set of champions seen during training
        known_teams: Set of teams seen during training
        known_patches: Set of patches seen during training
        team_match_counts: Dictionary of team -> match count
        champion_match_counts: Dictionary of champion -> match count
    """

    # Confidence penalties for different edge case types
    PENALTIES = {
        'new_champion': 0.15,      # 15% penalty for unknown champion
        'new_team': 0.20,          # 20% penalty for unknown team
        'new_patch': 0.10,         # 10% penalty for new patch
        'low_sample_team': 0.10,   # 10% penalty for team with few matches
        'low_sample_champion': 0.05,  # 5% penalty per low-sample champion
        'multiple_edge_cases': 0.05   # 5% additional for multiple issues
    }

    # Minimum samples for reliable predictions
    MIN_TEAM_MATCHES = 10
    MIN_CHAMPION_MATCHES = 20

    def __init__(self, known_champions: Optional[Set[str]] = None,
                 known_teams: Optional[Set[str]] = None,
                 known_patches: Optional[Set[str]] = None,
                 team_match_counts: Optional[Dict[str, int]] = None,
                 champion_match_counts: Optional[Dict[str, int]] = None):
        """
        Initialize the EdgeCaseHandler.

        Args:
            known_champions: Set of champion names from training data
            known_teams: Set of team names from training data
            known_patches: Set of patch versions from training data
            team_match_counts: Dict of team -> number of matches in training
            champion_match_counts: Dict of champion -> number of matches in training
        """
        self.known_champions = known_champions or set()
        self.known_teams = known_teams or set()
        self.known_patches = known_patches or set()
        self.team_match_counts = team_match_counts or {}
        self.champion_match_counts = champion_match_counts or {}

        # Fallback values (computed from training data)
        self._role_averages = {}
        self._league_averages = {}
        self._global_averages = {}

    @classmethod
    def from_training_data(cls, df: pd.DataFrame) -> 'EdgeCaseHandler':
        """
        Create an EdgeCaseHandler from training data.

        Args:
            df: Training DataFrame with match data

        Returns:
            Configured EdgeCaseHandler instance
        """
        # Extract known champions
        champion_columns = [
            'top_champion', 'jng_champion', 'mid_champion',
            'bot_champion', 'sup_champion'
        ]
        known_champions = set()
        champion_counts = defaultdict(int)

        for col in champion_columns:
            if col in df.columns:
                champions = df[col].dropna().unique()
                known_champions.update(champions)
                for champ in df[col].dropna():
                    champion_counts[champ] += 1

        # Extract known teams
        known_teams = set()
        team_counts = defaultdict(int)

        if 'team' in df.columns:
            known_teams.update(df['team'].dropna().unique())
            team_counts = df['team'].value_counts().to_dict()

        # Extract known patches
        known_patches = set()
        if 'patch' in df.columns:
            known_patches.update(df['patch'].dropna().astype(str).unique())

        handler = cls(
            known_champions=known_champions,
            known_teams=known_teams,
            known_patches=known_patches,
            team_match_counts=dict(team_counts),
            champion_match_counts=dict(champion_counts)
        )

        # Calculate fallback values
        handler._calculate_fallback_values(df)

        return handler

    def _calculate_fallback_values(self, df: pd.DataFrame) -> None:
        """
        Calculate fallback values for edge case handling.
        """
        # Role averages (average stats for each champion role)
        champion_columns = {
            'top': 'top_champion',
            'jungle': 'jng_champion',
            'mid': 'mid_champion',
            'bot': 'bot_champion',
            'support': 'sup_champion'
        }

        for role, col in champion_columns.items():
            if col in df.columns and 'result' in df.columns:
                # Calculate win rate per role
                role_data = df.groupby(col)['result'].agg(['mean', 'count'])
                role_data = role_data[role_data['count'] >= 10]
                self._role_averages[role] = {
                    'mean_winrate': role_data['mean'].mean() if len(role_data) > 0 else 0.5,
                    'champions': role_data.to_dict('index')
                }

        # League averages
        if 'league' in df.columns and 'result' in df.columns:
            league_stats = df.groupby('league')['result'].agg(['mean', 'count'])
            self._league_averages = league_stats.to_dict('index')

        # Global averages
        if 'result' in df.columns:
            self._global_averages = {
                'win_rate': df['result'].mean(),
                'total_matches': len(df)
            }

    def detect(self, match_data: Dict) -> Dict:
        """
        Detect edge cases in match data.

        Args:
            match_data: Dictionary containing match information
                Expected keys: team, champions (top, jng, mid, bot, sup), patch

        Returns:
            Dictionary containing:
            - edge_cases: List of detected edge case types
            - confidence_penalty: Total confidence penalty (0-1)
            - warnings: List of warning messages
            - details: Detailed information about each edge case
        """
        edge_cases = []
        warnings_list = []
        details = {}
        total_penalty = 0.0

        # Check for new team
        team = match_data.get('team') or match_data.get('blue_team')
        if team and team not in self.known_teams:
            edge_cases.append('new_team')
            warnings_list.append(f"Team '{team}' not in training data")
            details['new_team'] = team
            total_penalty += self.PENALTIES['new_team']
        elif team and self.team_match_counts.get(team, 0) < self.MIN_TEAM_MATCHES:
            edge_cases.append('low_sample_team')
            match_count = self.team_match_counts.get(team, 0)
            warnings_list.append(
                f"Team '{team}' has only {match_count} matches in training"
            )
            details['low_sample_team'] = {
                'team': team,
                'match_count': match_count
            }
            total_penalty += self.PENALTIES['low_sample_team']

        # Check for new champions
        champion_keys = [
            'top_champion', 'jng_champion', 'mid_champion',
            'bot_champion', 'sup_champion'
        ]
        new_champions = []
        low_sample_champions = []

        for key in champion_keys:
            champion = match_data.get(key)
            if champion:
                if champion not in self.known_champions:
                    new_champions.append(champion)
                elif self.champion_match_counts.get(champion, 0) < self.MIN_CHAMPION_MATCHES:
                    low_sample_champions.append({
                        'champion': champion,
                        'count': self.champion_match_counts.get(champion, 0)
                    })

        if new_champions:
            edge_cases.append('new_champion')
            warnings_list.append(
                f"Unknown champion(s): {', '.join(new_champions)}"
            )
            details['new_champions'] = new_champions
            total_penalty += self.PENALTIES['new_champion'] * len(new_champions)

        if low_sample_champions:
            edge_cases.append('low_sample_champion')
            warnings_list.append(
                f"Low-sample champions: {', '.join([c['champion'] for c in low_sample_champions])}"
            )
            details['low_sample_champions'] = low_sample_champions
            total_penalty += self.PENALTIES['low_sample_champion'] * len(low_sample_champions)

        # Check for new patch
        patch = str(match_data.get('patch', ''))
        if patch and patch not in self.known_patches:
            edge_cases.append('new_patch')
            nearest = self._find_nearest_patch(patch)
            warnings_list.append(
                f"Patch '{patch}' not in training data (nearest: {nearest})"
            )
            details['new_patch'] = {
                'patch': patch,
                'nearest_known': nearest
            }
            total_penalty += self.PENALTIES['new_patch']

        # Additional penalty for multiple edge cases
        if len(edge_cases) > 1:
            total_penalty += self.PENALTIES['multiple_edge_cases']

        # Cap penalty at 0.5 (50%)
        total_penalty = min(total_penalty, 0.5)

        return {
            'edge_cases': edge_cases,
            'confidence_penalty': total_penalty,
            'warnings': warnings_list,
            'details': details,
            'has_edge_cases': len(edge_cases) > 0
        }

    def _find_nearest_patch(self, patch: str) -> Optional[str]:
        """
        Find the nearest known patch to the given patch.
        """
        if not self.known_patches:
            return None

        try:
            # Parse patch version (e.g., "14.1" -> (14, 1))
            major, minor = map(int, patch.split('.')[:2])

            best_match = None
            best_distance = float('inf')

            for known_patch in self.known_patches:
                try:
                    k_major, k_minor = map(int, str(known_patch).split('.')[:2])
                    distance = abs(major - k_major) * 100 + abs(minor - k_minor)
                    if distance < best_distance:
                        best_distance = distance
                        best_match = known_patch
                except (ValueError, IndexError):
                    continue

            return best_match
        except (ValueError, IndexError):
            # Return most recent patch if parsing fails
            return max(self.known_patches) if self.known_patches else None

    def apply_fallback(self, match_data: Dict,
                       edge_case_result: Dict) -> Dict:
        """
        Apply fallback strategies for detected edge cases.

        Args:
            match_data: Original match data dictionary
            edge_case_result: Result from detect()

        Returns:
            Modified match data with fallback values applied
        """
        modified_data = match_data.copy()
        fallback_applied = []

        details = edge_case_result.get('details', {})

        # Handle new champions with role averages
        if 'new_champions' in details:
            for champion in details['new_champions']:
                role = self._get_champion_role(champion, match_data)
                if role and role in self._role_averages:
                    fallback_applied.append(
                        f"Using role average for {champion} ({role})"
                    )
                    # Add fallback indicator to data
                    modified_data[f'{role}_champion_fallback'] = True

        # Handle new teams with league averages
        if 'new_team' in details:
            team = details['new_team']
            league = match_data.get('league')
            if league and league in self._league_averages:
                fallback_applied.append(
                    f"Using league average for team {team}"
                )
                modified_data['team_fallback'] = True
                modified_data['team_fallback_source'] = 'league_average'

        # Handle new patches with nearest patch
        if 'new_patch' in details:
            nearest = details['new_patch'].get('nearest_known')
            if nearest:
                fallback_applied.append(
                    f"Using nearest patch {nearest} instead of {details['new_patch']['patch']}"
                )
                modified_data['patch_fallback'] = nearest

        modified_data['fallbacks_applied'] = fallback_applied
        return modified_data

    def _get_champion_role(self, champion: str, match_data: Dict) -> Optional[str]:
        """
        Determine which role a champion is playing based on match data.
        """
        role_mapping = {
            'top_champion': 'top',
            'jng_champion': 'jungle',
            'mid_champion': 'mid',
            'bot_champion': 'bot',
            'sup_champion': 'support'
        }

        for key, role in role_mapping.items():
            if match_data.get(key) == champion:
                return role
        return None

    def get_confidence_multiplier(self, edge_case_result: Dict) -> float:
        """
        Calculate a confidence multiplier based on edge cases.

        Args:
            edge_case_result: Result from detect()

        Returns:
            Multiplier between 0.5 and 1.0 (1.0 = no penalty)
        """
        penalty = edge_case_result.get('confidence_penalty', 0)
        return 1.0 - penalty

    def format_warnings(self, edge_case_result: Dict) -> str:
        """
        Format edge case warnings as a human-readable string.

        Args:
            edge_case_result: Result from detect()

        Returns:
            Formatted warning string (empty if no edge cases)
        """
        if not edge_case_result.get('has_edge_cases'):
            return ""

        lines = ["Edge Case Warnings:"]
        for warning in edge_case_result.get('warnings', []):
            lines.append(f"  - {warning}")

        penalty = edge_case_result.get('confidence_penalty', 0)
        if penalty > 0:
            lines.append(f"  Confidence adjusted by: -{penalty:.0%}")

        return '\n'.join(lines)

    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics about known entities.

        Returns:
            Dictionary with counts of known entities
        """
        return {
            'known_champions': len(self.known_champions),
            'known_teams': len(self.known_teams),
            'known_patches': len(self.known_patches),
            'teams_with_10plus_matches': sum(
                1 for c in self.team_match_counts.values()
                if c >= self.MIN_TEAM_MATCHES
            ),
            'champions_with_20plus_matches': sum(
                1 for c in self.champion_match_counts.values()
                if c >= self.MIN_CHAMPION_MATCHES
            )
        }
