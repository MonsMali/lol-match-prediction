"""
Training Scheduler for LoL Match Prediction System.

Manages training schedules and determines when retraining is needed.
"""

import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import MODELS_DIR, DATASET_PATH

from .config import SchedulerConfig, DEFAULT_CONFIG


@dataclass
class TrainingTrigger:
    """Represents a training trigger event."""
    trigger_type: str  # 'scheduled', 'drift', 'data', 'manual'
    triggered_at: str = field(default_factory=lambda: datetime.now().isoformat())
    reason: str = ""
    config: Dict = field(default_factory=dict)


@dataclass
class SchedulerState:
    """State of the training scheduler."""
    last_training_time: Optional[str] = None
    last_training_trigger: Optional[str] = None
    training_count: int = 0
    last_check_time: Optional[str] = None
    last_data_row_count: int = 0


class TrainingScheduler:
    """
    Manages training schedules and determines when to retrain models.

    Supports multiple trigger types:
    - Scheduled: Time-based (daily, weekly, monthly)
    - Drift-based: When performance or features drift
    - Data-based: When significant new data is available
    - Manual: Explicit trigger requests
    """

    STATE_FILENAME = "scheduler_state.json"
    LOG_FILENAME = "scheduler_log.json"

    def __init__(self, config: Optional[SchedulerConfig] = None,
                 state_dir: Optional[Path] = None):
        """
        Initialize scheduler.

        Args:
            config: Scheduler configuration
            state_dir: Directory for state storage
        """
        self.config = config or DEFAULT_CONFIG.scheduler
        self.state_dir = state_dir or MODELS_DIR
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.state_path = self.state_dir / self.STATE_FILENAME
        self.log_path = self.state_dir / self.LOG_FILENAME

        self.state = self._load_state()

    def _load_state(self) -> SchedulerState:
        """Load scheduler state from file."""
        if self.state_path.exists():
            with open(self.state_path, 'r') as f:
                data = json.load(f)
            return SchedulerState(**data)
        return SchedulerState()

    def _save_state(self) -> None:
        """Save scheduler state to file."""
        with open(self.state_path, 'w') as f:
            json.dump(asdict(self.state), f, indent=2)

    def _log_event(self, event_type: str, details: Dict) -> None:
        """Log scheduler event."""
        log = []
        if self.log_path.exists():
            with open(self.log_path, 'r') as f:
                log = json.load(f)

        log.append({
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'details': details
        })

        # Keep last 1000 events
        log = log[-1000:]

        with open(self.log_path, 'w') as f:
            json.dump(log, f, indent=2)

    def should_retrain(self) -> Tuple[bool, str]:
        """
        Determine if retraining should occur.

        Returns:
            Tuple of (should_retrain, reason)
        """
        now = datetime.now()
        self.state.last_check_time = now.isoformat()
        self._save_state()

        # Check cooldown
        if self.state.last_training_time:
            last_training = datetime.fromisoformat(self.state.last_training_time)
            hours_since = (now - last_training).total_seconds() / 3600

            if hours_since < self.config.min_hours_between_training:
                return False, f"Cooldown active ({hours_since:.1f}h since last training)"

        # Check force retrain threshold
        if self.state.last_training_time:
            last_training = datetime.fromisoformat(self.state.last_training_time)
            days_since = (now - last_training).days

            if days_since >= self.config.force_retrain_days:
                self._log_event('trigger_check', {
                    'result': True,
                    'reason': f'Force retrain after {days_since} days'
                })
                return True, f"Force retrain threshold reached ({days_since} days)"

        # Check scheduled time
        should_schedule, schedule_reason = self._check_schedule(now)
        if should_schedule:
            self._log_event('trigger_check', {
                'result': True,
                'reason': schedule_reason
            })
            return True, schedule_reason

        # Check data-based trigger
        should_data, data_reason = self._check_data_trigger()
        if should_data:
            self._log_event('trigger_check', {
                'result': True,
                'reason': data_reason
            })
            return True, data_reason

        return False, "No retrain trigger met"

    def _check_schedule(self, now: datetime) -> Tuple[bool, str]:
        """Check if scheduled training should occur."""
        # If never trained, should train
        if not self.state.last_training_time:
            return True, "Initial training required"

        last_training = datetime.fromisoformat(self.state.last_training_time)

        if self.config.schedule_type == 'daily':
            # Check if we're past scheduled hour and haven't trained today
            if now.hour >= self.config.hour and last_training.date() < now.date():
                return True, "Daily scheduled training"

        elif self.config.schedule_type == 'weekly':
            # Check if we're on scheduled day/hour and haven't trained this week
            days_since = (now - last_training).days
            if now.weekday() == self.config.day_of_week and days_since >= 7:
                if now.hour >= self.config.hour:
                    return True, "Weekly scheduled training"

        elif self.config.schedule_type == 'monthly':
            # Check if we're on scheduled day/hour and haven't trained this month
            if now.day == self.config.day_of_month:
                if now.month != last_training.month or now.year != last_training.year:
                    if now.hour >= self.config.hour:
                        return True, "Monthly scheduled training"

        return False, ""

    def _check_data_trigger(self) -> Tuple[bool, str]:
        """Check if new data warrants retraining."""
        if not DATASET_PATH.exists():
            return False, ""

        # Count rows in dataset
        with open(DATASET_PATH, 'r') as f:
            current_rows = sum(1 for _ in f) - 1  # Subtract header

        new_rows = current_rows - self.state.last_data_row_count

        if new_rows >= self.config.min_new_matches:
            return True, f"New data available ({new_rows} new matches)"

        return False, ""

    def schedule_training(self, trigger: TrainingTrigger) -> str:
        """
        Schedule a training run.

        Args:
            trigger: Training trigger details

        Returns:
            Training job ID
        """
        job_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self._log_event('training_scheduled', {
            'job_id': job_id,
            'trigger_type': trigger.trigger_type,
            'reason': trigger.reason
        })

        return job_id

    def record_training_complete(self, job_id: str, success: bool,
                                  metrics: Optional[Dict] = None) -> None:
        """
        Record completion of a training run.

        Args:
            job_id: Training job ID
            success: Whether training succeeded
            metrics: Optional training metrics
        """
        now = datetime.now()

        if success:
            self.state.last_training_time = now.isoformat()
            self.state.training_count += 1

            # Update data row count
            if DATASET_PATH.exists():
                with open(DATASET_PATH, 'r') as f:
                    self.state.last_data_row_count = sum(1 for _ in f) - 1

        self._save_state()

        self._log_event('training_complete', {
            'job_id': job_id,
            'success': success,
            'metrics': metrics or {}
        })

    def get_next_scheduled_time(self) -> Optional[datetime]:
        """Get the next scheduled training time."""
        now = datetime.now()

        if self.config.schedule_type == 'daily':
            next_time = now.replace(hour=self.config.hour, minute=0, second=0, microsecond=0)
            if next_time <= now:
                next_time += timedelta(days=1)
            return next_time

        elif self.config.schedule_type == 'weekly':
            days_until = (self.config.day_of_week - now.weekday()) % 7
            if days_until == 0 and now.hour >= self.config.hour:
                days_until = 7
            next_time = now.replace(hour=self.config.hour, minute=0, second=0, microsecond=0)
            next_time += timedelta(days=days_until)
            return next_time

        elif self.config.schedule_type == 'monthly':
            next_time = now.replace(day=self.config.day_of_month, hour=self.config.hour,
                                   minute=0, second=0, microsecond=0)
            if next_time <= now:
                # Move to next month
                if now.month == 12:
                    next_time = next_time.replace(year=now.year + 1, month=1)
                else:
                    next_time = next_time.replace(month=now.month + 1)
            return next_time

        return None

    def get_scheduler_status(self) -> Dict:
        """Get current scheduler status."""
        next_scheduled = self.get_next_scheduled_time()

        return {
            'last_training': self.state.last_training_time,
            'training_count': self.state.training_count,
            'last_check': self.state.last_check_time,
            'last_data_row_count': self.state.last_data_row_count,
            'schedule_type': self.config.schedule_type,
            'next_scheduled': next_scheduled.isoformat() if next_scheduled else None,
            'min_new_matches_trigger': self.config.min_new_matches,
            'force_retrain_days': self.config.force_retrain_days
        }

    def get_recent_events(self, n: int = 10) -> List[Dict]:
        """Get recent scheduler events."""
        if not self.log_path.exists():
            return []

        with open(self.log_path, 'r') as f:
            log = json.load(f)

        return log[-n:]


def print_scheduler_status(scheduler: TrainingScheduler) -> None:
    """Print formatted scheduler status."""
    status = scheduler.get_scheduler_status()

    print(f"\n{'='*60}")
    print("TRAINING SCHEDULER STATUS")
    print(f"{'='*60}")
    print(f"Schedule Type: {status['schedule_type']}")
    print(f"Last Training: {status['last_training'] or 'Never'}")
    print(f"Training Count: {status['training_count']}")
    print(f"Next Scheduled: {status['next_scheduled'] or 'N/A'}")
    print(f"Data Trigger: {status['min_new_matches_trigger']} new matches")
    print(f"Force Retrain: After {status['force_retrain_days']} days")

    print(f"\nRecent Events:")
    for event in scheduler.get_recent_events(5):
        print(f"  {event['timestamp'][:19]}: {event['event_type']}")
