"""
Autonomy and resume system for the coding agent.
Supports:
- Cron-based scheduling of agent tasks
- Checkpoint/resume for long-running tasks
- Background task execution
- Idle detection and auto-wake
"""

import os
import json
import time
import asyncio
import logging
from pathlib import Path
from typing import Optional, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .config import AutonomyConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint system
# ---------------------------------------------------------------------------


@dataclass
class Checkpoint:
    """A serializable checkpoint of agent state."""
    session_id: str
    timestamp: str
    messages: list[dict]
    system_prompt: str = ""
    step_count: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "messages": self.messages,
            "system_prompt": self.system_prompt,
            "step_count": self.step_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Checkpoint":
        return cls(
            session_id=data.get("session_id", ""),
            timestamp=data.get("timestamp", ""),
            messages=data.get("messages", []),
            system_prompt=data.get("system_prompt", ""),
            step_count=data.get("step_count", 0),
            metadata=data.get("metadata", {}),
        )


class CheckpointManager:
    """Manages saving and loading agent checkpoints."""

    def __init__(self, config: AutonomyConfig, workspace: str = "."):
        self.config = config
        self.checkpoint_dir = Path(workspace) / config.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self, checkpoint: Checkpoint) -> str:
        """Save a checkpoint to disk. Returns the checkpoint file path."""
        checkpoint.timestamp = datetime.now().isoformat()
        filename = f"ckpt_{checkpoint.session_id}_{int(time.time())}.json"
        filepath = self.checkpoint_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(checkpoint.to_dict(), f, indent=2, ensure_ascii=False)

        # Also update the "latest" symlink-like marker
        latest_path = self.checkpoint_dir / "latest.json"
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump({"session_id": checkpoint.session_id, "file": filename}, f)

        return str(filepath)

    def load_latest(self) -> Optional[Checkpoint]:
        """Load the most recent checkpoint."""
        latest_path = self.checkpoint_dir / "latest.json"
        if not latest_path.exists():
            return None

        try:
            with open(latest_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            checkpoint_file = self.checkpoint_dir / info["file"]
            if checkpoint_file.exists():
                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    return Checkpoint.from_dict(json.load(f))
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")

        return None

    def list_checkpoints(self) -> list[dict]:
        """List all available checkpoints."""
        checkpoints = []
        for fpath in sorted(self.checkpoint_dir.glob("ckpt_*.json"), reverse=True):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                checkpoints.append({
                    "session_id": data.get("session_id", ""),
                    "timestamp": data.get("timestamp", ""),
                    "step_count": data.get("step_count", 0),
                    "file": fpath.name,
                })
            except Exception:
                pass
        return checkpoints

    def delete_old(self, keep: int = 10) -> int:
        """Delete old checkpoints, keeping the N most recent. Returns count deleted."""
        checkpoints = self.list_checkpoints()
        deleted = 0
        for cp in checkpoints[keep:]:
            filepath = self.checkpoint_dir / cp["file"]
            if filepath.exists():
                filepath.unlink()
                deleted += 1
        return deleted


# ---------------------------------------------------------------------------
# Cron scheduler
# ---------------------------------------------------------------------------


@dataclass
class ScheduledTask:
    """A task scheduled to run at specific times."""
    task_id: str
    cron_expression: str  # 5-field cron: minute hour dom month dow
    prompt: str
    recurring: bool = True
    durable: bool = False  # Survive restarts
    created_at: str = ""
    last_run: Optional[str] = None
    next_run: Optional[str] = None
    enabled: bool = True

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "cron_expression": self.cron_expression,
            "prompt": self.prompt,
            "recurring": self.recurring,
            "durable": self.durable,
            "created_at": self.created_at,
            "last_run": self.last_run,
            "next_run": self.next_run,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ScheduledTask":
        return cls(**data)


class CronScheduler:
    """
    Simple cron-based task scheduler.
    Uses croniter for cron expression parsing and next-run calculation.
    """

    def __init__(self, config: AutonomyConfig, workspace: str = "."):
        self.config = config
        self.workspace = workspace
        self._tasks: dict[str, ScheduledTask] = {}
        self._task_id_counter = 0
        self._running = False
        self._task_callback: Optional[Callable[[ScheduledTask], Awaitable[None]]] = None

        # Load durable tasks
        self._load_durable_tasks()

    def set_callback(self, callback: Callable[[ScheduledTask], Awaitable[None]]) -> None:
        """Set the callback for executing a scheduled task."""
        self._task_callback = callback

    def schedule(self, cron_expression: str, prompt: str,
                 recurring: bool = True, durable: bool = False) -> str:
        """Schedule a new task. Returns the task ID."""
        self._task_id_counter += 1
        task_id = f"cron_{self._task_id_counter}"

        now = datetime.now()
        task = ScheduledTask(
            task_id=task_id,
            cron_expression=cron_expression,
            prompt=prompt,
            recurring=recurring,
            durable=durable,
            created_at=now.isoformat(),
            next_run=self._calculate_next_run(cron_expression, now),
        )

        self._tasks[task_id] = task

        if durable:
            self._save_durable_tasks()

        return task_id

    def cancel(self, task_id: str) -> bool:
        """Cancel a scheduled task."""
        if task_id in self._tasks:
            del self._tasks[task_id]
            self._save_durable_tasks()
            return True
        return False

    def list_tasks(self) -> list[ScheduledTask]:
        """List all scheduled tasks."""
        return list(self._tasks.values())

    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Get a scheduled task by ID."""
        return self._tasks.get(task_id)

    async def start(self) -> None:
        """Start the scheduler loop (runs in background)."""
        self._running = True
        logger.info("Cron scheduler started")

        while self._running:
            now = datetime.now()
            due_tasks = []

            for task in self._tasks.values():
                if not task.enabled:
                    continue
                if task.next_run and datetime.fromisoformat(task.next_run) <= now:
                    due_tasks.append(task)

            for task in due_tasks:
                logger.info(f"Running scheduled task: {task.task_id}")
                if self._task_callback:
                    try:
                        await self._task_callback(task)
                    except Exception as e:
                        logger.error(f"Scheduled task {task.task_id} error: {e}")

                task.last_run = now.isoformat()

                if task.recurring:
                    task.next_run = self._calculate_next_run(task.cron_expression, now)
                else:
                    # One-shot task — remove after execution
                    del self._tasks[task.task_id]

            self._save_durable_tasks()

            # Sleep until next check (every ~30 seconds)
            await asyncio.sleep(30)

    def stop(self) -> None:
        """Stop the scheduler loop."""
        self._running = False
        logger.info("Cron scheduler stopped")

    def _calculate_next_run(self, cron_expression: str, from_time: datetime) -> str:
        """Calculate the next run time from a cron expression."""
        try:
            from croniter import croniter
            cron = croniter(cron_expression, from_time)
            return cron.get_next(datetime).isoformat()
        except ImportError:
            # Fallback: simple interval parsing for "*/N" patterns
            return self._simple_next_run(cron_expression, from_time)
        except Exception:
            # If parsing fails, default to 1 hour from now
            return (from_time + timedelta(hours=1)).isoformat()

    def _simple_next_run(self, cron_expr: str, from_time: datetime) -> str:
        """
        Simple cron parser fallback.
        Supports: "*/N * * * *" (every N minutes), "0 * * * *" (hourly).
        """
        parts = cron_expr.strip().split()
        if len(parts) != 5:
            return (from_time + timedelta(minutes=10)).isoformat()

        minute_part = parts[0]
        if minute_part.startswith("*/"):
            interval = int(minute_part[2:])
            return (from_time + timedelta(minutes=interval)).isoformat()
        elif minute_part.isdigit():
            target_minute = int(minute_part)
            target_hour = int(parts[1]) if parts[1].isdigit() else from_time.hour
            next_run = from_time.replace(minute=target_minute, second=0, microsecond=0)
            if parts[1] != "*" and next_run.hour < target_hour:
                next_run = next_run.replace(hour=target_hour)
            if next_run <= from_time:
                next_run += timedelta(hours=1)
            return next_run.isoformat()

        return (from_time + timedelta(minutes=10)).isoformat()

    def _save_durable_tasks(self) -> None:
        """Save durable tasks to disk."""
        durable_tasks = [t.to_dict() for t in self._tasks.values() if t.durable]
        if not durable_tasks:
            return

        tasks_file = Path(self.workspace) / ".agent" / "scheduled_tasks.json"
        tasks_file.parent.mkdir(parents=True, exist_ok=True)
        with open(tasks_file, "w", encoding="utf-8") as f:
            json.dump(durable_tasks, f, indent=2, ensure_ascii=False)

    def _load_durable_tasks(self) -> None:
        """Load durable tasks from disk."""
        tasks_file = Path(self.workspace) / ".agent" / "scheduled_tasks.json"
        if not tasks_file.exists():
            return

        try:
            with open(tasks_file, "r", encoding="utf-8") as f:
                tasks_data = json.load(f)
            for td in tasks_data:
                task = ScheduledTask.from_dict(td)
                self._tasks[task.task_id] = task
                # Update counter
                if task.task_id.startswith("cron_"):
                    try:
                        num = int(task.task_id.split("_")[1])
                        self._task_id_counter = max(self._task_id_counter, num)
                    except (ValueError, IndexError):
                        pass
        except Exception as e:
            logger.error(f"Error loading durable tasks: {e}")


# ---------------------------------------------------------------------------
# Autonomy manager
# ---------------------------------------------------------------------------


class AutonomyManager:
    """
    Top-level autonomy manager.
    Combines checkpoint management and cron scheduling for autonomous operation.
    """

    def __init__(self, config: AutonomyConfig, workspace: str = "."):
        self.config = config
        self.workspace = workspace
        self.checkpoints = CheckpointManager(config, workspace)
        self.scheduler = CronScheduler(config, workspace)
        self._last_activity = datetime.now()
        self._idle_callback: Optional[Callable[[], Awaitable[None]]] = None

    def record_activity(self) -> None:
        """Record user activity to reset idle timer."""
        self._last_activity = datetime.now()

    def is_idle(self) -> bool:
        """Check if the agent has been idle beyond the configured threshold."""
        idle_seconds = (datetime.now() - self._last_activity).total_seconds()
        return idle_seconds > self.config.max_idle_seconds

    def save_checkpoint(self, session_id: str, messages: list[dict],
                        system_prompt: str = "", step_count: int = 0,
                        metadata: dict | None = None) -> str:
        """Save a checkpoint and clean up old ones."""
        checkpoint = Checkpoint(
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            messages=messages,
            system_prompt=system_prompt,
            step_count=step_count,
            metadata=metadata or {},
        )
        path = self.checkpoints.save(checkpoint)
        self.checkpoints.delete_old(keep=20)
        return path

    def restore_checkpoint(self) -> Optional[Checkpoint]:
        """Restore the latest checkpoint."""
        return self.checkpoints.load_latest()

    def schedule_task(self, cron_expression: str, prompt: str,
                      recurring: bool = False, durable: bool = False) -> str:
        """Schedule an autonomous task."""
        return self.scheduler.schedule(cron_expression, prompt, recurring, durable)

    async def start(self) -> None:
        """Start the autonomy system."""
        await self.scheduler.start()

    async def stop(self) -> None:
        """Stop the autonomy system."""
        self.scheduler.stop()

    @property
    def status(self) -> str:
        """Get a status summary."""
        tasks = self.scheduler.list_tasks()
        checkpoints = self.checkpoints.list_checkpoints()
        idle_seconds = (datetime.now() - self._last_activity).total_seconds()

        return (
            f"## Autonomy Status\n"
            f"- Scheduled tasks: {len(tasks)}\n"
            f"- Checkpoints: {len(checkpoints)}\n"
            f"- Idle: {idle_seconds:.0f}s (threshold: {self.config.max_idle_seconds}s)\n"
            f"- Active tasks: {sum(1 for t in tasks if t.enabled)}"
        )
