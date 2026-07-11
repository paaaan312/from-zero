"""
Plan Mode system for the coding agent.
Implements a plan-before-execute workflow:
1. Explore — gather information about the codebase
2. Design — create an implementation plan
3. Present — show the plan to the user
4. Approve — get user sign-off
5. Execute — carry out the plan
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable, Awaitable


class PlanPhase(Enum):
    IDLE = "idle"
    EXPLORING = "exploring"
    DESIGNING = "designing"
    PRESENTING = "presenting"
    WAITING_APPROVAL = "waiting_approval"
    EXECUTING = "executing"
    DONE = "done"
    REJECTED = "rejected"


@dataclass
class PlanStep:
    """A single step in an execution plan."""
    order: int
    description: str
    files_to_modify: list[str] = field(default_factory=list)
    files_to_create: list[str] = field(default_factory=list)
    estimated_complexity: str = "medium"  # low, medium, high
    dependencies: list[int] = field(default_factory=list)  # step numbers this depends on
    status: str = "pending"  # pending, in_progress, completed, skipped


@dataclass
class Plan:
    """An execution plan."""
    title: str = ""
    summary: str = ""
    steps: list[PlanStep] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)
    phase: PlanPhase = PlanPhase.IDLE
    approved: bool = False

    def format_for_display(self) -> str:
        """Format the plan as a readable string."""
        lines = [
            f"# Plan: {self.title}",
            "",
            self.summary,
            "",
            "## Steps",
        ]
        for step in self.steps:
            status_icon = {
                "pending": "⬜",
                "in_progress": "🔄",
                "completed": "✅",
                "skipped": "⏭️",
            }.get(step.status, "⬜")

            deps = f" (depends on: {', '.join(map(str, step.dependencies))})" if step.dependencies else ""
            files = f"\n  Files: {', '.join(step.files_to_modify + step.files_to_create)}" if step.files_to_modify or step.files_to_create else ""
            lines.append(f"{status_icon} **Step {step.order}**: {step.description} [{step.estimated_complexity}]{deps}{files}")

        if self.risks:
            lines.append("\n## Risks")
            for r in self.risks:
                lines.append(f"- ⚠️ {r}")

        if self.alternatives:
            lines.append("\n## Alternatives Considered")
            for a in self.alternatives:
                lines.append(f"- {a}")

        return "\n".join(lines)

    def format_short(self) -> str:
        """A short summary for the plan."""
        lines = [f"**Plan: {self.title}**"]
        lines.append(self.summary)
        for step in self.steps:
            status_icon = {
                "pending": "⬜", "in_progress": "🔄",
                "completed": "✅", "skipped": "⏭️",
            }.get(step.status, "⬜")
            lines.append(f"  {status_icon} {step.order}. {step.description}")
        return "\n".join(lines)


class PlanMode:
    """
    Manages the plan-before-execute workflow.

    Usage:
        plan_mode = PlanMode()
        await plan_mode.enter()          # Start planning
        plan = await plan_mode.explore()  # Gather context
        await plan_mode.present(plan)     # Show to user
        if await plan_mode.request_approval():
            await plan_mode.execute(plan)  # Carry out
    """

    def __init__(self):
        self.phase = PlanPhase.IDLE
        self._current_plan: Optional[Plan] = None
        self._context_gathered: dict = {}
        self._approval_callback: Optional[Callable[[Plan], Awaitable[bool]]] = None

    def set_approval_handler(self, callback: Callable[[Plan], Awaitable[bool]]) -> None:
        """Set the callback for requesting user approval."""
        self._approval_callback = callback

    async def enter(self) -> Plan:
        """Enter plan mode and create a new plan template."""
        self.phase = PlanPhase.EXPLORING
        self._current_plan = Plan(
            title="New Plan",
            summary="Gathering requirements...",
            phase=PlanPhase.EXPLORING,
        )
        self._context_gathered = {}
        return self._current_plan

    async def explore(self, findings: dict | None = None) -> Plan:
        """
        Gather context and findings during the exploration phase.
        The agent should read files, search code, and understand the problem
        before designing a solution.
        """
        if findings:
            self._context_gathered.update(findings)

        self.phase = PlanPhase.DESIGNING
        if self._current_plan:
            self._current_plan.phase = PlanPhase.DESIGNING

        return self._current_plan

    def design(self, title: str, summary: str, steps: list[dict],
               risks: list[str] | None = None, alternatives: list[str] | None = None) -> Plan:
        """
        Design a concrete plan with steps.

        Each step dict: {
            "description": str,
            "files_to_modify": [str],
            "files_to_create": [str],
            "complexity": "low"|"medium"|"high",
            "depends_on": [int],
        }
        """
        plan_steps = []
        for i, s in enumerate(steps, 1):
            plan_steps.append(PlanStep(
                order=i,
                description=s.get("description", f"Step {i}"),
                files_to_modify=s.get("files_to_modify", []),
                files_to_create=s.get("files_to_create", []),
                estimated_complexity=s.get("complexity", "medium"),
                dependencies=s.get("depends_on", []),
            ))

        self._current_plan = Plan(
            title=title,
            summary=summary,
            steps=plan_steps,
            risks=risks or [],
            alternatives=alternatives or [],
            phase=PlanPhase.DESIGNING,
        )
        self.phase = PlanPhase.PRESENTING
        return self._current_plan

    async def request_approval(self) -> bool:
        """Request user approval for the current plan."""
        if not self._current_plan:
            return False

        self.phase = PlanPhase.WAITING_APPROVAL

        if self._approval_callback:
            approved = await self._approval_callback(self._current_plan)
        else:
            # In non-interactive mode, auto-approve
            approved = True

        if approved:
            self._current_plan.approved = True
            self.phase = PlanPhase.EXECUTING
            self._current_plan.phase = PlanPhase.EXECUTING
        else:
            self._current_plan.approved = False
            self.phase = PlanPhase.REJECTED
            self._current_plan.phase = PlanPhase.REJECTED

        return approved

    def start_step(self, step_number: int) -> None:
        """Mark a plan step as in progress."""
        if self._current_plan:
            for step in self._current_plan.steps:
                if step.order == step_number:
                    step.status = "in_progress"
                    break

    def complete_step(self, step_number: int) -> None:
        """Mark a plan step as completed."""
        if self._current_plan:
            for step in self._current_plan.steps:
                if step.order == step_number:
                    step.status = "completed"
                    break

    def skip_step(self, step_number: int, reason: str = "") -> None:
        """Skip a plan step."""
        if self._current_plan:
            for step in self._current_plan.steps:
                if step.order == step_number:
                    step.status = "skipped"
                    if reason:
                        step.description += f" [SKIPPED: {reason}]"
                    break

    def is_complete(self) -> bool:
        """Check if all plan steps are done."""
        if not self._current_plan:
            return True
        return all(s.status in ("completed", "skipped") for s in self._current_plan.steps)

    def exit(self) -> Plan | None:
        """Exit plan mode."""
        self.phase = PlanPhase.DONE
        if self._current_plan:
            self._current_plan.phase = PlanPhase.DONE
        plan = self._current_plan
        self._current_plan = None
        return plan

    @property
    def current_plan(self) -> Optional[Plan]:
        return self._current_plan

    @property
    def is_active(self) -> bool:
        return self.phase not in (PlanPhase.IDLE, PlanPhase.DONE, PlanPhase.REJECTED)
