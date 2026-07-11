"""
Multi-agent architecture for the coding agent.
Enables spawning sub-agents with independent contexts for parallel work,
specialized tasks, and result aggregation.

Architecture:
- AgentTeam: orchestrates multiple sub-agents
- SubAgent: lightweight agent with its own context and tool set
- AgentType: predefined agent configurations for common tasks
"""

import asyncio
import uuid
from typing import Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum

from .config import MultiAgentConfig
from .core import Agent, AgentResult
from .tools import ToolRegistry, create_builtin_registry


@dataclass
class AgentType:
    """Predefined configuration for a specialized sub-agent."""
    name: str
    description: str
    system_prompt_override: str = ""
    tools_allowlist: list[str] | None = None  # None = all tools
    model_override: str | None = None
    max_iterations: int = 10


# Predefined agent types
AGENT_TYPES: dict[str, AgentType] = {
    "explorer": AgentType(
        name="explorer",
        description="Read-only agent for searching and exploring codebases",
        system_prompt_override="You are an explorer agent. Your job is to search and read code to gather information. You cannot modify files. Be thorough and report everything you find.",
        tools_allowlist=["read_file", "glob", "grep", "web_search", "web_fetch"],
    ),
    "reviewer": AgentType(
        name="reviewer",
        description="Code reviewer that finds bugs and issues",
        system_prompt_override="You are a code reviewer. Your job is to find bugs, security issues, and code quality problems. Be thorough and specific. Cite exact file paths and line numbers.",
        tools_allowlist=["read_file", "glob", "grep"],
    ),
    "implementer": AgentType(
        name="implementer",
        description="Agent for writing and editing code",
        system_prompt_override="You are an implementer agent. Your job is to write clean, well-documented code that follows existing patterns. Always read files before editing them.",
        tools_allowlist=["read_file", "write_file", "edit_file", "glob", "grep", "bash"],
    ),
    "researcher": AgentType(
        name="researcher",
        description="Agent for web research and information gathering",
        system_prompt_override="You are a researcher agent. Your job is to gather information from the web and synthesize findings. Be comprehensive and cite sources.",
        tools_allowlist=["web_search", "web_fetch", "read_file", "grep"],
    ),
}


@dataclass
class SubAgent:
    """A lightweight sub-agent wrapper for parallel execution."""
    agent: Agent
    agent_type: AgentType
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    async def run(self, task: str) -> AgentResult:
        """Execute a task on this sub-agent."""
        # Override system prompt if specified
        if self.agent_type.system_prompt_override:
            self.agent._init_conversation()
            self.agent._messages[0] = {
                "role": "system",
                "content": self.agent_type.system_prompt_override,
            }

        return await self.agent.run(task)


@dataclass
class TeamResult:
    """Aggregate result from a team of sub-agents."""
    results: dict[str, AgentResult] = field(default_factory=dict)  # task_id -> result
    combined_output: str = ""
    total_tokens: int = 0
    all_successful: bool = True


class AgentOrchestrator:
    """
    Orchestrates multiple sub-agents for parallel or pipelined execution.

    Patterns supported:
    - Parallel: Run N agents on independent tasks concurrently
    - Pipeline: Chain agents where output of one feeds the next
    - Map-Reduce: Fan-out to N agents, then aggregate results
    """

    def __init__(self, config: MultiAgentConfig, base_workspace: str = "."):
        self.config = config
        self.workspace = base_workspace

    def create_sub_agent(self, agent_type: str | AgentType) -> SubAgent:
        """Create a sub-agent of the specified type."""
        if isinstance(agent_type, str):
            agent_type = AGENT_TYPES.get(agent_type, AGENT_TYPES["explorer"])

        # Create a tool registry filtered by the agent type's allowlist
        registry = create_builtin_registry(self.workspace)

        if agent_type.tools_allowlist is not None:
            filtered = ToolRegistry()
            for tool_name in agent_type.tools_allowlist:
                tool = registry.get(tool_name)
                if tool:
                    filtered.register(tool)
            registry = filtered

        # Create agent with the type's configuration
        from .config import AgentConfig, LLMConfig
        config = AgentConfig(
            llm=LLMConfig(
                model=agent_type.model_override or self.config.default_model,
            ),
            workspace_dir=self.workspace,
            max_tool_iterations=agent_type.max_iterations,
        )

        agent = Agent(config=config, tool_registry=registry)
        return SubAgent(agent=agent, agent_type=agent_type)

    async def parallel(self, tasks: dict[str, str], agent_type: str | AgentType = "explorer") -> TeamResult:
        """
        Run multiple tasks in parallel with separate sub-agents.

        Args:
            tasks: {task_id: task_description} mapping
            agent_type: The type of agent to spawn for each task

        Returns:
            TeamResult with aggregated results.
        """
        sub_agents = {
            task_id: self.create_sub_agent(agent_type)
            for task_id in tasks
        }

        async def run_one(task_id: str, sub: SubAgent):
            try:
                result = await sub.run(tasks[task_id])
                return task_id, result
            except Exception as e:
                return task_id, AgentResult(
                    content=f"Sub-agent error: {e}",
                    success=False,
                    error=str(e),
                )

        # Execute all in parallel
        coros = [run_one(tid, sub) for tid, sub in sub_agents.items()]
        results_list = await asyncio.gather(*coros)

        # Aggregate
        results = dict(results_list)
        all_ok = all(r.success for r in results.values())
        total_tokens = sum(r.total_tokens for r in results.values())

        # Build combined output
        combined_parts = []
        for tid, r in results.items():
            combined_parts.append(f"## {tid}\n{r.content}\n")

        return TeamResult(
            results=results,
            combined_output="\n".join(combined_parts),
            total_tokens=total_tokens,
            all_successful=all_ok,
        )

    async def pipeline(self, task: str, stages: list[tuple[str, str | AgentType]]) -> AgentResult:
        """
        Run a task through a pipeline of specialized agents.

        Each stage receives the output of the previous stage.
        Useful for: explore → plan → implement → review

        Args:
            task: The initial task description
            stages: List of (stage_name, agent_type) tuples

        Returns:
            Final AgentResult after all pipeline stages.
        """
        current_input = task

        for stage_name, agent_type in stages:
            sub = self.create_sub_agent(agent_type)
            prompt = f"Stage: {stage_name}\n\nTask:\n{current_input}"
            result = await sub.run(prompt)

            if not result.success:
                return result  # Fail fast

            current_input = result.content

        return AgentResult(content=current_input, success=True)

    async def map_reduce(self, task: str, items: list[str],
                         map_agent: str | AgentType = "explorer",
                         reduce_agent: str | AgentType = "explorer") -> TeamResult:
        """
        Map: process each item independently in parallel.
        Reduce: aggregate all results into a summary.

        Args:
            task: The overall task description
            items: List of items to process
            map_agent: Agent type for processing each item
            reduce_agent: Agent type for aggregating results
        """
        # Map phase: process each item in parallel
        map_tasks = {
            f"item_{i}": f"{task}\n\nFocus on this specific item: {item}"
            for i, item in enumerate(items)
        }

        map_result = await self.parallel(map_tasks, map_agent)

        if not map_result.all_successful:
            return map_result

        # Reduce phase: aggregate
        reducer = self.create_sub_agent(reduce_agent)
        reduce_prompt = (
            f"Task: {task}\n\n"
            f"Below are results from analyzing {len(items)} items individually.\n"
            f"Synthesize these into a comprehensive summary:\n\n"
            f"{map_result.combined_output}"
        )

        reduce_result = await reducer.run(reduce_prompt)

        return TeamResult(
            results={"map": AgentResult(content=map_result.combined_output, success=True),
                      "reduce": reduce_result},
            combined_output=reduce_result.content,
            total_tokens=map_result.total_tokens + reduce_result.total_tokens,
            all_successful=reduce_result.success,
        )

    async def close_all(self) -> None:
        """Not needed for inline agents, but provided for API consistency."""
        pass
