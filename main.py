#!/usr/bin/env python3
"""
From-Zero Coding Agent — Main Entry Point

A from-scratch implementation of a coding agent with:
  Phase 1: Agent loop, tools, system prompt, CLI, streaming, permissions, context
  Phase 2: Memory, skills, plan mode, multi-agent, MCP integration
  Phase 3: Autonomy, scheduling, checkpoint/resume

Usage:
    python main.py                    # Interactive mode
    python main.py "fix the bug"      # Single prompt mode
    python main.py --config config.yaml  # Custom config
    python main.py --help             # Show help
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from agent.config import AgentConfig
from agent.cli import run_cli, run_cli_sync


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="From-Zero Coding Agent — a from-scratch AI coding assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Interactive REPL
  python main.py "Explain this codebase"  # Single prompt
  python main.py --config agent.yaml      # Custom config file
  python main.py --model gpt-4o           # Override model
  python main.py --workspace /path/to/proj  # Set workspace
  python main.py --debug                  # Enable debug output
        """,
    )
    parser.add_argument(
        "prompt", nargs="?", default=None,
        help="Single prompt to run (non-interactive mode)",
    )
    parser.add_argument(
        "-c", "--config", type=str, default=None,
        help="Path to YAML/JSON configuration file",
    )
    parser.add_argument(
        "-w", "--workspace", type=str, default=".",
        help="Workspace directory (default: current directory)",
    )
    parser.add_argument(
        "-m", "--model", type=str, default=None,
        help="LLM model to use (e.g., gpt-4o, claude-sonnet-5, deepseek-v3)",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="API key for the LLM provider",
    )
    parser.add_argument(
        "--api-base", type=str, default=None,
        help="API base URL for the LLM provider",
    )
    parser.add_argument(
        "--provider", type=str, default=None,
        choices=["openai", "anthropic", "deepseek"],
        help="LLM provider",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--version", action="store_true",
        help="Show version and exit",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> AgentConfig:
    """Build the agent configuration from CLI arguments."""
    # Start with environment-based config
    config = AgentConfig.from_env()

    # Override with config file if provided
    if args.config:
        config = AgentConfig.from_file(args.config)

    # Apply CLI overrides
    if args.workspace:
        config.workspace_dir = args.workspace
    if args.model:
        config.llm.model = args.model
    if args.api_key:
        config.llm.api_key = args.api_key
    if args.api_base:
        config.llm.api_base = args.api_base
    if args.provider:
        config.llm.provider = args.provider
    if args.debug:
        config.debug = True

    return config


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if args.version:
        from agent import __version__
        print(f"From-Zero Coding Agent v{__version__}")
        return

    # Build configuration
    config = build_config(args)

    # Setup logging
    if config.debug:
        import logging
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    # Run the agent
    try:
        run_cli_sync(config, args.prompt)
    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye!")
    except Exception as e:
        print(f"\nFatal error: {e}")
        if config.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
