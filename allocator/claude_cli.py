"""Shared Claude CLI utility functions."""

import logging
import subprocess

logger = logging.getLogger(__name__)


def call_claude_cli(
    prompt: str,
    timeout: int = 120,
    model: str = "haiku",
    output_format: str = "text",
) -> str | None:
    """
    Call Claude CLI with the given prompt.

    Returns the CLI output stripped of whitespace, or None if the call failed.
    """
    try:
        cmd = ["claude", "-p", "--model", model]
        if output_format != "text":
            cmd.extend(["--output-format", output_format])

        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            logger.warning(f"Claude CLI error: {result.stderr}")
            return None

        return result.stdout.strip()

    except subprocess.TimeoutExpired:
        logger.warning("Claude CLI timed out")
        return None
    except FileNotFoundError:
        logger.error(
            "Claude CLI not found. Install: npm install -g @anthropic-ai/claude-code"
        )
        return None
    except Exception as e:
        logger.warning(f"Claude CLI failed: {e}")
        return None
