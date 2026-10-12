"""Call one of the MCP tools from the shell, without going through the server.

Usage: uv run python scripts/tool.py TOOL [key=value ...]

TOOL is run_prompt, steer, feature_info, logit_lens or ablate; keys are the
tool's parameters. Values are parsed as JSON when they are valid JSON
(numbers, lists, true/false), otherwise kept as strings. Examples:

    scripts/tool.py logit_lens prompt="The capital of France is" top_k=3
    scripts/tool.py steer prompt="..." feature=9193 target_tokens='["grief","loss"]'

Each run loads the model from scratch (about a minute); the MCP server keeps
it warm, so for many calls in a row ask Claude to use the server instead.
"""

import json
import sys

from interp_playground import server


def parse(value: str):
    try:
        return json.loads(value)
    except ValueError:
        return value


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in server.TOOLS:
        sys.exit(f"usage: tool.py {{{','.join(server.TOOLS)}}} [key=value ...]")
    name, *pairs = sys.argv[1:]
    kwargs = {k: parse(v) for k, v in (p.split("=", 1) for p in pairs)}
    print(json.dumps(getattr(server, name)(**kwargs), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
