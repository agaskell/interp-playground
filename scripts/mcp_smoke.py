"""End-to-end check: launch the MCP server over stdio, list tools, call run_prompt."""

import asyncio
import json
import sys
from pathlib import Path

from mcp import Client
from mcp.client.stdio import StdioServerParameters

ROOT = Path(__file__).resolve().parent.parent


async def main() -> None:
    params = StdioServerParameters(
        command="uv", args=["run", "--directory", str(ROOT), "interp-playground-mcp"]
    )
    async with Client(params, read_timeout_seconds=300) as client:
        tools = await client.list_tools()
        print("tools:", [t.name for t in tools.tools])

        result = await client.call_tool(
            "run_prompt",
            {
                "prompt": "We drove across the old bridge into town.",
                "top_k": 0,
                "features": [7272, 4667],
            },
        )
        if result.is_error:
            sys.exit(f"tool error: {result.content}")
        data = result.structured_content
        layer = data["layers"]["12"]
        print("top feature:", json.dumps(layer["top_features"][0]))
        print("hidden:", layer["hidden"])
        for fid, w in layer["watched"].items():
            print(f"watched {fid} ({w['hidden'] or 'visible'}):", w["acts"])
        print("response bytes:", len(json.dumps(data)))

        result = await client.call_tool(
            "steer",
            {"prompt": "My favorite thing to do on weekends is", "feature": 7272,
             "scales": [0], "max_new_tokens": 8},
        )
        if result.is_error:
            sys.exit(f"tool error: {result.content}")
        data = result.structured_content
        run = data["runs"][0]
        assert run["steered"]["completion"] == data["baseline"]["completion"], "scale 0 != baseline"
        print("steer scale 0 == baseline:", repr(data["baseline"]["completion"]))


if __name__ == "__main__":
    asyncio.run(main())
