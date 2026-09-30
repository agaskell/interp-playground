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
            "run_prompt", {"prompt": "We drove across the old bridge into town.", "top_k": 2}
        )
        if result.is_error:
            sys.exit(f"tool error: {result.content}")
        data = result.structured_content
        top = data["layers"]["12"]["top_features"][0]
        print("top feature:", json.dumps(top))
        print("response bytes:", len(json.dumps(data)))


if __name__ == "__main__":
    asyncio.run(main())
