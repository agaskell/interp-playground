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

        for name, args in [
            ("feature_info", {"feature": 7272}),
            ("logit_lens", {"prompt": "The Eiffel Tower is located in the city of", "top_k": 1}),
            ("ablate", {"prompt": "We drove across the old bridge.", "feature": 7272}),
        ]:
            result = await client.call_tool(name, args)
            if result.is_error:
                sys.exit(f"{name} error: {result.content}")
            d = result.structured_content
            brief = {
                "feature_info": lambda: f"{d['label']!r}, {len(d['top_examples'] or [])} examples",
                "logit_lens": lambda: f"{d['final_prediction']!r} p_final@25={d['layers'][-1]['p_final']}",
                "ablate": lambda: f"kl={d['next_token']['kl']} active={len(d['active_positions'])}",
            }[name]()
            print(f"{name}: {brief} ({len(json.dumps(d))} bytes)")


if __name__ == "__main__":
    asyncio.run(main())
