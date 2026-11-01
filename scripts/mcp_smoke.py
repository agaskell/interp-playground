"""End-to-end check: launch the MCP server over stdio, list tools, call run_prompt."""

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import Client
from mcp.client.stdio import StdioServerParameters

ROOT = Path(__file__).resolve().parent.parent


async def main() -> None:
    params = StdioServerParameters(
        command="uv",
        args=["run", "--directory", str(ROOT), "interp-playground-mcp"],
        env={**os.environ, "INTERP_PLAYGROUND_RESULTS": str(ROOT / ".smoke_evals.jsonl")},
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

        result = await client.call_tool(
            "eval_sentences", {"feature": 7272, "n_heldout": 2, "n_random": 3}
        )
        if result.is_error:
            sys.exit(f"eval_sentences error: {result.content}")
        pool = result.structured_content["sentences"]
        assert all(set(s) == {"id", "text"} for s in pool), "pool leaks fields"
        print("eval_sentences:", len(pool), "sentences")

        preds = {"zeros": [{"id": s["id"], "expected": 0} for s in pool]
                 + [{"text": "We drove across the old bridge.", "expected": 9, "on": "bridge"}]}
        result = await client.call_tool(
            "grade_explanation",
            {"feature": 7272, "explanation": "smoke", "predictions": preds,
             "n_heldout": 2, "n_random": 3},
        )
        if result.is_error:
            sys.exit(f"grade_explanation error: {result.content}")
        s = result.structured_content["sets"]["zeros"]
        assert s["n"] == len(pool) + 1
        agent_row = next(r for r in s["rows"] if r["source"] == "agent")
        assert agent_row["peak_hit"] is True, agent_row
        print("grade_explanation:", {k: s[k] for k in ("n", "f1", "spearman", "peak_hit")})


if __name__ == "__main__":
    asyncio.run(main())
