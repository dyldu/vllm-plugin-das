#!/usr/bin/env python3
"""Audit EvalScope HE32 item outcomes and preserve the evaluated prompt set."""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(directory):
    directory = Path(directory)
    predictions = list(directory.glob("predictions/*/humaneval_openai_humaneval.jsonl"))
    reviews = list(directory.glob("reviews/*/humaneval_openai_humaneval.jsonl"))
    if len(predictions) != 1 or len(reviews) != 1:
        raise ValueError("Expected one prediction file and one review file")
    pred_rows = list(map(json.loads, predictions[0].read_text().splitlines()))
    review_rows = list(map(json.loads, reviews[0].read_text().splitlines()))
    if len(pred_rows) != 32 or len(review_rows) != 32:
        raise ValueError("Expected exactly 32 predictions and reviews")
    pred = {r["index"]: r for r in pred_rows}
    review = {r["index"]: r for r in review_rows}
    if set(pred) != set(review) or len(pred) != 32:
        raise ValueError("Missing, duplicate or unexpected HE32 items")
    items = []
    for index, row in sorted(pred.items()):
        score = review[index]["sample_score"]["score"]
        output = row["model_output"]
        choices = output.get("choices", [])
        reasons = [c.get("stop_reason") for c in choices]
        prompt = row["messages"][0]["content"]
        items.append({
            "index": index, "task_id": row["metadata"]["task_id"],
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "passed": bool(score["value"]["acc"]),
            "execution": score.get("metadata", {}).get("execution_result"),
            "error": output.get("error"), "stop_reasons": reasons,
            "truncated": any(r in ("length", "max_tokens", "model_length") for r in reasons),
            "output_tokens": output.get("usage", {}).get("output_tokens"),
        })
    if {x["task_id"] for x in items} != {"HumanEval/" + str(i) for i in range(32)}:
        raise ValueError("Evaluation did not use the first 32 HumanEval tasks")
    return {"directory": str(directory), "items": items,
            "passed": sum(x["passed"] for x in items), "total": len(items),
            "errors": sum(x["error"] is not None for x in items),
            "truncated": sum(x["truncated"] for x in items)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline", help="Baseline summary JSON for paired acceptance")
    args = parser.parse_args()
    result = summarize(args.directory)
    minimum = 29
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text())
        identity = lambda rows: {(r["task_id"], r["prompt_sha256"]) for r in rows}
        if identity(result["items"]) != identity(baseline["items"]):
            raise ValueError("Candidate prompt set differs from baseline")
        minimum = max(minimum, baseline["passed"])
    result["required_pass_count"] = minimum
    result["accepted"] = (result["passed"] >= minimum and result["errors"] == 0
                          and result["truncated"] == 0)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "items"}))
    raise SystemExit(0 if result["accepted"] else 1)
