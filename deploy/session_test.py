"""
End-to-end test of the multi-source session flow against a running API.

Usage: python deploy/session_test.py [base_url] [data_dir]
  base_url defaults to http://127.0.0.1:8096, data_dir to input/multisource
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8096").rstrip("/") + "/api"
DATA = Path(sys.argv[2] if len(sys.argv) > 2 else "input/multisource")
CHUNK = 1024 * 1024  # small chunks so the test exercises several of them


def check(res: requests.Response) -> dict:
    if not res.ok:
        raise SystemExit(f"{res.request.method} {res.url} -> {res.status_code}: {res.text[:500]}")
    return res.json() if res.content else {}


def upload(session_id: str, path: Path) -> dict:
    size = path.stat().st_size
    info = check(requests.post(f"{BASE}/sessions/{session_id}/files",
                               json={"filename": path.name, "size_bytes": size, "chunk_bytes": CHUNK}))
    fid = info["file_id"]
    with open(path, "rb") as fh:
        index = 0
        while True:
            chunk = fh.read(CHUNK)
            if not chunk:
                break
            # Re-sending a chunk must be rejected with 409 (resume protocol).
            if index == 1:
                dup = requests.put(f"{BASE}/sessions/{session_id}/files/{fid}/chunks/0", data=b"x")
                assert dup.status_code == 409, dup.text
            check(requests.put(f"{BASE}/sessions/{session_id}/files/{fid}/chunks/{index}", data=chunk))
            index += 1
    t = time.time()
    done = check(requests.post(f"{BASE}/sessions/{session_id}/files/{fid}/complete"))
    p = done["profile"]
    print(f"  {path.name}: {p['row_count']} rows x {p['column_count']} cols, "
          f"dates={p['date_columns']} ids={p['id_columns']} ({time.time() - t:.1f}s)")
    return done


def main() -> None:
    session = check(requests.post(f"{BASE}/sessions"))
    sid = session["session_id"]
    print("session", sid)

    files = {}
    for name in ("loan_dump.csv", "bureau.json", "bank_statement.csv"):
        files[name] = upload(sid, DATA / name)

    t = time.time()
    proposal = check(requests.post(f"{BASE}/sessions/{sid}/analyse"))
    print(f"analyse ({time.time() - t:.1f}s): primary={files_by_id(files)[proposal['primary_file_id']]} "
          f"roles={ {files_by_id(files)[k]: v for k, v in proposal['roles'].items()} } "
          f"target={proposal['suggested_target_col']} time={proposal['time_col']}")
    for j in proposal["joins"]:
        sel = j["selected"]
        print(f"  join {j['filename']} ({j['role']}): {sel['primary_col']} = {sel['file_col']} "
              f"match {sel['match_rate']:.0%}, {sel['rows_per_key']} rows/key; "
              f"{len(j['candidates'])} candidates")
        assert sel["match_rate"] > 0.8, "expected a strong key match"

    joined = check(requests.post(f"{BASE}/sessions/{sid}/join", json={
        "primary_file_id": proposal["primary_file_id"],
        "joins": [{"file_id": j["file_id"], **{k: j["selected"][k] for k in ("primary_col", "file_col")}}
                  for j in proposal["joins"]],
    }))
    dp = joined["dataset_profile"]
    print(f"joined: {dp['row_count']} rows x {dp['column_count']} cols; stats={joined['join']['stats']}")
    assert dp["row_count"] == files["loan_dump.csv"]["profile"]["row_count"], "join must keep one row per loan"

    target_col = proposal["suggested_target_col"] or "default_90dpd"
    t = time.time()
    target = check(requests.post(f"{BASE}/sessions/{sid}/target", json={"target_col": target_col}))
    print(f"target ({time.time() - t:.1f}s): {target_col} positive_rate={target['positive_rate']} "
          f"split={target['split_strategy']} time_col={target['time_col']}")
    print("  top IV:", [(r["feature"], r["iv"]) for r in target["iv"][:6]])
    print("  flags:", [(f["column"], f["kind"]) for f in target["flags"]])
    print("  excluded:", target["excluded_by_default"])
    flagged: dict = {}
    for f in target["flags"]:
        flagged.setdefault(f["column"], set()).add(f["kind"])
    assert "pii" in flagged.get("customer_name", ()) and "pii" in flagged.get("mobile", ()), flagged
    assert "leakage" in flagged.get("writeoff_amount", ()), flagged
    assert "identifier" in flagged.get("loan_id", ()), flagged

    job = check(requests.post(f"{BASE}/sessions/{sid}/train", json={
        "target_col": target_col, "exclude_cols": target["excluded_by_default"], "time_col": target["time_col"], "tier": "quick",
    }))
    t = time.time()
    while True:
        status = check(requests.get(f"{BASE}/jobs/{job['job_id']}"))
        if status["status"] in ("succeeded", "failed"):
            break
        time.sleep(2)
    print(f"job {status['status']} in {time.time() - t:.0f}s: {status['message']}")
    if status["status"] != "succeeded":
        raise SystemExit(1)
    result = check(requests.get(f"{BASE}/jobs/{job['job_id']}/result"))
    ds = result["dataset"]
    print(f"  split={ds['split_strategy']} ranges={ds.get('time_ranges')} rows={ds['n_rows']} feats={ds['n_features']}")
    print("  excluded:", ds["excluded_columns"])
    print("  leaderboard:", [(r["model"], round(r["roc_auc"], 3)) for r in result["leaderboard"]])
    best = result["results"][0]
    print("  top drivers:", [f["feature"] for f in best["feature_importance"][:6]])
    assert set(best["evaluations"]) == {"train", "validation", "calibration", "test", "oot"}
    assert result["warnings"] and result["drift"]["oot"]["score"]["psi"] >= 0
    assert result["diagnostics"]["auc_bootstrap_95_interval"] is not None
    assert ds["feature_iv"][0]["bins"] and ds["feature_glossary"]
    print("  cohorts:", ds["cohort_sizes"], "| oot:", ds["split"]["oot_status"])
    print("  warnings:", len(result["warnings"]), "| drift oot:", result["drift"]["oot"]["score"]["status"],
          "| CI:", result["diagnostics"]["auc_bootstrap_95_interval"])
    gone = requests.get(f"{BASE}/sessions/{sid}")
    assert gone.status_code == 404, "session must be deleted after training"
    print("session deleted after training: OK")

    # Score new applications with the saved bundle.
    model_id = result["model_id"]
    schema = check(requests.get(f"{BASE}/models/{model_id}/schema"))
    defaults = {f["name"]: f["default"] for f in schema["fields"]}
    risky = dict(defaults, bureau__score_value=520, bank_statement__bounces=4)
    scored = check(requests.post(f"{BASE}/models/{model_id}/score", json={"records": [defaults, risky]}))["results"]
    print("  scored:", [(s["nu_score"], s["band"], s["base_score"], s["ai_adjustment"], [r["feature"] for r in s["reasons"][:3]]) for s in scored])
    assert scored[1]["nu_score"] < scored[0]["nu_score"], "riskier record must score lower"
    print("  champion:", result["champion"]["name"], "blend members:", result["nuscore"].get("base_model"),
          [m["model"] for m in result["leaderboard"] if m["family"] == "ensemble"])

    # Batch scoring with renamed columns.
    import io
    import pandas as pd
    batch = pd.DataFrame([defaults] * 5)
    batch = batch.rename(columns={"bureau__score_value": "Bureau Score", "ltv": "LTV (%)"})
    batch["LTV (%)"] = batch["LTV (%)"] * 100
    body = batch.to_csv(index=False).encode()
    res = check(requests.post(f"{BASE}/models/{model_id}/score-file?filename=batch.csv", data=body,
                              headers={"Content-Type": "application/octet-stream"}))
    methods = {m["schema_col"]: m["method"] for m in res["mapping"]}
    print("  batch:", res["rows"], "rows; mapping for renamed:", methods.get("bureau__score_value"), methods.get("ltv"),
          "missing:", res["missing"])
    csv = requests.get(f"{BASE}/models/{model_id}/score-file/{res['file_id']}/download")
    assert csv.ok and b"nu_score" in csv.content.splitlines()[0], "scored CSV download"
    print("  scored CSV download: OK")


def files_by_id(files: dict) -> dict:
    return {info["file_id"]: name for name, info in files.items()}


if __name__ == "__main__":
    main()
