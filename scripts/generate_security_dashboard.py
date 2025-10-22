#!/usr/bin/env python3
"""Generate a consolidated security dashboard report.

This script inspects SARIF/JSON artifacts from CodeQL, SAST tooling
(Bandit/Semgrep) and pip-audit. It also queries Dependabot alerts via
GitHub's REST API when a token is available. The result is exported as a
Markdown summary (plus optional JSON metadata) ready to publish under
``docs/security/reports``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import sys
import typing as t
import urllib.error
import urllib.parse
import urllib.request

SEVERITY_ORDER = ["critical", "high", "medium", "low", "info"]
CANONICAL_TOOL_NAMES = {
    "codeql": "CodeQL",
    "bandit": "Bandit",
    "bandit security linter": "Bandit",
    "semgrep": "Semgrep",
}


class ToolSummary(t.TypedDict, total=False):
    name: str
    total: int
    critical: int
    high: int
    medium: int
    low: int
    info: int
    missing: bool
    artifacts: t.List[str]
    notes: t.List[str]


class DependabotSummary(t.TypedDict, total=False):
    total: int | None
    critical: int
    high: int
    medium: int
    low: int
    url: str | None
    error: str | None


def _blank_tool(name: str) -> ToolSummary:
    return ToolSummary(
        name=name,
        total=0,
        critical=0,
        high=0,
        medium=0,
        low=0,
        info=0,
        missing=True,
        artifacts=[],
        notes=[],
    )


def _normalise_severity(raw: t.Any) -> str:
    if raw is None:
        return "info"
    value = str(raw).strip().lower()
    if not value:
        return "info"
    mapping = {
        "err": "high",
        "error": "high",
        "errors": "high",
        "warn": "medium",
        "warning": "medium",
        "note": "low",
        "none": "info",
        "moderate": "medium",
        "informational": "info",
    }
    if value in mapping:
        return mapping[value]
    if value in SEVERITY_ORDER:
        return value
    # Attempt to parse numeric severities (CodeQL uses 0-10 scale).
    try:
        numeric = float(value)
    except ValueError:
        pass
    else:
        if numeric >= 9.0:
            return "critical"
        if numeric >= 7.0:
            return "high"
        if numeric >= 4.0:
            return "medium"
        if numeric > 0:
            return "low"
        return "info"
    return "info"


def _accumulate(summary: ToolSummary, severity: str) -> None:
    severity = severity if severity in summary else "info"
    summary[severity] = int(summary.get(severity, 0)) + 1
    summary["total"] = int(summary.get("total", 0)) + 1


def _relativise(path: pathlib.Path, root: pathlib.Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _canonical_tool_name(raw: str | None, fallback: str) -> str:
    if not raw:
        return fallback
    normalized = raw.strip().lower()
    return CANONICAL_TOOL_NAMES.get(normalized, raw.strip())


def gather_sarif(root: pathlib.Path, repo_root: pathlib.Path) -> dict[str, ToolSummary]:
    results: dict[str, ToolSummary] = {}
    if not root.exists():
        return results
    for sarif_path in root.rglob("*.sarif"):
        try:
            data = json.loads(sarif_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            tool_name = f"SARIF ({sarif_path.name})"
            summary = results.setdefault(tool_name, _blank_tool(tool_name))
            summary["notes"].append(f"Error reading {sarif_path.name}: {exc}")
            summary["artifacts"].append(_relativise(sarif_path, repo_root))
            summary["missing"] = False
            continue
        runs = data.get("runs", [])
        if not runs:
            tool_name = f"SARIF ({sarif_path.name})"
            summary = results.setdefault(tool_name, _blank_tool(tool_name))
            summary["notes"].append("SARIF file without recorded runs")
            summary["artifacts"].append(_relativise(sarif_path, repo_root))
            summary["missing"] = False
            continue
        for run in runs:
            raw_name = run.get("tool", {}).get("driver", {}).get("name")
            tool = _canonical_tool_name(raw_name, f"SARIF ({sarif_path.stem})")
            summary = results.setdefault(tool, _blank_tool(tool))
            summary["name"] = tool
            summary["missing"] = False
            relative_path = _relativise(sarif_path, repo_root)
            if relative_path not in summary["artifacts"]:
                summary["artifacts"].append(relative_path)
            for result in run.get("results", []) or []:
                properties = result.get("properties", {}) or {}
                severity = None
                for key in (
                    "security-severity",
                    "securitySeverity",
                    "problem.severity",
                    "severity",
                    "impact",
                ):
                    if key in properties:
                        severity = properties[key]
                        break
                if severity is None:
                    severity = result.get("level")
                if severity is None:
                    severity = run.get("defaultConfiguration", {}).get("level")
                level = _normalise_severity(severity)
                _accumulate(summary, level)
    for summary in results.values():
        summary["artifacts"] = list(dict.fromkeys(summary.get("artifacts", [])))
    return results


def gather_pip_audit(report_path: pathlib.Path, repo_root: pathlib.Path) -> ToolSummary:
    summary = _blank_tool("pip-audit")
    if not report_path.exists():
        return summary
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        summary["notes"].append(f"Error reading pip-audit.json: {exc}")
        summary["missing"] = False
        summary["artifacts"].append(_relativise(report_path, repo_root))
        return summary
    entries: list = (
        data if isinstance(data, list) else data.get("dependencies", []) or []
    )
    for entry in entries:
        vulns = entry.get("vulns") or entry.get("vulnerabilities") or []
        for vuln in vulns:
            severity = (
                vuln.get("severity")
                or vuln.get("advisory", {}).get("severity")
                or vuln.get("metadata", {}).get("severity")
            )
            level = _normalise_severity(severity)
            _accumulate(summary, level)
    summary["missing"] = False
    summary["artifacts"].append(_relativise(report_path, repo_root))
    return summary


def fetch_dependabot(repo: str, token: str | None) -> DependabotSummary:
    base_summary: DependabotSummary = {
        "total": None,
        "critical": 0,
        "high": 0,
        "medium": 0,
        "low": 0,
        "url": None,
        "error": None,
    }
    if not repo or not token:
        base_summary["error"] = "Repository or token unavailable to query Dependabot"
        return base_summary
    api_url = f"https://api.github.com/repos/{repo}/dependabot/alerts"
    params = {"state": "open", "per_page": 100, "page": 1}
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "tnfr-security-dashboard",
    }
    total = 0
    try:
        while True:
            url = api_url + "?" + urllib.parse.urlencode(params)
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, list):
                base_summary["error"] = "Unexpected response from the Dependabot API"
                break
            for alert in payload:
                severity = (alert.get("security_advisory") or {}).get("severity") or (
                    alert.get("security_vulnerability") or {}
                ).get("severity")
                level = _normalise_severity(severity)
                if level not in base_summary:
                    level = "low"
                base_summary[level] = int(base_summary.get(level, 0)) + 1
                total += 1
            if len(payload) < params["per_page"]:
                break
            params["page"] += 1
        base_summary["total"] = total
        base_summary["url"] = f"https://github.com/{repo}/security/dependabot"
    except urllib.error.HTTPError as exc:  # pragma: no cover - network errors
        base_summary["error"] = f"HTTP {exc.code} while querying Dependabot"
    except urllib.error.URLError as exc:  # pragma: no cover - network errors
        base_summary["error"] = f"Network error while querying Dependabot: {exc.reason}"
    return base_summary


def render_markdown(
    generated_at: dt.datetime,
    run_url: str | None,
    tool_summaries: dict[str, ToolSummary],
    dependabot: DependabotSummary,
) -> str:
    lines: list[str] = []
    timestamp = generated_at.strftime("%Y-%m-%d %H:%M:%S UTC")
    lines.append(f"# Security dashboard report ({timestamp})")
    lines.append("")
    if run_url:
        lines.append(f"- Consolidated run: [{run_url}]({run_url})")
    lines.append(
        "- Aggregated sources: CodeQL, Bandit, Semgrep, pip-audit, and Dependabot alerts."
    )
    lines.append("")
    lines.append("## Findings summary")
    lines.append("")
    header = "| Tool | Findings | Critical | High | Medium | Low | Info | Status |"
    lines.append(header)
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    if not tool_summaries:
        lines.append("| *(no data)* | — | — | — | — | — | — | No artifacts located |")
    else:
        for tool in sorted(tool_summaries.values(), key=lambda x: x["name"].lower()):
            total = tool.get("total", 0)
            if tool.get("missing"):
                status = "Missing artifact"
                total_display = "—"
            else:
                status = "OK"
                total_display = str(total)
            row = "| {name} | {total} | {critical} | {high} | {medium} | {low} | {info} | {status} |".format(
                name=tool["name"],
                total=total_display,
                critical=tool.get("critical", 0),
                high=tool.get("high", 0),
                medium=tool.get("medium", 0),
                low=tool.get("low", 0),
                info=tool.get("info", 0),
                status=status,
            )
            lines.append(row)
    lines.append("")

    lines.append("## Dependabot alerts")
    lines.append("")
    if dependabot.get("total") is None:
        lines.append(
            "- Unable to query Dependabot: {}".format(
                dependabot.get("error", "unknown reason")
            )
        )
    else:
        total = dependabot.get("total", 0)
        lines.append(f"- Open alerts: **{total}**")
        lines.append(
            "- Severity: Critical {critical}, High {high}, Medium {medium}, Low {low}".format(
                critical=dependabot.get("critical", 0),
                high=dependabot.get("high", 0),
                medium=dependabot.get("medium", 0),
                low=dependabot.get("low", 0),
            )
        )
        if dependabot.get("url"):
            lines.append(
                f"- Dependabot dashboard: [{dependabot['url']}]({dependabot['url']})"
            )
    lines.append("")

    lines.append("## Included artifacts")
    lines.append("")
    artifacts: list[str] = []
    for tool in tool_summaries.values():
        artifacts.extend(tool.get("artifacts", []))
    if artifacts:
        for artifact in sorted(dict.fromkeys(artifacts)):
            lines.append(f"- `{artifact}`")
    else:
        lines.append("- (No artifacts were found in this run)")
    lines.append("")

    for tool in tool_summaries.values():
        if tool.get("notes"):
            lines.append(f"### Notes for {tool['name']}")
            lines.append("")
            for note in tool["notes"]:
                lines.append(f"- {note}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=pathlib.Path,
        default=pathlib.Path("artifacts"),
        help="Base directory where artifacts were downloaded",
    )
    parser.add_argument(
        "--pip-audit",
        type=pathlib.Path,
        default=None,
        help="Path to the pip-audit JSON report (optional)",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        required=True,
        help="Output path for the Markdown report",
    )
    parser.add_argument(
        "--json-output",
        type=pathlib.Path,
        default=None,
        help="Additional path to export the data in JSON",
    )
    parser.add_argument(
        "--run-url",
        type=str,
        default=None,
        help="URL of the GitHub Actions run that consolidates the results",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    repo_root = pathlib.Path.cwd()
    artifact_root: pathlib.Path = args.artifact_root

    sarif_summaries = gather_sarif(artifact_root, repo_root)

    pip_audit_path = args.pip_audit
    if pip_audit_path is None:
        pip_audit_path = artifact_root / "pip-audit" / "pip-audit.json"
    pip_summary = gather_pip_audit(pip_audit_path, repo_root)

    # Merge tool summaries ensuring expected names are present.
    expected_tools = {
        "CodeQL": None,
        "Bandit": None,
        "Semgrep": None,
        "pip-audit": pip_summary,
    }
    merged: dict[str, ToolSummary] = {}
    for tool_name, summary in sarif_summaries.items():
        merged[tool_name] = summary
        if tool_name in expected_tools:
            expected_tools[tool_name] = summary
    if expected_tools["pip-audit"] is None:
        merged["pip-audit"] = pip_summary
    else:
        merged.setdefault("pip-audit", pip_summary)

    for expected in ("CodeQL", "Bandit", "Semgrep"):
        if expected_tools.get(expected) is None:
            merged.setdefault(expected, _blank_tool(expected))

    dependabot = fetch_dependabot(
        repo=os.environ.get("GITHUB_REPOSITORY", ""),
        token=os.environ.get("GITHUB_TOKEN"),
    )

    generated_at = dt.datetime.utcnow().replace(microsecond=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(generated_at, args.run_url, merged, dependabot)
    args.output.write_text(markdown, encoding="utf-8")

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        json_payload = {
            "generated_at": generated_at.isoformat() + "Z",
            "run_url": args.run_url,
            "tools": merged,
            "dependabot": dependabot,
        }
        args.json_output.write_text(
            json.dumps(json_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
