#!/usr/bin/env python3
"""Render selected TNFR markdown documents to static HTML for gh-pages."""

import argparse
import html
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List
from urllib.parse import urlsplit

import markdown  # type: ignore[import]
from markdown.extensions import Extension  # type: ignore[import]
from markdown.treeprocessors import Treeprocessor  # type: ignore[import]

REPO_ROOT = Path(__file__).resolve().parents[1]
SITE_BASE = "/TNFR-Python-Engine/"
REPO_URL = "https://github.com/fermga/TNFR-Python-Engine"
VERSION = "0.0.2"
DOI = "10.5281/zenodo.17764207"
MARKDOWN_EXTENSIONS = [
    "toc",
    "fenced_code",
    "tables",
    "admonition",
    "codehilite",
]
SPECIAL_PATHS = {
    "benchmarks/README.md": Path("benchmarks"),
    "src/tnfr/physics/README.md": Path("src") / "tnfr" / "physics",
}


class LinkRewriter(Treeprocessor):
    def __init__(
        self,
        md: markdown.Markdown,  # type: ignore[name-defined]
        source_path: str,
        site_paths: Dict[str, str],
    ) -> None:
        super().__init__(md)
        self.source_path = PurePosixPath(source_path)
        self.base = self.source_path.parent
        self.site_paths = site_paths

    def run(self, root):  # type: ignore[override]
        for element in root.iter("a"):
            href = element.get("href")
            new_href = self._rewrite_href(href)
            if new_href:
                element.set("href", new_href)
        return root

    def _rewrite_href(self, href: str | None) -> str | None:
        if not href:
            return None
        parsed = urlsplit(href)
        if parsed.scheme or (not parsed.path and parsed.fragment):
            return None
        path = parsed.path
        if not path or path.startswith("#"):
            return None
        anchor = f"#{parsed.fragment}" if parsed.fragment else ""
        repo_target = None
        for candidate in self._candidate_keys(path):
            repo_target = candidate
            if candidate in self.site_paths:
                return f"{self.site_paths[candidate]}{anchor}"
        if repo_target is None:
            repo_target = path.lstrip("./")
        return f"{REPO_URL}/blob/main/{repo_target}{anchor}"

    def _candidate_keys(self, relative: str) -> List[str]:
        joined = self.base.joinpath(PurePosixPath(relative)).as_posix()
        normalized = joined.lstrip("./")
        raw = PurePosixPath(relative).as_posix().lstrip("./")
        candidates = []
        if normalized:
            candidates.append(normalized)
        if raw and raw not in candidates:
            candidates.append(raw)
        return candidates


class LinkRewriteExtension(Extension):
    def __init__(self, source_path: str, site_paths: Dict[str, str]) -> None:
        super().__init__()
        self.source_path = source_path
        self.site_paths = site_paths

    def extendMarkdown(
        self,
        md: markdown.Markdown,  # type: ignore[name-defined]
    ) -> None:
        md.treeprocessors.register(
            LinkRewriter(md, self.source_path, self.site_paths),
            "tnfr_link_rewriter",
            15,
        )


def run_git(*args: str) -> str:
    """Run a git command and return stdout."""
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return result.stdout


def gather_sources(limit: Iterable[str] | None = None) -> List[str]:
    """Collect markdown files from origin/main that we need to publish."""
    listing = run_git(
        "ls-tree",
        "-r",
        "origin/main",
        "--name-only",
    ).splitlines()
    allowed: List[str] = []
    limit_set = set(limit) if limit else None
    for rel in listing:
        rel_lower = rel.lower()
        if not rel_lower.endswith(".md"):
            continue
        if limit_set and rel not in limit_set and rel_lower not in limit_set:
            continue
        parts = rel.split("/")
        if len(parts) == 1:
            allowed.append(rel)
            continue
        top = parts[0]
        if top in {"docs", "theory"}:
            allowed.append(rel)
            continue
        if rel in SPECIAL_PATHS:
            allowed.append(rel)
    for special in SPECIAL_PATHS:
        if special not in allowed:
            allowed.append(special)
    return sorted(set(allowed))


def slugify_segment(segment: str) -> str:
    return (
        segment.replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace("__", "_")
    )


def output_dir_for(path: str, used: Dict[Path, str]) -> Path:
    source_path = Path(path)
    if path in SPECIAL_PATHS:
        target = SPECIAL_PATHS[path]
        return ensure_unique(target, used, source_path)

    parents = source_path.parts[:-1]
    stem = slugify_segment(source_path.stem)

    if not parents:
        candidate = Path(stem)
    elif parents[0] in {"docs", "theory"}:
        extras = [slugify_segment(part) for part in parents[1:]]
        name = "_".join([p for p in extras if p] + [stem]) if extras else stem
        candidate = Path(name)
    else:
        cleaned = [slugify_segment(part) for part in parents]
        candidate = Path(*cleaned)
        if stem.lower() != "readme":
            candidate = candidate / stem

    return ensure_unique(candidate, used, source_path)


def ensure_unique(
    candidate: Path,
    used: Dict[Path, str],
    source: Path,
) -> Path:
    unique = candidate
    idx = 2
    hint = slugify_segment("_".join(source.parts[:-1])) or "doc"
    while unique in used:
        parent = unique.parent if str(unique.parent) != "." else Path("")
        suffix = f"{hint}_{idx}" if hint else str(idx)
        new_name = f"{unique.name}_{suffix}" if unique.name else suffix
        unique = (parent / new_name) if parent != Path("") else Path(new_name)
        idx += 1
    return unique


def fetch_markdown(path: str) -> str:
    return run_git("show", f"origin/main:{path}")


def fetch_last_updated(path: str) -> str:
    try:
        out = run_git(
            "log",
            "-1",
            "--format=%cs",
            "origin/main",
            "--",
            path,
        ).strip()
        return out or "unknown"
    except subprocess.CalledProcessError:
        return "unknown"


def extract_title(markdown_text: str, default: str) -> str:
    for line in markdown_text.splitlines():
        if line.startswith("#"):
            return line.lstrip("# ").strip()
    return default


def render_html(title: str, body: str, source_path: str, updated: str) -> str:
    source_url = f"{REPO_URL}/blob/main/{source_path}"
    escaped_title = html.escape(title)
    metadata = (
        f"Version {VERSION} · "
        f"DOI <a href=\"https://doi.org/{DOI}\" "
        f"target=\"_blank\" rel=\"noopener\">{DOI}</a> · "
        f"Updated {updated}"
    )
    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{escaped_title} — TNFR Python Engine</title>
  <meta name=\"description\" content=\"TNFR documentation: {escaped_title}\">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link
        rel="stylesheet"
        href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap"
    >
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0e0f11;
      --panel: rgba(24, 25, 27, 0.9);
      --border: rgba(90, 92, 95, 0.4);
      --text: #e4e6eb;
      --muted: #9ea4b3;
      --accent: #6aa0ff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      line-height: 1.6;
      background: radial-gradient(circle at 20% 20%, #1a1c20, var(--bg));
      color: var(--text);
      min-height: 100vh;
      padding: 2rem;
    }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
        .page {{
            max-width: 960px;
            margin: 0 auto;
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 2.5rem 3rem 3rem;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
        }}
        header {{
            border-bottom: 1px solid var(--border);
            padding-bottom: 1.5rem;
            margin-bottom: 2rem;
        }}
    .home-link {{ font-weight: 600; color: var(--text); }}
        .doc-title {{
            display: block;
            font-size: 2.2rem;
            font-weight: 600;
            margin-top: 0.4rem;
        }}
        .meta {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.8rem;
            margin-top: 1rem;
            font-size: 0.95rem;
            color: var(--muted);
        }}
    main {{ font-size: 1rem; }}
    main h1 {{ font-size: 2rem; margin-top: 2.5rem; }}
        main h2 {{
            font-size: 1.5rem;
            margin-top: 2rem;
            border-bottom: 1px solid var(--border);
            padding-bottom: 0.3rem;
        }}
    main h3 {{ font-size: 1.2rem; margin-top: 1.5rem; }}
    main p {{ margin: 1rem 0; }}
        pre {{
            background: #0a0b0d;
            padding: 1rem;
            border-radius: 10px;
            overflow-x: auto;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }}
    code {{ font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; }}
    table {{ width: 100%; border-collapse: collapse; margin: 1.5rem 0; }}
        th, td {{
            border: 1px solid var(--border);
            padding: 0.6rem 0.8rem;
            text-align: left;
        }}
        blockquote {{
            border-left: 4px solid var(--accent);
            margin: 1.5rem 0;
            padding: 0.2rem 1rem;
            color: var(--muted);
            background: rgba(255, 255, 255, 0.02);
        }}
        footer {{
            border-top: 1px solid var(--border);
            margin-top: 2.5rem;
            padding-top: 1.5rem;
            font-size: 0.9rem;
            color: var(--muted);
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            justify-content: space-between;
        }}
  </style>
</head>
<body>
  <div class=\"page\">
    <header>
      <a class=\"home-link\" href=\"{SITE_BASE}\">TNFR Python Engine</a>
      <span class=\"doc-title\">{escaped_title}</span>
      <div class=\"meta\">
        <span>{metadata}</span>
                <span>
                    Source
                    <a
                        href=\"{source_url}\"
                        target=\"_blank\"
                        rel=\"noopener\"
                    >
                        {html.escape(source_path)}
                    </a>
                </span>
      </div>
    </header>
    <main>
      {body}
    </main>
    <footer>
            <span>
                &copy; {datetime.now(UTC).year}
                TNFR Python Engine — Licensed under MIT.
            </span>
            <span>
                Feedback?
                <a
                    href=\"{REPO_URL}\"
                    target=\"_blank\"
                    rel=\"noopener\"
                >
                    Open an issue
                </a>.
            </span>
    </footer>
  </div>
</body>
</html>
"""


def convert_markdown(
    source_path: str,
    target_dir: Path,
    site_paths: Dict[str, str],
) -> None:
    markdown_text = fetch_markdown(source_path)
    title = extract_title(markdown_text, source_path)
    extensions = MARKDOWN_EXTENSIONS + [
        LinkRewriteExtension(source_path, site_paths),
    ]
    body_html = markdown.markdown(
        markdown_text,
        extensions=extensions,
        output_format="html5",
    )
    updated = fetch_last_updated(source_path)
    html_text = render_html(title, body_html, source_path, updated)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "index.html").write_text(html_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render Markdown documents to HTML",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Specific markdown paths to render",
    )
    args = parser.parse_args()
    used: Dict[Path, str] = {}
    sources = gather_sources(args.paths if args.paths else None)
    assignments: Dict[str, Path] = {}
    for source in sources:
        target = output_dir_for(source, used)
        assignments[source] = target
        used[target] = source
    site_paths = {
        src: f"{SITE_BASE}{path.as_posix().rstrip('/')}/"
        for src, path in assignments.items()
    }

    print(f"Rendering {len(assignments)} markdown files...")
    for source, target in assignments.items():
        convert_markdown(source, REPO_ROOT / target, site_paths)
        print(f"  ✔ {source} -> {target}")
    print("Done.")


if __name__ == "__main__":
    main()
