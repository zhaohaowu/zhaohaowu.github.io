#!/usr/bin/env python3
from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Iterable

try:
    from pygments import highlight
    from pygments.formatters import HtmlFormatter
    from pygments.lexers import TextLexer, get_lexer_by_name
    from pygments.util import ClassNotFound
except ImportError:  # pragma: no cover
    highlight = None
    HtmlFormatter = None
    TextLexer = None
    get_lexer_by_name = None
    ClassNotFound = Exception


ROOT = Path(__file__).resolve().parent
SECTIONS = ("基础", "核心", "提高")
LIST_RE = re.compile(r"^(\s*)([*+-]|\d+\.)\s+(.*)$")
TABLE_SEP_RE = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")


def normalize_lines(text: str) -> list[str]:
    return [line.expandtabs(4).replace("\u200b", "") for line in text.splitlines()]


def trim_blank_edges(lines: list[str]) -> list[str]:
    start = 0
    end = len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


def strip_indent(line: str, width: int) -> str:
    removed = 0
    idx = 0
    while idx < len(line) and removed < width and line[idx] == " ":
        idx += 1
        removed += 1
    return line[idx:]


def split_table_row(line: str) -> list[str]:
    text = line.strip()
    if text.startswith("|"):
        text = text[1:]
    if text.endswith("|"):
        text = text[:-1]

    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for ch in text:
        if escaped:
            current.append(ch)
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == "|":
            cells.append("".join(current).strip())
            current = []
            continue
        current.append(ch)
    cells.append("".join(current).strip())
    return cells


def heading_slug(text: str, seen: dict[str, int]) -> str:
    clean = re.sub(r"`([^`]*)`", r"\1", text)
    clean = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", clean)
    clean = re.sub(r"\[[^\]]+\]\([^)]+\)", "", clean)
    clean = re.sub(r"[*_=~#]+", "", clean)
    clean = re.sub(r"\s+", "-", clean.strip())
    clean = re.sub(r"[^\w\u4e00-\u9fff-]", "", clean)
    slug = clean or "section"
    seen[slug] = seen.get(slug, 0) + 1
    if seen[slug] > 1:
        slug = f"{slug}-{seen[slug]}"
    return slug


def format_code(code: str, language: str) -> str:
    if highlight and HtmlFormatter and get_lexer_by_name and TextLexer:
        lexer_name = (language or "text").lower()
        lexer_name = {
            "c++": "cpp",
            "cpp": "cpp",
            "c": "c",
            "plaintext": "text",
            "plain": "text",
            "text": "text",
            "shell": "bash",
            "sh": "bash",
        }.get(lexer_name, lexer_name)
        try:
            lexer = get_lexer_by_name(lexer_name)
        except ClassNotFound:
            lexer = TextLexer()
        formatter = HtmlFormatter(cssclass="codehilite", nowrap=False, style="monokai")
        return highlight(code, lexer, formatter)
    class_name = f' class="language-{html.escape(language)}"' if language else ""
    return f"<pre><code{class_name}>{html.escape(code)}</code></pre>"


class Renderer:
    def __init__(self, section_names: Iterable[str]) -> None:
        self.section_names = tuple(section_names)
        self.slug_seen: dict[str, int] = {}
        self.headings: list[tuple[int, str, str]] = []

    def render_inline(self, text: str) -> str:
        placeholders: dict[str, str] = {}

        def stash(value: str) -> str:
            key = f"@@INLINE{len(placeholders)}@@"
            placeholders[key] = value
            return key

        def code_repl(match: re.Match[str]) -> str:
            return stash(f"<code>{html.escape(match.group(1))}</code>")

        text = re.sub(r"`([^`]+)`", code_repl, text)
        text = html.escape(text, quote=False)

        def image_repl(match: re.Match[str]) -> str:
            alt = match.group(1)
            src = match.group(2)
            return f'<img src="{src}" alt="{alt}" loading="lazy" />'

        def link_repl(match: re.Match[str]) -> str:
            label = match.group(1)
            href = match.group(2)
            return f'<a href="{href}">{label}</a>'

        text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", image_repl, text)
        text = re.sub(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)", link_repl, text)
        text = re.sub(r"==(.+?)==", r"<mark>\1</mark>", text)
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<em>\1</em>", text)

        for key, value in placeholders.items():
            text = text.replace(key, value)
        return text

    def is_table_start(self, lines: list[str], idx: int) -> bool:
        if idx + 1 >= len(lines):
            return False
        if not lines[idx].lstrip().startswith("|"):
            return False
        return bool(TABLE_SEP_RE.match(lines[idx + 1]))

    def is_heading(self, line: str) -> bool:
        return bool(HEADING_RE.match(line.strip()))

    def is_blockquote(self, line: str) -> bool:
        return bool(re.match(r"^\s*>\s?", line))

    def is_list_item(self, line: str) -> bool:
        return bool(LIST_RE.match(line))

    def starts_block(self, lines: list[str], idx: int) -> bool:
        line = lines[idx]
        if not line.strip():
            return False
        return (
            line.strip().startswith("```")
            or self.is_heading(line)
            or self.is_blockquote(line)
            or self.is_list_item(line)
            or self.is_table_start(lines, idx)
        )

    def parse_list(self, lines: list[str], start: int) -> tuple[str, int]:
        first = LIST_RE.match(lines[start])
        assert first
        base_indent = len(first.group(1))
        marker = first.group(2)
        list_tag = "ol" if marker.endswith(".") else "ul"
        attrs = ""
        if list_tag == "ol":
            number = int(marker[:-1])
            if number != 1:
                attrs = f' start="{number}"'

        parts = [f"<{list_tag}{attrs}>"]
        idx = start
        while idx < len(lines):
            current = lines[idx]
            if not current.strip():
                idx += 1
                continue

            match = LIST_RE.match(current)
            if not match:
                break

            indent = len(match.group(1))
            current_tag = "ol" if match.group(2).endswith(".") else "ul"
            if indent < base_indent or (indent == base_indent and current_tag != list_tag):
                break
            if indent > base_indent:
                break

            item_lines = [match.group(3)]
            idx += 1
            while idx < len(lines):
                upcoming = lines[idx]
                if not upcoming.strip():
                    item_lines.append("")
                    idx += 1
                    continue

                next_match = LIST_RE.match(upcoming)
                if next_match:
                    next_indent = len(next_match.group(1))
                    next_tag = "ol" if next_match.group(2).endswith(".") else "ul"
                    if next_indent < base_indent:
                        break
                    if next_indent == base_indent and next_tag == list_tag:
                        break
                    if next_indent == base_indent and next_tag != list_tag:
                        break

                item_lines.append(strip_indent(upcoming, base_indent + 2))
                idx += 1

            rendered = self.render_blocks(trim_blank_edges(item_lines))
            parts.append(f"<li>{rendered}</li>")

        parts.append(f"</{list_tag}>")
        return "".join(parts), idx

    def parse_table(self, lines: list[str], start: int) -> tuple[str, int]:
        header = split_table_row(lines[start])
        rows: list[list[str]] = []
        idx = start + 2
        while idx < len(lines) and lines[idx].lstrip().startswith("|"):
            rows.append(split_table_row(lines[idx]))
            idx += 1

        parts = [
            "<div class=\"table-wrap\"><table><thead><tr>",
            "".join(f"<th>{self.render_inline(cell)}</th>" for cell in header),
            "</tr></thead>",
        ]
        if rows:
            parts.append("<tbody>")
            for row in rows:
                parts.append("<tr>")
                row = row + [""] * (len(header) - len(row))
                for cell in row[: len(header)]:
                    parts.append(f"<td>{self.render_inline(cell)}</td>")
                parts.append("</tr>")
            parts.append("</tbody>")
        parts.append("</table></div>")
        return "".join(parts), idx

    def parse_blockquote(self, lines: list[str], start: int) -> tuple[str, int]:
        quote_lines: list[str] = []
        idx = start
        while idx < len(lines):
            line = lines[idx]
            if not line.strip():
                quote_lines.append("")
                idx += 1
                continue
            match = re.match(r"^\s*>\s?(.*)$", line)
            if not match:
                break
            quote_lines.append(match.group(1))
            idx += 1
        return f"<blockquote>{self.render_blocks(trim_blank_edges(quote_lines))}</blockquote>", idx

    def parse_code(self, lines: list[str], start: int) -> tuple[str, int]:
        opener = lines[start].strip()
        language = opener[3:].strip()
        code_lines: list[str] = []
        idx = start + 1
        while idx < len(lines) and not lines[idx].strip().startswith("```"):
            code_lines.append(lines[idx])
            idx += 1
        if idx < len(lines):
            idx += 1
        code = "\n".join(code_lines)
        return format_code(code, language), idx

    def parse_paragraph(self, lines: list[str], start: int) -> tuple[str, int]:
        paragraph: list[str] = []
        idx = start
        while idx < len(lines):
            line = lines[idx]
            if not line.strip():
                break
            if paragraph and self.starts_block(lines, idx):
                break
            paragraph.append(line.strip())
            idx += 1
        text = " ".join(part for part in paragraph if part)
        return f"<p>{self.render_inline(text)}</p>", idx

    def render_blocks(self, lines: list[str]) -> str:
        parts: list[str] = []
        idx = 0
        while idx < len(lines):
            line = lines[idx]
            if not line.strip():
                idx += 1
                continue

            if line.strip().startswith("```"):
                block, idx = self.parse_code(lines, idx)
                parts.append(block)
                continue

            heading_match = HEADING_RE.match(line.strip())
            if heading_match:
                level = len(heading_match.group(1))
                raw_text = heading_match.group(2).strip()
                slug = heading_slug(raw_text, self.slug_seen)
                self.headings.append((level, raw_text, slug))
                parts.append(
                    f'<h{level} id="{slug}">{self.render_inline(raw_text)}</h{level}>'
                )
                idx += 1
                continue

            if self.is_table_start(lines, idx):
                block, idx = self.parse_table(lines, idx)
                parts.append(block)
                continue

            if self.is_blockquote(line):
                block, idx = self.parse_blockquote(lines, idx)
                parts.append(block)
                continue

            if self.is_list_item(line):
                block, idx = self.parse_list(lines, idx)
                parts.append(block)
                continue

            block, idx = self.parse_paragraph(lines, idx)
            parts.append(block)
        return "\n".join(parts)

    def build_toc(self) -> str:
        entries = [item for item in self.headings if item[0] <= 3]
        if not entries:
            return ""
        items = []
        for level, text, slug in entries:
            items.append(
                f'<li class="toc-level-{level}"><a href="#{slug}">{self.render_inline(text)}</a></li>'
            )
        return "<nav class=\"toc\"><div class=\"toc-title\">目录</div><ol>" + "".join(items) + "</ol></nav>"

    def build_nav(self, current_section: str) -> str:
        links = []
        for section in self.section_names:
            class_name = "is-active" if section == current_section else ""
            links.append(
                f'<a class="{class_name}" href="../{section}/index.html">{section}</a>'
            )
        return "<nav class=\"site-nav\">" + "".join(links) + "</nav>"

    def wrap_page(self, title: str, current_section: str, body: str) -> str:
        nav = self.build_nav(current_section)
        toc = self.build_toc()
        code_css = (
            HtmlFormatter(cssclass="codehilite", style="monokai").get_style_defs(".codehilite")
            if HtmlFormatter
            else ""
        )
        return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)} | 黑马C++笔记</title>
  <style>
    :root {{
      --bg: #f5efe6;
      --surface: rgba(255, 252, 247, 0.88);
      --surface-strong: #fffdfa;
      --text: #1f2933;
      --muted: #52606d;
      --line: rgba(31, 41, 51, 0.12);
      --accent: #b5542f;
      --accent-soft: rgba(181, 84, 47, 0.12);
      --shadow: 0 18px 50px rgba(63, 45, 32, 0.12);
      --code-bg: #f3efe7;
      --mark: #ffe08a;
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      font-family: "Noto Serif SC", "Source Han Serif SC", "Songti SC", serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(196, 109, 58, 0.16), transparent 22rem),
        linear-gradient(180deg, #f6f0e5 0%, #efe6d8 100%);
      line-height: 1.72;
    }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    img {{
      max-width: 100%;
      height: auto;
      display: block;
      margin: 1.5rem auto;
      border-radius: 14px;
      box-shadow: 0 12px 34px rgba(31, 41, 51, 0.12);
      background: #fff;
    }}
    code {{
      font-family: "JetBrains Mono", "Fira Code", monospace;
      background: var(--code-bg);
      border-radius: 6px;
      padding: 0.12rem 0.35rem;
      font-size: 0.92em;
    }}
    pre {{
      overflow-x: auto;
      padding: 1rem 1.15rem;
      border-radius: 16px;
      background: #272822;
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.04);
      color: #f8f8f2;
    }}
    pre code {{
      background: transparent;
      padding: 0;
      border-radius: 0;
      color: inherit;
    }}
    .codehilite {{
      background: #272822;
      color: #f8f8f2;
      border-radius: 16px;
    }}
    .codehilite pre {{
      margin: 0;
      background: transparent;
      color: inherit;
    }}
    blockquote {{
      margin: 1.4rem 0;
      padding: 0.9rem 1rem 0.9rem 1.1rem;
      border-left: 4px solid var(--accent);
      background: rgba(255,255,255,0.45);
      border-radius: 0 12px 12px 0;
      color: var(--muted);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 40rem;
      background: var(--surface-strong);
    }}
    .table-wrap {{
      overflow-x: auto;
      border-radius: 16px;
      box-shadow: inset 0 0 0 1px var(--line);
      background: var(--surface-strong);
      margin: 1.4rem 0;
    }}
    th, td {{
      padding: 0.7rem 0.85rem;
      border: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: rgba(181, 84, 47, 0.08);
      font-weight: 700;
    }}
    mark {{
      background: var(--mark);
      padding: 0.05rem 0.22rem;
      border-radius: 4px;
    }}
    .page {{
      width: min(1240px, calc(100% - 2rem));
      margin: 0 auto;
      padding: 2rem 0 4rem;
    }}
    .hero {{
      margin-bottom: 1.5rem;
      padding: 1.4rem 1.6rem;
      border-radius: 24px;
      background: linear-gradient(135deg, rgba(255,255,255,0.72), rgba(255,248,239,0.92));
      box-shadow: var(--shadow);
      border: 1px solid rgba(255,255,255,0.5);
      backdrop-filter: blur(10px);
    }}
    .eyebrow {{
      color: var(--accent);
      font-size: 0.92rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .hero-title {{
      margin: 0.35rem 0 0;
      font-size: clamp(2rem, 4vw, 3rem);
      line-height: 1.15;
      font-weight: 700;
    }}
    .site-nav {{
      display: flex;
      gap: 0.65rem;
      flex-wrap: wrap;
      margin-top: 1.2rem;
    }}
    .site-nav a {{
      padding: 0.58rem 0.9rem;
      border-radius: 999px;
      background: rgba(255,255,255,0.66);
      border: 1px solid rgba(181, 84, 47, 0.18);
      color: var(--text);
      transition: transform 0.18s ease, background 0.18s ease;
    }}
    .site-nav a:hover {{
      text-decoration: none;
      transform: translateY(-1px);
      background: rgba(255,255,255,0.94);
    }}
    .site-nav a.is-active {{
      background: var(--accent);
      color: #fff8f2;
      border-color: transparent;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 290px;
      gap: 1.5rem;
      align-items: start;
    }}
    .content {{
      min-width: 0;
      padding: 1.6rem 1.8rem;
      border-radius: 24px;
      background: var(--surface);
      box-shadow: var(--shadow);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255,255,255,0.52);
    }}
    .content > :first-child {{ margin-top: 0; }}
    .content h1, .content h2, .content h3, .content h4 {{
      line-height: 1.25;
      scroll-margin-top: 1rem;
    }}
    .content h2 {{
      margin-top: 2.4rem;
      padding-top: 0.2rem;
      border-top: 1px solid rgba(31, 41, 51, 0.08);
    }}
    .content p, .content ul, .content ol, .content pre, .content table, .content blockquote {{
      margin-top: 1rem;
      margin-bottom: 1rem;
    }}
    .content ul, .content ol {{
      padding-left: 1.5rem;
    }}
    .toc {{
      position: sticky;
      top: 1rem;
      padding: 1rem 1rem 1.1rem;
      border-radius: 20px;
      background: rgba(255,255,255,0.68);
      box-shadow: var(--shadow);
      border: 1px solid rgba(255,255,255,0.55);
      backdrop-filter: blur(10px);
    }}
    .toc-title {{
      font-size: 0.92rem;
      color: var(--muted);
      margin-bottom: 0.8rem;
    }}
    .toc ol {{
      list-style: none;
      margin: 0;
      padding: 0;
    }}
    .toc li + li {{
      margin-top: 0.45rem;
    }}
    .toc-level-2 {{
      padding-left: 0.7rem;
    }}
    .toc-level-3 {{
      padding-left: 1.4rem;
      font-size: 0.96rem;
    }}
    .toc a {{
      color: var(--text);
    }}
    .toc a:hover {{
      color: var(--accent);
    }}
    @media (max-width: 960px) {{
      .layout {{
        grid-template-columns: 1fr;
      }}
      .toc {{
        position: static;
        order: -1;
      }}
    }}
    @media (max-width: 640px) {{
      .page {{
        width: min(100% - 1rem, 100%);
        padding-top: 0.75rem;
      }}
      .hero, .content {{
        padding-left: 1rem;
        padding-right: 1rem;
        border-radius: 20px;
      }}
      table {{
        min-width: 32rem;
      }}
    }}
    {code_css}
  </style>
</head>
<body>
  <div class="page">
    <header class="hero">
      <div class="eyebrow">Black Horse C++ Notes</div>
      <div class="hero-title">{html.escape(title)}</div>
      {nav}
    </header>
    <div class="layout">
      <main class="content">
        {body}
      </main>
      {toc}
    </div>
  </div>
</body>
</html>
"""


def render_section(section: str) -> None:
    source = ROOT / section / "readme.md"
    target = ROOT / section / "index.html"
    renderer = Renderer(SECTIONS)
    lines = normalize_lines(source.read_text(encoding="utf-8"))
    body = renderer.render_blocks(lines)
    title = renderer.headings[0][1] if renderer.headings else section
    page = renderer.wrap_page(title, section, body)
    target.write_text(page, encoding="utf-8")
    print(f"generated {target}")


def main() -> None:
    for section in SECTIONS:
        render_section(section)


if __name__ == "__main__":
    main()
