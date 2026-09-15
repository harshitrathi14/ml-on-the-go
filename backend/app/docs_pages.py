"""
Documentation served by the portal itself: the usage guide rendered from
docs/NU_SCORE_USAGE_GUIDE.md as a branded HTML page, and downloads for the
guide (Markdown) and the product deck (PPTX).
"""

from __future__ import annotations

from pathlib import Path

import markdown
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"
GUIDE_MD = DOCS_DIR / "NU_SCORE_USAGE_GUIDE.md"
DOWNLOADS = {
    "Nu_Score_On_the_Go.pptx": ("application/vnd.openxmlformats-officedocument.presentationml.presentation",
                                DOCS_DIR / "Nu_Score_On_the_Go.pptx"),
    "NU_SCORE_USAGE_GUIDE.md": ("text/markdown", GUIDE_MD),
}

router = APIRouter()

_PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Usage guide · Nu Score On the Go</title>
<link rel="stylesheet" href="brand/fonts.css">
<style>
  :root {{ --cyan:#4fc6e0; --cyan-deep:#1b9ab8; --navy:#0d1c31; --ink:#0d1c31; --ink-2:#44546b; --ink-3:#7a879a;
           --line:#e3e8ef; --ground:#f5f7fa; --wash:#e9f8fc; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:"Host Grotesk","Segoe UI",system-ui,sans-serif; color:var(--ink); background:var(--ground); }}
  .topbar {{ background:var(--navy); color:#fff; padding:12px 5vw; display:flex; align-items:center; justify-content:space-between; gap:16px; }}
  .topbar img {{ height:28px; }}
  .topbar a {{ color:var(--cyan); text-decoration:none; font-size:.9rem; margin-left:18px; }}
  .hero {{ background:linear-gradient(180deg,#0a1526,var(--navy)); color:#fff; padding:36px 5vw 44px; }}
  .hero .eyebrow {{ color:var(--cyan); font-size:.78rem; letter-spacing:.14em; text-transform:uppercase; font-family:ui-monospace,Menlo,monospace; }}
  .hero h1 {{ margin:8px 0 6px; font-size:2rem; }}
  .hero p {{ color:#c7d2e0; margin:0; max-width:760px; }}
  .downloads {{ margin-top:18px; display:flex; gap:10px; flex-wrap:wrap; }}
  .downloads a {{ background:var(--cyan); color:var(--navy); font-weight:600; padding:9px 16px; border-radius:10px; text-decoration:none; font-size:.9rem; }}
  .downloads a.ghost {{ background:transparent; color:#fff; border:1px solid rgba(79,198,224,.5); }}
  .wrap {{ display:grid; grid-template-columns:260px 1fr; gap:28px; max-width:1240px; margin:-24px auto 60px; padding:0 5vw; }}
  @media (max-width:900px) {{ .wrap {{ grid-template-columns:1fr; }} nav.toc {{ position:static; }} }}
  nav.toc {{ position:sticky; top:16px; align-self:start; background:#fff; border:1px solid var(--line); border-radius:14px; padding:14px 16px; font-size:.86rem; }}
  nav.toc ul {{ list-style:none; padding:0; margin:0; }}
  nav.toc ul ul {{ padding-left:12px; display:none; }}
  nav.toc li {{ margin:6px 0; }}
  nav.toc a {{ color:var(--ink-2); text-decoration:none; }}
  nav.toc a:hover {{ color:var(--cyan-deep); }}
  article {{ background:#fff; border:1px solid var(--line); border-radius:14px; padding:28px 34px; line-height:1.6; min-width:0; }}
  article h1 {{ font-size:1.6rem; margin-top:0; }}
  article h2 {{ font-size:1.25rem; margin-top:2em; padding-top:.6em; border-top:1px solid var(--line); }}
  article h3 {{ font-size:1.02rem; margin-top:1.6em; }}
  article table {{ border-collapse:collapse; width:100%; font-size:.9rem; margin:12px 0; display:block; overflow-x:auto; }}
  article th, article td {{ border-bottom:1px solid var(--line); padding:8px 10px; text-align:left; vertical-align:top; }}
  article th {{ background:var(--ground); font-size:.78rem; text-transform:uppercase; letter-spacing:.05em; color:var(--ink-3); }}
  article code {{ background:var(--ground); border:1px solid var(--line); border-radius:5px; padding:1px 6px; font-size:.86em; }}
  article pre {{ background:var(--navy); color:#dff5fa; padding:14px 16px; border-radius:10px; overflow-x:auto; }}
  article pre code {{ background:none; border:none; color:inherit; padding:0; }}
  article blockquote {{ border-left:3px solid var(--cyan); background:var(--wash); margin:12px 0; padding:10px 14px; border-radius:0 10px 10px 0; }}
  article a {{ color:var(--cyan-deep); }}
  article hr {{ border:none; border-top:1px solid var(--line); margin:28px 0; }}
  footer {{ text-align:center; color:var(--ink-3); font-size:.8rem; padding:0 5vw 32px; }}
</style></head>
<body>
<div class="topbar"><div><img src="brand/northern-arc-light.png" alt="Northern Arc"></div>
  <div><a href="./">← Back to the portal</a><a href="downloads/Nu_Score_On_the_Go.pptx">Product deck (PPTX)</a></div></div>
<header class="hero"><div class="eyebrow">Documentation</div><h1>Nu Score On the Go — Usage Guide</h1>
  <p>How to prepare your data, train and validate models, read the Nu Score, and score new applications.</p>
  <div class="downloads"><a href="downloads/Nu_Score_On_the_Go.pptx">Download the product deck</a>
  <a class="ghost" href="downloads/NU_SCORE_USAGE_GUIDE.md">Download this guide (Markdown)</a></div></header>
<div class="wrap"><nav class="toc">{toc}</nav><article>{body}</article></div>
<footer>Nu Score On the Go · Northern Arc / AltiFi internal analytics</footer>
</body></html>"""


def render_guide() -> str:
    if not GUIDE_MD.exists():
        raise HTTPException(status_code=404, detail="Guide not found.")
    md = markdown.Markdown(extensions=["tables", "toc", "fenced_code", "sane_lists"],
                           extension_configs={"toc": {"toc_depth": "2-3"}})
    body = md.convert(GUIDE_MD.read_text(encoding="utf-8"))
    return _PAGE.format(toc=md.toc, body=body)


@router.get("/guide", response_class=HTMLResponse, include_in_schema=False)
async def guide() -> HTMLResponse:
    return HTMLResponse(render_guide())


@router.get("/downloads/{name}", include_in_schema=False)
async def download(name: str) -> FileResponse:
    if name not in DOWNLOADS or not DOWNLOADS[name][1].exists():
        raise HTTPException(status_code=404, detail="File not found.")
    media_type, path = DOWNLOADS[name]
    return FileResponse(path, media_type=media_type, filename=name)
