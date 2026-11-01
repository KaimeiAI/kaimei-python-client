"""Kaimei-themed HTML cards for notebooks.

``render(record)`` turns a record the API returned (an explanation, insight, model, prediction,
analysis, a page of them, ...) into one self-contained HTML card: the Kaimei mark, the dashboard's
colours and type, and the dashboard's numbers formatted by the same rules (its ``fmtSigned``,
``pct``, ``outputParts``, ``compact``, ... are ported below, name for name).

Everything is HTML and CSS. Hover highlights, tooltips (``title`` attributes and SVG ``<title>``)
and collapsible ``<details>`` sections work wherever a notebook renders HTML, with no script to
run, and the card follows the host's dark theme (JupyterLab, VS Code, Colab). Image explanations
add thumbnails when NumPy and Pillow are installed; without them the card lists the regions.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
from datetime import datetime, timedelta
from html import escape
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ._inline import inline_styles

__all__ = ["render", "CSS"]

DASHBOARD_URL = "https://dashboard.kaimei.ai"

_MINUS = "−"
_MONO = '"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace'

#: The dashboard's mark (its favicon), inline so the card needs no network.
MARK = (
    '<svg width="18" height="18" viewBox="0 0 1024 1024" fill="none" aria-hidden="true">'
    '<rect width="1024" height="1024" rx="230" fill="#0f0f10"/>'
    '<rect x="248" y="232" width="122" height="558" rx="61" fill="#ffffff"/>'
    '<path d="M418 512 L730 232 A387 387 0 0 1 730 792 Z" fill="url(#km-k)" stroke="url(#km-k)" '
    'stroke-width="28" stroke-linejoin="round"/>'
    '<defs><linearGradient id="km-k" x1="640" y1="232" x2="640" y2="792" gradientUnits="userSpaceOnUse">'
    '<stop stop-color="#45c7ff"/><stop offset="0.5" stop-color="#8a8cff"/><stop offset="1" stop-color="#ee6fdc"/>'
    "</linearGradient></defs></svg>"
)

#: The dashboard's tokens (``index.css``), scoped to the card. The dark block follows the host's
#: theme attribute rather than the OS, because notebook front ends set their own.
CSS = (
    ".km{--bg:#ffffff;--bg2:#f7f8fc;--line:#e4e7f0;--line2:#d3d8e6;--text:#12141c;--muted:#53566a;--faint:#7a8096;"
    "--indigo:#4566e8;--blue:#6b8cff;--ok:#0f8f5b;--warn:#9a6700;--serious:#c2410c;--bad:#d14b7a;"
    "--ok-soft:rgba(15,143,91,.12);--warn-soft:rgba(154,103,0,.12);--serious-soft:rgba(194,65,12,.12);"
    "--bad-soft:rgba(209,75,122,.12);--info-soft:rgba(107,140,255,.14);--wash:rgba(18,20,28,.04);"
    "--track:rgba(18,20,28,.08);--midline:rgba(18,20,28,.18);--series-1:#4566e8;--series-2:#0891b2;"
    "--hover:rgba(107,140,255,.06);--shadow:0 1px 2px rgba(18,20,28,.05);"
    'font-family:Inter,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;font-size:13px;line-height:1.45;'
    "color:var(--text);background:var(--bg);border:1px solid var(--line);border-radius:14px;padding:14px 18px 14px;"
    "max-width:780px;box-shadow:var(--shadow);margin:4px 0;text-align:left;box-sizing:border-box}"
    ".km.wide{max-width:1080px}"
    ".km .km-raw{display:block!important}"
    ".km *{box-sizing:border-box}"
    'body[data-jp-theme-light="false"] .km,body.vscode-dark .km,body.vscode-high-contrast .km,body.theme-dark .km,'
    'body[theme="dark"] .km,.km.dark{--bg:#141826;--bg2:#10131d;--line:rgba(255,255,255,.08);'
    "--line2:rgba(255,255,255,.14);--text:#f3f5fb;--muted:#b7bccd;--faint:#7f879c;--indigo:#8aa4ff;--ok:#3dd68c;"
    "--warn:#ffc857;--serious:#ff9b6a;--bad:#ff6b8a;--ok-soft:rgba(61,214,140,.14);--warn-soft:rgba(255,200,87,.14);"
    "--serious-soft:rgba(255,155,106,.14);--bad-soft:rgba(255,107,138,.14);--info-soft:rgba(107,140,255,.16);"
    "--wash:rgba(255,255,255,.03);--track:rgba(255,255,255,.06);--midline:rgba(255,255,255,.22);--series-1:#6b8cff;"
    "--series-2:#3ec8ff;--hover:rgba(255,255,255,.035);--shadow:0 1px 2px rgba(0,0,0,.35)}"
    f".km .km-mono{{font-family:{_MONO}}}"
    ".km-head{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;margin:0 0 12px;"
    "padding-bottom:10px;border-bottom:1px solid var(--line)}"
    ".km-brand{display:inline-flex;align-items:center;gap:7px;color:inherit;text-decoration:none;flex-wrap:wrap}"
    ".km-brand svg{width:18px;height:18px;flex:none}"
    ".km-word{font-weight:600;font-size:13.5px;letter-spacing:-.01em;color:var(--text)}"
    ".km-word i{font-style:normal;color:#6b8cff}"
    "@supports(-webkit-background-clip:text){.km-word i{background:linear-gradient(90deg,#3ec8ff 0%,#6b8cff 48%,#a078ff 100%);"
    "-webkit-background-clip:text;background-clip:text;color:transparent}}"
    ".km-eyebrow{font-size:10.5px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--faint);"
    "margin-left:10px;padding-left:10px;border-left:1px solid var(--line2)}"
    ".km-id{font-size:11px;color:var(--muted);margin-left:8px}"
    ".km-when{font-size:11.5px;color:var(--faint)}"
    ".km-result{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;margin:2px 0 8px}"
    ".km-result .km-value{font-size:30px;font-weight:650;letter-spacing:-.03em;font-variant-numeric:tabular-nums;line-height:1.1}"
    ".km-result .km-title{font-size:20px;font-weight:650;letter-spacing:-.02em;line-height:1.2}"
    ".km-result .km-sub{color:var(--muted);font-size:14px}"
    ".km-badges{display:flex;gap:6px;flex-wrap:wrap;margin:0 0 12px}"
    ".km-badge{display:inline-flex;align-items:center;gap:5px;height:22px;padding:0 8px;border-radius:999px;font-size:11.5px;"
    "font-weight:500;border:1px solid var(--line);background:var(--wash);color:var(--muted);white-space:nowrap}"
    f".km-badge.mono{{font-family:{_MONO};font-size:11px}}"
    ".km-badge.ok{color:var(--ok);background:var(--ok-soft);border-color:transparent}"
    ".km-badge.warn{color:var(--warn);background:var(--warn-soft);border-color:transparent}"
    ".km-badge.serious{color:var(--serious);background:var(--serious-soft);border-color:transparent}"
    ".km-badge.bad{color:var(--bad);background:var(--bad-soft);border-color:transparent}"
    ".km-badge.info{color:var(--indigo);background:var(--info-soft);border-color:transparent}"
    ".km-badge .km-dot{width:6px;height:6px;border-radius:50%;background:currentColor}"
    ".km-lead{color:var(--muted);font-size:12.5px;margin:0 0 12px;max-width:660px}"
    ".km-h{font-size:13px;font-weight:600;margin:12px 0 8px;color:var(--text)}"
    ".km-legend{display:flex;gap:16px;font-size:12px;color:var(--muted);flex-wrap:wrap;margin:0 0 10px}"
    ".km-legend span{display:inline-flex;align-items:center;gap:6px}"
    ".km-legend i{width:12px;height:10px;border-radius:2px;display:inline-block}"
    ".km-bars{display:flex;flex-direction:column;gap:5px}"
    ".km-row{display:grid;grid-template-columns:minmax(120px,200px) 1fr 68px;align-items:center;gap:12px;font-size:12.5px;"
    "padding:2px 6px;margin:0 -6px;border-radius:6px}"
    ".km-row.share{grid-template-columns:minmax(110px,170px) 1fr 48px}"
    ".km-row:hover{background:var(--hover)}"
    f".km-row .km-name{{color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:{_MONO};font-size:11.5px}}"
    ".km-row .km-name b{color:var(--text);font-weight:500}"
    f".km-row .km-val{{text-align:right;font-variant-numeric:tabular-nums;font-family:{_MONO};font-size:11.5px;color:var(--text)}}"
    ".km-track{position:relative;height:14px}"
    ".km-zero{position:absolute;top:0;bottom:0;width:1px;background:var(--midline);left:50%}"
    ".km-fill{position:absolute;top:2px;height:10px;border-radius:0 3px 3px 0}"
    ".km-fill.pos{background:linear-gradient(90deg,#3ec8ff,#6b8cff)}"
    ".km-fill.neg{border-radius:3px 0 0 3px;background:linear-gradient(90deg,#ff6bb5,#c86bff)}"
    ".km-fill.share{background:linear-gradient(90deg,#6b8cff,#a078ff)}"
    ".km-fill.for{background:linear-gradient(90deg,#ffb04d,#f97316)}"
    ".km-fill.against{background:linear-gradient(90deg,#8ca6ff,#5b5bf5)}"
    ".km-fill.stable{background:var(--ok)}.km-fill.moderate{background:var(--warn)}.km-fill.significant{background:var(--bad)}"
    ".km-meter{height:6px;border-radius:3px;background:var(--track);overflow:hidden}"
    ".km-meter i{display:block;height:100%;border-radius:3px;background:linear-gradient(90deg,#3ec8ff,#6b8cff)}"
    ".km-tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(118px,1fr));gap:8px;margin:0 0 12px}"
    ".km-tile{background:var(--bg2);border:1px solid var(--line);border-radius:9px;padding:8px 10px;min-width:0}"
    ".km-tile .km-k{font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;color:var(--faint);font-weight:600}"
    ".km-tile .km-v{font-size:16px;font-weight:600;letter-spacing:-.02em;font-variant-numeric:tabular-nums;margin-top:2px;"
    "white-space:nowrap;overflow:hidden;text-overflow:ellipsis}"
    ".km-tile .km-f{font-size:11px;color:var(--muted)}"
    ".km .km-wrap{overflow-x:auto}"
    ".km table.km-table{width:100%;border-collapse:collapse;border-spacing:0;table-layout:auto;margin:0;font-size:12px;"
    "color:var(--text);background:transparent}"
    ".km .km-table th,.km .km-table td{text-align:left;padding:5px 8px;border:0;border-bottom:1px solid var(--line);"
    "vertical-align:middle;background:transparent;line-height:1.35;white-space:nowrap;font-weight:400}"
    ".km .km-table td.clip{max-width:280px;overflow:hidden;text-overflow:ellipsis}"
    ".km .km-table td.wrap{white-space:normal;line-height:1.9}"
    ".km .km-table th{font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;color:var(--faint);font-weight:600;"
    "border-bottom:1px solid var(--line2)}"
    ".km .km-table td.num,.km .km-table th.num{text-align:right;font-variant-numeric:tabular-nums}"
    f".km .km-table td.mono{{font-family:{_MONO};font-size:11.5px}}"
    ".km .km-table td.muted{color:var(--muted)}"
    ".km .km-table tbody tr:nth-child(odd),.km .km-table tbody tr:nth-child(even){background:transparent}"
    ".km .km-table tbody tr:hover{background:var(--hover)}"
    ".km .km-table tbody tr:last-child td{border-bottom:0}"
    ".km-details{margin-top:8px;border-top:1px solid var(--line);padding-top:6px}"
    ".km-details>summary{cursor:pointer;list-style:none;display:inline-flex;align-items:center;gap:7px;font-size:12px;"
    "color:var(--muted);font-weight:500;padding:4px 8px;margin-left:-8px;border-radius:7px;user-select:none}"
    ".km-details>summary::-webkit-details-marker{display:none}"
    '.km-details>summary::before{content:"";width:0;height:0;border-left:5px solid currentColor;border-top:4px solid transparent;'
    "border-bottom:4px solid transparent;transition:transform .15s}"
    ".km-details[open]>summary::before{transform:rotate(90deg)}"
    ".km-details>summary:hover{background:var(--hover);color:var(--text)}"
    ".km-details>summary .km-faint{font-weight:400}"
    ".km-details>div{padding:8px 0 4px}"
    f".km pre.km-json{{font-family:{_MONO};font-size:11.5px;line-height:1.5;background:var(--bg2);border:1px solid var(--line);"
    "border-radius:9px;padding:10px 12px;margin:0;max-height:360px;overflow:auto;color:var(--text);white-space:pre}"
    ".km-cf{display:flex;flex-direction:column;gap:6px;font-size:12.5px}"
    ".km-cf div{display:flex;gap:8px;align-items:center;flex-wrap:wrap}"
    ".km-cf .km-arrow{color:var(--indigo);font-weight:600}"
    ".km-faint{color:var(--faint);font-size:12px}"
    ".km-muted{color:var(--muted)}"
    ".km-pics{display:flex;gap:10px;flex-wrap:wrap;margin:0 0 12px}"
    ".km-pic{display:flex;flex-direction:column;gap:4px;font-size:11px;color:var(--muted)}"
    ".km-pic img{border-radius:8px;border:1px solid var(--line);display:block;max-width:176px;max-height:176px;"
    "image-rendering:auto}"
    ".km-pic img.pixel{image-rendering:pixelated}"
    ".km-curves{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px 16px}"
    ".km-curve{display:flex;gap:10px;align-items:center;min-width:0}"
    ".km-curve svg{flex:none;overflow:visible}"
    f".km-curve .km-name{{font-family:{_MONO};font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}"
    ".km-rules{display:flex;flex-direction:column;gap:6px;font-size:12.5px}"
    ".km-rules div{display:flex;gap:8px}"
    ".km-rules .km-n{color:var(--faint);font-variant-numeric:tabular-nums;min-width:18px;text-align:right}"
    ".km-chart{position:relative}.km-chart svg{display:block;width:100%;height:auto;max-width:420px}"
    ".km-chart rect.bin:hover{opacity:.7}"
    ".km-chart text{font-size:10px;fill:var(--faint)}"
    ".km-foot{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;margin-top:10px;"
    "padding-top:8px;border-top:1px solid var(--line);font-size:11.5px;color:var(--muted)}"
    ".km-foot a{color:var(--indigo);text-decoration:none;font-weight:500}.km-foot a:hover{text-decoration:underline}"
    ".km-prob{display:grid;grid-template-columns:minmax(80px,160px) 1fr 56px;align-items:center;gap:10px;font-size:12px;margin-bottom:4px}"
    f".km-prob .km-val{{text-align:right;font-variant-numeric:tabular-nums;font-family:{_MONO};font-size:11.5px}}"
    ".km-desc{font-size:12.5px;color:var(--muted);margin:0 0 10px}"
    ".km-error{color:var(--bad);font-size:12.5px;margin:0 0 10px}"
)

METHOD_LABEL = {
    "shap": "KernelSHAP",
    "shapley": "Permutation Shapley",
    "lime": "LIME",
    "auto": "Auto",
    "gradcam": "Grad-CAM",
    "occlusion": "Occlusion",
}
TASK_SHORT = {"binary_classification": "Binary", "multiclass_classification": "Multiclass", "regression": "Regression"}
SEVERITY_TONE = {"info": "info", "low": "ok", "medium": "warn", "high": "bad"}
STATUS_TONE = {
    "queued": "",
    "running": "info",
    "completed": "ok",
    "failed": "bad",
    "active": "ok",
    "archived": "",
    "stable": "ok",
    "moderate": "warn",
    "significant": "bad",
}
#: A JSON array longer than this, or any nested one, is elided in the card's JSON view.
_JSON_ARRAY_LIMIT = 12


# --- the dashboard's formatting (src/lib/format.ts), name for name ----------------------------------


def fmt_number(value: Any, digits: int = 2) -> str:
    """``fmtNumber``: up to ``digits`` decimals, thousands separators, trailing zeros dropped."""
    number = _float(value)
    if number is None:
        return "—"
    text = f"{number:,.{digits}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("-0", "−0") else text


def compact(value: Any) -> str:
    """``compact``: ``1.2M``, ``12.3K``, else the number with separators."""
    number = _float(value)
    if number is None:
        return "—"
    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:.{0 if number % 1_000_000 == 0 else 1}f}M"
    if abs(number) >= 10_000:
        return f"{number / 1000:.{0 if number % 1000 == 0 else 1}f}K"
    return f"{int(number):,}" if number == int(number) else fmt_number(number, 3)


def pct(value: Any, digits: int = 0) -> str:
    """``pct``: a fraction as a percentage."""
    number = _float(value)
    return "—" if number is None else f"{number * 100:.{digits}f}%"


def fmt_signed(value: Any, digits: int = 3) -> str:
    """``fmtSigned``: ``+0.123``, ``−0.123`` (a real minus sign) or ``0.000``."""
    number = _float(value)
    if number is None:
        return "—"
    text = f"{abs(number):.{digits}f}"
    if number > 0:
        return "+" + text
    if number < 0:
        return _MINUS + text
    return text


def fmt_value(value: Any, unit: Optional[str] = None) -> str:
    """``fmtValue``: an input value as the dashboard prints it."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return "—"
        text = f"{number:,.0f}" if abs(number) >= 1000 else fmt_number(number, 3)
        return f"{text} {unit}" if unit else text
    if isinstance(value, (list, tuple)):
        return f"[{len(value)} values]"
    if isinstance(value, Mapping):
        return "{…}"
    return str(value)


def output_parts(output: Any) -> Tuple[str, Optional[str]]:
    """``outputParts``: the headline of a model output and its secondary figure."""
    if not isinstance(output, Mapping) or not output:
        return "—", None
    label = output.get("label")
    if isinstance(label, str):
        probability = output.get("probability")
        if not isinstance(probability, (int, float)) or isinstance(probability, bool):
            probability = output.get("score")
        if not isinstance(probability, (int, float)) or isinstance(probability, bool):
            return label, None
        return label, pct(probability, 1)
    value = output.get("value")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        unit = output.get("unit")
        return fmt_number(value, 3), str(unit) if isinstance(unit, str) else None
    score = output.get("score")
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        return fmt_number(score, 3), None
    return "—", None


def title_case(text: Any) -> str:
    return " ".join(word[:1].upper() + word[1:] for word in str(text or "").replace("_", " ").split(" "))


def long_date(value: Any) -> str:
    """``longDate``, in UTC: ``Oct 25, 2026, 15:04 UTC``."""
    if not value:
        return "—"
    text = str(value)
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    suffix = " UTC" if moment.tzinfo is not None and moment.utcoffset() == timedelta(0) else ""
    return f"{moment:%b} {moment.day}, {moment.year}, {moment:%H:%M}{suffix}"


def short_date(value: Any) -> str:
    """A date for a tile or a table cell, in UTC: ``Oct 25, 15:04``."""
    if not value:
        return "—"
    text = str(value)
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    return f"{moment:%b} {moment.day}, {moment:%H:%M}"


def _date_cell(value: Any) -> str:
    """The short date, with the full one on hover."""
    if not value:
        return "—"
    return f'<span title="{_e(long_date(value))}">{_e(short_date(value))}</span>'


def duration_between(start: Any, end: Any) -> str:
    if not start or not end:
        return "—"
    try:
        seconds = round(
            (datetime.fromisoformat(str(end).replace("Z", "+00:00")) - datetime.fromisoformat(str(start).replace("Z", "+00:00"))).total_seconds()
        )
    except ValueError:
        return "—"
    if seconds < 0:
        return "—"
    return f"{seconds}s" if seconds < 60 else f"{seconds // 60}m {seconds % 60}s"


def _map(value: Any) -> Mapping[str, Any]:
    """``value`` when it is a mapping, else an empty one."""
    return value if isinstance(value, Mapping) else {}


def _float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return None if math.isnan(number) or math.isinf(number) else number


# --- building blocks -----------------------------------------------------------------------------


def _e(value: Any) -> str:
    return escape(str(value), quote=True)


def _badge(text: Any, tone: str = "", *, mono: bool = False, title: Optional[str] = None, dot: bool = False) -> str:
    classes = " ".join(c for c in ("km-badge", tone, "mono" if mono else "") if c)
    attrs = f' title="{_e(title)}"' if title else ""
    return f'<span class="{classes}"{attrs}>{"<i class=km-dot></i>" if dot else ""}{_e(text)}</span>'


def _badges(items: Iterable[str]) -> str:
    items = [item for item in items if item]
    return f'<div class="km-badges">{"".join(items)}</div>' if items else ""


def _when(data: Mapping[str, Any], prefix: str = "", key: str = "created_at") -> str:
    """A record's timestamp for the header or footer, or nothing when it has none."""
    return f"{prefix}{_e(long_date(data[key]))}" if data.get(key) else ""


def _head(eyebrow: str, ident: Any = None, right: str = "") -> str:
    parts = [f'<span class="km-brand">{MARK}<span class="km-word">kai<i>mei</i></span>']
    parts.append(f'<span class="km-eyebrow">{_e(eyebrow)}</span>')
    if ident:
        parts.append(f'<span class="km-id km-mono">{_e(ident)}</span>')
    parts.append("</span>")
    return f'<div class="km-head">{"".join(parts)}<span class="km-when">{right}</span></div>'


def _details(summary: str, body: str, *, open_: bool = False, note: str = "") -> str:
    hint = f' <span class="km-faint">{note}</span>' if note else ""
    return (
        f'<details class="km-details"{" open" if open_ else ""}><summary>{summary}{hint}</summary>'
        f"<div>{body}</div></details>"
    )


def _raw_json(summary: str, value: Any, note: str = "") -> str:
    """A folded JSON section, hidden where the stylesheet is missing.

    GitHub's notebook viewer drops ``<style>``, ``<details>`` and ``<summary>`` (keeping their
    contents), which would unfold a long JSON dump in every card. The wrapper is hidden inline and
    the stylesheet shows it again with ``!important``.
    """
    return f'<div class="km-raw" style="display:none">{_details(summary, _json_block(value), note=note)}</div>'


def _json_block(value: Any) -> str:
    text = json.dumps(_light(value), indent=2, default=str, ensure_ascii=False)
    return f'<pre class="km-json">{_e(text)}</pre>'


def _light(value: Any) -> Any:
    """The JSON view's data: long arrays and nested arrays become ``[N values]`` labels."""
    if isinstance(value, Mapping):
        return {str(k): _light(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (list, tuple)):
            return f"[{'×'.join(str(n) for n in _shape(value))} values]"
        if len(value) > _JSON_ARRAY_LIMIT:
            return f"[{len(value)} values]"
        return [_light(v) for v in value]
    return value


def _shape(value: Any) -> List[int]:
    shape: List[int] = []
    while isinstance(value, (list, tuple)) and value:
        shape.append(len(value))
        value = value[0]
    return shape or [0]


def _table(columns: Sequence[Tuple[str, str]], rows: Iterable[Sequence[str]]) -> str:
    """``columns`` are ``(label, css class)``; cells are HTML already."""
    head = "".join(f'<th class="{_e(cls)}">{_e(label)}</th>' for label, cls in columns)
    body = []
    for row in rows:
        cells = "".join(f'<td class="{_e(cls)}">{cell}</td>' for (_, cls), cell in zip(columns, row))
        body.append(f"<tr>{cells}</tr>")
    return f'<div class="km-wrap"><table class="km-table"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _kv_table(items: Iterable[Tuple[str, str]]) -> str:
    rows = "".join(f'<tr><td class="mono muted">{_e(k)}</td><td class="num mono">{v}</td></tr>' for k, v in items)
    return f'<div class="km-wrap"><table class="km-table"><tbody>{rows}</tbody></table></div>'


def _diverging_bars(rows: Sequence[Tuple[str, Optional[str], float]], *, legend: Optional[Tuple[str, str]] = None) -> str:
    """The dashboard's ``AttributionBars``: ``rows`` are ``(name, input text or None, value)``."""
    if not rows:
        return '<div class="km-faint">No attributions yet.</div>'
    largest = max(max(abs(v) for _, _, v in rows), 1e-9)
    out = []
    if legend:
        out.append(
            '<div class="km-legend"><span><i style="background:linear-gradient(90deg,#3ec8ff,#6b8cff)"></i>'
            f"{_e(legend[0])}</span><span><i style=\"background:linear-gradient(90deg,#ff6bb5,#c86bff)\"></i>{_e(legend[1])}</span></div>"
        )
    out.append('<div class="km-bars">')
    for name, input_text, value in rows:
        width = max(abs(value) / largest * 50, 1.5)
        positive = value >= 0
        left = 50.0 if positive else 50.0 - width
        suffix = f' <span class="km-muted">· {_e(input_text)}</span>' if input_text else ""
        out.append(
            f'<div class="km-row" title="{_e(name)}: {fmt_signed(value, 6)}"><div class="km-name"><b>{_e(name)}</b>{suffix}</div>'
            f'<div class="km-track"><span class="km-zero"></span><span class="km-fill {"pos" if positive else "neg"}" '
            f'style="left:{left:.2f}%;width:{width:.2f}%"></span></div><div class="km-val">{fmt_signed(value)}</div></div>'
        )
    out.append("</div>")
    return "".join(out)


def _share_bars(
    rows: Sequence[Tuple[str, float, str, str]], *, legend: Optional[Sequence[Tuple[str, str]]] = None
) -> str:
    """Single-hue bars from zero: ``rows`` are ``(name, value, value text, fill class)``."""
    if not rows:
        return ""
    largest = max(max(abs(v) for _, v, _, _ in rows), 1e-9)
    out = []
    if legend:
        chips = "".join(f'<span><i class="km-fill {_e(cls)}" style="position:static"></i>{_e(text)}</span>' for text, cls in legend)
        out.append(f'<div class="km-legend">{chips}</div>')
    out.append('<div class="km-bars">')
    for name, value, text, fill in rows:
        out.append(
            f'<div class="km-row share" title="{_e(name)}: {_e(text)}"><div class="km-name"><b>{_e(name)}</b></div>'
            f'<div class="km-track"><span class="km-fill {_e(fill)}" style="left:0;width:{abs(value) / largest * 100:.2f}%"></span></div>'
            f'<div class="km-val">{_e(text)}</div></div>'
        )
    out.append("</div>")
    return "".join(out)


def _tiles(items: Iterable[Tuple[str, str, str]]) -> str:
    cells = "".join(
        f'<div class="km-tile"><div class="km-k">{_e(k)}</div><div class="km-v">{v}</div>{f"<div class=km-f>{_e(f)}</div>" if f else ""}</div>'
        for k, v, f in items
    )
    return f'<div class="km-tiles">{cells}</div>' if cells else ""


def _link(path: str, text: str) -> str:
    base = os.environ.get("KAIMEI_DASHBOARD_URL", DASHBOARD_URL).rstrip("/")
    return f'<a href="{_e(base + path)}" target="_blank" rel="noopener">{_e(text)} ↗</a>'


def _foot(left: str, link: str = "") -> str:
    return f'<div class="km-foot"><span>{left}</span><span>{link}</span></div>'


def _sparkline(values: Sequence[float], width: int = 150, height: int = 44, title: str = "") -> str:
    """The dashboard's ``Curve``: a behaviour curve scaled to its own range."""
    if len(values) < 2:
        return ""
    low, high = min(values), max(values)
    span = (high - low) or 1.0
    points = [
        (i / (len(values) - 1) * width, height - 4 - (v - low) / span * (height - 8)) for i, v in enumerate(values)
    ]
    path = " ".join(f"{'L' if i else 'M'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(points))
    caption = f"<title>{_e(title)}</title>" if title else ""
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" aria-hidden="true">{caption}'
        f'<path d="{path}" fill="none" stroke="var(--series-1)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'
    )


def _histogram(counts: Sequence[float], edges: Sequence[float], threshold: Optional[float] = None, height: int = 130) -> str:
    """The dashboard's ``Histogram``, with a tooltip per bin."""
    if not counts or len(edges) < len(counts) + 1:
        return ""
    width, pad_l, pad_r, pad_t, pad_b, gap = 400, 8, 8, 8, 20, 2
    largest = max(max(counts), 1)
    inner_w, inner_h = width - pad_l - pad_r, height - pad_t - pad_b
    bar_w = inner_w / max(1, len(counts)) - gap
    low, high = float(edges[0]), float(edges[-1])
    span = (high - low) or 1.0

    def fmt(v: float) -> str:
        return f"{v:.0f}" if abs(v) >= 100 else f"{v:.2f}"

    out = [f'<div class="km-chart"><svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Output distribution">']
    for i, count in enumerate(counts):
        h = count / largest * inner_h
        x = pad_l + i * (bar_w + gap)
        middle = (float(edges[i]) + float(edges[i + 1])) / 2
        above = threshold is not None and middle >= threshold
        out.append(
            f'<rect class="bin" x="{x:.1f}" y="{pad_t + inner_h - h:.1f}" width="{max(bar_w, 1):.1f}" height="{h:.1f}" rx="2" '
            f'fill="var(--series-{2 if above else 1})"><title>{fmt(float(edges[i]))} – {fmt(float(edges[i + 1]))}: '
            f"{compact(count)}</title></rect>"
        )
    if threshold is not None:
        x = pad_l + (threshold - low) / span * inner_w
        out.append(
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{pad_t}" y2="{pad_t + inner_h}" stroke="var(--faint)" stroke-width="1" stroke-dasharray="3 3"/>'
            f'<text x="{x:.1f}" y="{height - 4}" text-anchor="middle">threshold {fmt_number(threshold, 3)}</text>'
        )
    out.append(f'<text x="{pad_l}" y="{height - 4}">{fmt(low)}</text>')
    out.append(f'<text x="{width - pad_r}" y="{height - 4}" text-anchor="end">{fmt(high)}</text>')
    out.append("</svg></div>")
    return "".join(out)


# --- cards ---------------------------------------------------------------------------------------


def render(data: Mapping[str, Any]) -> str:
    """One HTML card for a record, chosen by its ``object`` type.

    The stylesheet comes first and each element repeats its resting style inline, so the card keeps
    its layout in viewers that remove ``<style>`` blocks, such as GitHub's notebook viewer.
    """
    kind = str(data.get("object") or "")
    builder: Callable[[Mapping[str, Any]], str] = {
        "explanation": _explanation,
        "insight": _insight,
        "model": _model,
        "prediction": _prediction,
        "analysis": _analysis,
        "list": _page,
    }.get(kind, _generic)
    card = f'<div class="{"km wide" if kind == "list" else "km"}">{builder(data)}</div>'
    # The stylesheet drives themes and hover; the inline copy keeps the layout where a viewer
    # drops <style> blocks (GitHub's notebook viewer does).
    return f"<style>{CSS}</style>{inline_styles(card, CSS)}"


def _explanation(data: Mapping[str, Any]) -> str:
    output = _map(data.get("output"))
    main, sub = output_parts(output)
    label = output.get("label")
    target = output.get("target_label")
    classification = isinstance(label, str) or bool(output.get("probabilities"))
    inputs = _map(data.get("input"))
    image_key = _image_key(inputs)
    method = str(data.get("method") or "")

    out = [_head("Explanation", data.get("id"), _when(data))]
    head_sub = ""
    if sub:
        head_sub = f"P({_e(label)}) = {_e(sub)}" if classification and isinstance(label, str) and sub.endswith("%") else _e(sub)
    out.append(f'<div class="km-result"><span class="km-value">{_e(main)}</span>' + (f'<span class="km-sub">{head_sub}</span>' if head_sub else "") + "</div>")
    badges = [_badge(METHOD_LABEL.get(method, method or "attributions"), "ok")]
    if isinstance(target, str) and target != label:
        badges.append(_badge(f"explains {target}", "info"))
    if _float(data.get("base_value")) is not None:
        badges.append(_badge(f"base {fmt_number(data['base_value'], 3)}", mono=True, title="Average prediction over the reference data"))
    if data.get("n_evals") is not None:
        badges.append(_badge(f"{compact(data.get('n_evals'))} evals", mono=True, title="Model calls made for this explanation"))
    if _float(data.get("latency_ms")) is not None:
        badges.append(_badge(f"{fmt_number(data['latency_ms'], 0)} ms", mono=True, title="Server time"))
    if data.get("prediction_id"):
        badges.append(_badge(data["prediction_id"], mono=True, title="Prediction"))
    out.append(_badges(badges))

    if image_key:
        out.append(_image_section(data, image_key, target if isinstance(target, str) else label))
    else:
        lead = _tabular_lead(data)
        if lead:
            out.append(f'<div class="km-lead">{_e(lead)}</div>')
        rows = _attribution_rows(data, inputs)
        if classification and isinstance(target, str):
            legend = (f"pushes toward {target}", "pushes away")
        elif classification:
            legend = ("pushes toward the prediction", "pushes away")
        else:
            legend = ("pushes higher", "pushes lower")
        out.append('<div class="km-h">Attributions</div>')
        out.append(_diverging_bars(rows[:14], legend=legend))
        if len(rows) > 14:
            out.append(_details(f"All {len(rows)} features", _diverging_bars(rows[14:])))

    probabilities = output.get("probabilities")
    if isinstance(probabilities, Mapping) and len(probabilities) >= 2:
        ranked = sorted(((str(k), _float(v) or 0.0) for k, v in probabilities.items()), key=lambda kv: -kv[1])
        shown = ranked[:6]
        lines = "".join(
            f'<div class="km-prob" title="P({_e(name)}) = {pct(value, 2)}"><span class="km-mono">{_e(name)}</span>'
            f'<div class="km-meter"><i style="width:{max(value * 100, 0.5):.2f}%"></i></div><span class="km-val">{pct(value, 1)}</span></div>'
            for name, value in shown
        )
        note = f' <span class="km-faint">{len(ranked)} classes</span>' if len(ranked) > len(shown) else ""
        out.append(_details(f"Probabilities{note}", lines, open_=len(ranked) <= 6 and not image_key))

    counterfactual = data.get("counterfactual")
    if isinstance(counterfactual, Mapping) and counterfactual:
        quantity = f"P({target})" if classification and isinstance(target, str) else None
        out.append(_counterfactual(counterfactual, classification, quantity))

    if inputs:
        out.append(_details("Input", _kv_table((k, _e(_input_text(v))) for k, v in inputs.items()), note=f"{len(inputs)} features"))
    out.append(_raw_json("JSON", data, note="arrays elided; to_json() has everything"))
    left = f"Logged as <span class=km-mono>{_e(data.get('id') or '—')}</span>"
    if data.get("model_id"):
        left += f" · model <span class=km-mono>{_e(data['model_id'])}</span>"
    link = _link(f"/explanations/{data['id']}", "Open in dashboard") if data.get("id") else ""
    out.append(_foot(left, link))
    return "".join(out)


def _attribution_rows(data: Mapping[str, Any], inputs: Mapping[str, Any]) -> List[Tuple[str, Optional[str], float]]:
    rows: List[Tuple[str, Optional[str], float]] = []
    top = [f for f in (data.get("top_features") or []) if isinstance(f, Mapping)]
    seen = set()
    for feature in top:
        name = str(feature.get("feature"))
        value = _float(feature.get("attribution"))
        if value is None:
            value = _float(feature.get("mean_attribution")) or _float(feature.get("mean_abs_attribution")) or 0.0
        rows.append((name, _bar_input(feature.get("value", inputs.get(name))), value))
        seen.add(name)
    attributions = _map(data.get("attributions"))
    rest = [(str(k), _float(v) or 0.0) for k, v in attributions.items() if str(k) not in seen]
    rest.sort(key=lambda kv: -abs(kv[1]))
    rows.extend((name, _bar_input(inputs.get(name)), value) for name, value in rest)
    return rows


def _bar_input(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = fmt_value(value)
    return None if text == "{…}" or text.startswith("[") else text


def _input_text(value: Any) -> str:
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        return "×".join(str(n) for n in _shape(value)) + " image"
    return fmt_value(value)


def _tabular_lead(data: Mapping[str, Any]) -> str:
    try:
        from .tabular import explanation_summary

        return explanation_summary(data)
    except Exception:  # pragma: no cover - a record shape the summary cannot read
        return ""


def _counterfactual(cf: Mapping[str, Any], classification: bool, quantity: Optional[str] = None) -> str:
    """The what-if search. ``quantity`` names the output it moves, e.g. ``P(survived)``."""
    changes = [c for c in (cf.get("changes") or []) if isinstance(c, Mapping)]
    found = bool(cf.get("found"))
    lines = []
    for change in changes:
        feature = str(change.get("feature") or change.get("name") or "")
        after = _float(change.get("output_after"))
        tail = ""
        if after is not None:
            label_after = change.get("label_after")
            value = pct(after, 1) if classification else fmt_number(after, 3)
            tail = f' <span class="km-faint">→ {_e(quantity) + " " if quantity else ""}{value}'
            tail += f" · {_e(label_after)}</span>" if label_after else "</span>"
        lines.append(
            f'<div><span class="km-arrow">→</span><span>Set <b>{_e(feature)}</b> from <b>{_e(fmt_value(change.get("from")))}</b> '
            f'<span class="km-faint">→</span> <b>{_e(fmt_value(change.get("to")))}</b>{tail}</span></div>'
        )
    before, after = _float(cf.get("output_before")), _float(cf.get("output_after"))
    if before is not None and after is not None:
        shown = (pct(before, 1), pct(after, 1)) if classification else (fmt_number(before, 3), fmt_number(after, 3))
        lines.append(f'<span class="km-faint">{_e(quantity or "Output")} {shown[0]} → {shown[1]}</span>')
    goal = _map(cf.get("target"))
    if goal.get("kind") == "flip_decision" and goal.get("threshold") is not None:
        lines.append(f'<span class="km-faint">Decision threshold {pct(goal["threshold"], 1) if classification else fmt_number(goal["threshold"], 3)}</span>')
    elif goal.get("kind") == "reach_value" and goal.get("target_value") is not None:
        lines.append(f'<span class="km-faint">Target value {fmt_number(goal["target_value"], 3)}</span>')
    if not changes:
        body = '<span class="km-faint">The prediction already meets the target; nothing needs to change.</span>' if found else \
            '<span class="km-faint">No change to the allowed features moves the prediction to the target.</span>'
    else:
        body = f'<div class="km-cf">{"".join(lines)}</div>'
    title = "Smallest change that flips the decision" if found else "Closest the counterfactual search got"
    return _details(title, body, open_=True)


def _image_key(inputs: Mapping[str, Any]) -> Optional[str]:
    for key, value in inputs.items():
        if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)) and value[0] and isinstance(value[0][0], (list, tuple)):
            return str(key)
    return None


def _image_section(data: Mapping[str, Any], key: str, target: Any) -> str:
    from . import images

    out = []
    try:
        summary = images.explanation_summary(data)
    except Exception:
        summary = str(_map(data.get("options")).get("summary") or "")
    if summary:
        out.append(f'<div class="km-lead">{_e(summary)}</div>')
    out.append(_image_panels(data, key))
    options = _map(data.get("options"))
    evidence = _map(options.get("evidence"))
    if evidence:
        who = evidence.get("label") or target or "the class"
        chips = [
            _badge(f"P({who}) photo {pct(evidence.get('full'), 1)}", mono=True, title="The whole photo"),
            _badge(f"focus hidden {pct(evidence.get('without_focus'), 1)}", mono=True, title="The focus region hidden"),
            _badge(f"focus only {pct(evidence.get('focus_only'), 1)}", mono=True, title="Only the focus region kept"),
        ]
        out.append(_badges(chips))
    regions: List[Dict[str, Any]] = []
    try:
        regions = images.numbered_regions(data)
    except Exception:
        regions = []
    if regions:
        rows = [
            (f"{r['number']} · {r['name']}", float(r["attribution"]), fmt_signed(r["attribution"]), "for" if r["group"] == "for" else "against")
            for r in regions
        ]
        who = target or "the class"
        legend = [(f"For {who}", "for")] + ([(f"Against {who}", "against")] if any(r["group"] == "against" for r in regions) else [])
        out.append('<div class="km-h">Regions</div>')
        out.append(_share_bars(rows, legend=legend))
    else:
        attributions = _map(data.get("attributions"))
        rows = [(str(k), _float(v) or 0.0, fmt_signed(v), "for" if (_float(v) or 0.0) >= 0 else "against") for k, v in attributions.items()]
        if rows:
            out.append('<div class="km-h">Regions</div>')
            out.append(_share_bars(rows))
    return "".join(out)


def _image_panels(data: Mapping[str, Any], key: str) -> str:
    try:
        import numpy as np
        from PIL import Image

        from . import images
    except ImportError:
        return ""
    try:
        photo = data["input"][key]
        pixels = np.asarray(photo, dtype=float)
        feature = {"type": "image", "value_range": [0, 255]} if pixels.size and float(pixels.max()) > 1.0 else None
        panels = []
        for mode, label in (("original", "Photo"), ("heatmap", "Evidence"), ("spotlight", "Focus")):
            picture = images.render_explanation(photo, data, mode=mode, feature=feature).convert("RGB")
            pixelated = max(picture.size) < 96
            if pixelated:
                scale = max(1, 128 // max(picture.size))
                picture = picture.resize((picture.width * scale, picture.height * scale), getattr(Image, "Resampling", Image).NEAREST)
            else:
                picture.thumbnail((176, 176))
            buffer = io.BytesIO()
            picture.save(buffer, format="PNG")
            uri = base64.b64encode(buffer.getvalue()).decode("ascii")
            panels.append(
                f'<div class="km-pic"><img class="{"pixel" if pixelated else ""}" src="data:image/png;base64,{uri}" '
                f'width="{picture.width}" height="{picture.height}" alt="{label}" title="{label}"><span>{label}</span></div>'
            )
    except Exception:
        return ""
    return f'<div class="km-pics">{"".join(panels)}</div>'


def _insight(data: Mapping[str, Any]) -> str:
    kind = str(data.get("type") or "")
    payload = _map(data.get("payload"))
    severity = str(data.get("severity") or "")
    out = [_head("Insight", data.get("id"), _when(data))]
    out.append(f'<div class="km-result"><span class="km-title">{_e(data.get("title") or title_case(kind))}</span></div>')
    badges = [_badge(title_case(kind), "info")]
    if severity:
        badges.append(_badge(title_case(severity), SEVERITY_TONE.get(severity, ""), dot=True))
    if _float(data.get("score")) is not None:
        badges.append(_badge(f"score {fmt_number(data['score'], 3)}", mono=True))
    if data.get("analysis_id"):
        badges.append(_badge(data["analysis_id"], mono=True, title="Analysis"))
    out.append(_badges(badges))
    summary = str(data.get("summary") or "").replace("`", "")
    if summary:
        out.append(f'<div class="km-lead">{_e(summary)}</div>')
    out.append(_insight_payload(kind, payload))
    out.append(_raw_json("Payload JSON", payload, note="arrays elided; to_dict() has everything"))
    left = f"model <span class=km-mono>{_e(data['model_id'])}</span>" if data.get("model_id") else ""
    link = _link(f"/insights/{data['id']}", "Open in dashboard") if data.get("id") else ""
    out.append(_foot(left, link))
    return "".join(out)


def _dicts(value: Any) -> List[Mapping[str, Any]]:
    return [v for v in value if isinstance(v, Mapping)] if isinstance(value, (list, tuple)) else []


def _insight_payload(kind: str, payload: Mapping[str, Any]) -> str:
    """The dashboard's ``InsightPayload`` for each insight type."""
    limit = 10
    if kind == "global_importance":
        rows = [
            (str(f.get("feature")), _float(f.get("share")) or _float(f.get("importance")) or 0.0, "", "share")
            for f in _dicts(payload.get("features"))
            if f.get("feature")
        ][:limit]
        rows = [(n, v, f"{v * 100:.0f}%", fill) for n, v, _, fill in rows]
        chips = []
        metric = payload.get("baseline_metric")
        if isinstance(metric, Mapping) and metric.get("name"):
            chips.append(_badge(f"{metric['name']} {fmt_number(metric.get('value'), 3)}", mono=True, title="Model metric on the reference rows"))
        return _badges(chips) + _share_bars(rows)
    if kind == "drift":
        features = [
            (str(f.get("feature")), _float(f.get("psi")) or 0.0, str(f.get("status") or ""))
            for f in _dicts(payload.get("features"))
            if f.get("feature")
        ][:limit]
        chips = []
        if _float(payload.get("overall_psi")) is not None:
            chips.append(_badge(f"PSI {fmt_number(payload['overall_psi'], 3)}", mono=True, title="Population stability index over every input"))
        if isinstance(payload.get("status"), str):
            chips.append(_badge(payload["status"], STATUS_TONE.get(payload["status"], "")))
        if payload.get("reference_rows") is not None and payload.get("current_rows") is not None:
            chips.append(_badge(f"{compact(payload['reference_rows'])} reference vs {compact(payload['current_rows'])} recent rows", mono=True))
        output_drift = payload.get("output_drift")
        if isinstance(output_drift, Mapping) and _float(output_drift.get("psi")) is not None:
            chips.append(_badge(f"predictions PSI {fmt_number(output_drift['psi'], 3)} · {output_drift.get('status', '')}", mono=True))
        rows = [(name, psi, fmt_number(psi, 3), status if status in ("stable", "moderate", "significant") else "share") for name, psi, status in features]
        legend = [("stable < 0.1", "stable"), ("moderate", "moderate"), ("significant > 0.25", "significant")]
        return _badges(chips) + _share_bars(rows, legend=legend if rows else None)
    if kind == "interactions":
        rows = []
        for pair in _dicts(payload.get("pairs")):
            pair_names = [str(f) for f in (pair.get("features") or [])]
            if len(pair_names) == 2:
                strength = _float(pair.get("strength")) or 0.0
                rows.append((f"{pair_names[0]} × {pair_names[1]}", strength, f"{strength * 100:.0f}%", "share"))
        return _share_bars(rows[:limit])
    if kind == "learned_rules":
        statements = [str(r.get("statement") or "").replace("`", "") for r in _dicts(payload.get("rules"))]
        statements = [s for s in statements if s][:12]
        if not statements:
            return ""
        items = "".join(f'<div><span class="km-n">{i}.</span><span>{_e(s)}</span></div>' for i, s in enumerate(statements, 1))
        return f'<div class="km-rules">{items}</div>'
    if kind == "partial_dependence":
        cards = []
        for curve in _dicts(payload.get("features"))[:6]:
            values = [v for v in (_float(x) for x in (curve.get("pd") or [])) if v is not None]
            if not curve.get("feature") or len(values) < 2:
                continue
            direction = curve.get("monotonicity")
            if isinstance(direction, Mapping):
                direction = direction.get("direction")
            direction = "" if curve.get("type") == "categorical" else str(direction or "").replace("_", " ")
            grid = list(curve.get("grid") or [])
            caption = f"{curve['feature']}: {fmt_number(min(values), 3)} to {fmt_number(max(values), 3)} over {len(grid) or len(values)} points"
            cards.append(
                f'<div class="km-curve">{_sparkline(values, title=caption)}<div style="min-width:0"><div class="km-name">{_e(curve["feature"])}</div>'
                + (f'<div class="km-faint">{_e(direction)}</div>' if direction else "")
                + "</div></div>"
            )
        return f'<div class="km-curves">{"".join(cards)}</div>' if cards else ""
    if kind in ("behavior", "behaviour"):
        histogram = _map(payload.get("histogram"))
        counts = [c for c in (_float(x) for x in (histogram.get("counts") or [])) if c is not None]
        edges = [e for e in (_float(x) for x in (histogram.get("edges") or [])) if e is not None]
        threshold = _float(payload.get("threshold"))
        chips = []
        positive = _float(payload.get("positive_rate"))
        if positive is not None:
            chips.append(_badge(f"positive {pct(positive, 1)}", mono=True))
        if _float(payload.get("near_threshold_rate")) is not None:
            chips.append(_badge(f"near threshold {pct(payload['near_threshold_rate'], 1)}", mono=True))
        balance = payload.get("class_balance")
        if isinstance(balance, Mapping) and positive is None:
            for name, share in list(balance.items())[:3]:
                chips.append(_badge(f"{name} {pct(share, 1)}", mono=True))
        if _float(payload.get("low_margin_rate")) is not None:
            chips.append(_badge(f"close calls {pct(payload['low_margin_rate'], 1)}", mono=True))
        if _float(payload.get("mean")) is not None:
            chips.append(_badge(f"mean {fmt_number(payload['mean'], 3)}", mono=True))
        if _float(payload.get("std")) is not None:
            chips.append(_badge(f"σ {fmt_number(payload['std'], 3)}", mono=True))
        return _histogram(counts, edges, threshold) + _badges(chips)
    if kind == "robustness":
        rows = []
        for f in _dicts(payload.get("fragile_features") or payload.get("features")):
            value = _float(f.get("sensitivity"))
            if value is None:
                value = _float(f.get("flip_rate")) or _float(f.get("share")) or _float(f.get("importance")) or 0.0
            if f.get("feature") and value > 0:
                rows.append((str(f["feature"]), value, fmt_number(value, 3), "share"))
        chips = []
        flip = _float(payload.get("flip_rate")) if _float(payload.get("flip_rate")) is not None else _float(payload.get("decision_flip_rate"))
        if flip is not None:
            chips.append(_badge(f"flips {pct(flip, 1)}", mono=True, title="Decisions that flip under small input noise"))
        sensitivity = _float(payload.get("sensitivity")) if _float(payload.get("sensitivity")) is not None else _float(payload.get("output_sensitivity"))
        if sensitivity is not None:
            chips.append(_badge(f"sensitivity {pct(sensitivity, 1)}", mono=True))
        if _float(payload.get("epsilon")) is not None:
            chips.append(_badge(f"ε {fmt_number(payload['epsilon'], 2)}", mono=True, title="Noise, as a fraction of each input's spread"))
        return _badges(chips) + _share_bars(rows[:limit])
    return ""


def _model(data: Mapping[str, Any]) -> str:
    out = [_head("Model", data.get("slug") or data.get("id"), _when(data, "created "))]
    out.append(f'<div class="km-result"><span class="km-title">{_e(data.get("name") or data.get("slug") or "Model")}</span></div>')
    task = str(data.get("task") or "")
    status = str(data.get("status") or "")
    badges = [
        _badge(f"v{data['version']}", mono=True) if data.get("version") else "",
        _badge(TASK_SHORT.get(task, task.replace("_", " ")), "info") if task else "",
        _badge(data["framework"]) if data.get("framework") else "",
        _badge(title_case(status), STATUS_TONE.get(status, ""), dot=True) if status else "",
        _badge(data["id"], mono=True, title="Model id") if data.get("id") and data.get("slug") else "",
    ]
    tags = data.get("tags")
    if isinstance(tags, (list, tuple)):
        badges.extend(_badge(f"#{tag}") for tag in tags)
    out.append(_badges(badges))
    if data.get("description"):
        out.append(f'<div class="km-desc">{_e(data["description"])}</div>')
    features = _dicts(data.get("features"))
    tiles = [
        ("Features", str(len(features)), ""),
        ("Reference rows", compact(data.get("reference_rows") or 0), "for explanations"),
        ("Predictions", compact(data.get("predictions_count") or 0), ""),
        ("Explanations", compact(data.get("explanations_count") or 0), ""),
        ("Last analyzed", _date_cell(data.get("last_analyzed_at")) if data.get("last_analyzed_at") else "never", ""),
    ]
    out.append(_tiles(tiles))
    outputs = _map(data.get("outputs"))
    chips = []
    labels = outputs.get("labels")
    if isinstance(labels, (list, tuple)) and labels:
        shown = ", ".join(str(label) for label in labels[:6]) + (f", … ({len(labels)})" if len(labels) > 6 else "")
        chips.append(_badge(f"classes {shown}", mono=True))
    if outputs.get("positive_label") is not None:
        chips.append(_badge(f"positive {outputs['positive_label']}", mono=True))
    if _float(outputs.get("threshold")) is not None:
        chips.append(_badge(f"threshold {fmt_number(outputs['threshold'], 3)}", mono=True))
    if outputs.get("name"):
        chips.append(_badge(f"output {outputs['name']}" + (f" ({outputs['unit']})" if outputs.get("unit") else ""), mono=True))
    methods = data.get("explain_methods")
    if isinstance(methods, (list, tuple)) and methods:
        chips.append(_badge("explains with " + ", ".join(METHOD_LABEL.get(str(m), str(m)) for m in methods)))
    inference = _map(data.get("inference"))
    if inference.get("type"):
        chips.append(_badge(f"inference {inference['type']}", mono=True))
    if chips:
        out.append('<div class="km-h">Outputs</div>' + _badges(chips))
    if features:
        rows = []
        for f in features:
            kind = str(f.get("type") or "")
            if kind == "image":
                detail = f"{f.get('height', '?')}×{f.get('width', '?')}×{f.get('channels', '?')}"
                if f.get("value_range"):
                    detail += f" in [{', '.join(fmt_number(v, 3) for v in f['value_range'])}]"
            elif f.get("categories"):
                cats = [str(c) for c in f["categories"]]
                detail = ", ".join(cats[:5]) + (f", … ({len(cats)})" if len(cats) > 5 else "")
            elif f.get("min") is not None or f.get("max") is not None:
                detail = f"{fmt_value(f.get('min'))} – {fmt_value(f.get('max'))}"
            else:
                detail = ""
            rows.append(
                [
                    f"<b>{_e(f.get('name'))}</b>",
                    _e(kind),
                    _e(f.get("unit") or ""),
                    _e(detail),
                    _e(fmt_value(f.get("baseline"))) if f.get("baseline") is not None else "",
                    _e(f.get("description") or ""),
                ]
            )
        table = _table(
            [("Feature", "mono"), ("Type", "muted"), ("Unit", "muted"), ("Range / categories", "muted"), ("Baseline", "num mono"), ("Description", "muted")],
            rows,
        )
        out.append(_details("Features", table, open_=len(features) <= 12, note=f"{len(features)}"))
    out.append(_raw_json("JSON", data, note="arrays elided; to_json() has everything"))
    link = _link(f"/models/{data['slug']}", "Open in dashboard") if data.get("slug") else ""
    out.append(_foot(_when(data, "updated ", "updated_at"), link))
    return "".join(out)


def _prediction(data: Mapping[str, Any]) -> str:
    output = _map(data.get("output"))
    main, sub = output_parts(output)
    label = output.get("label")
    out = [_head("Prediction", data.get("id"), _when(data))]
    head_sub = ""
    if sub:
        head_sub = f"P({_e(label)}) = {_e(sub)}" if isinstance(label, str) and sub.endswith("%") else _e(sub)
    out.append(f'<div class="km-result"><span class="km-value">{_e(main)}</span>' + (f'<span class="km-sub">{head_sub}</span>' if head_sub else "") + "</div>")
    badges = [
        _badge(data["source"], "info") if data.get("source") else "",
        _badge(f"{fmt_number(data['latency_ms'], 0)} ms", mono=True) if _float(data.get("latency_ms")) is not None else "",
        _badge(data["model_id"], mono=True, title="Model") if data.get("model_id") else "",
    ]
    out.append(_badges(badges))
    probabilities = output.get("probabilities")
    if isinstance(probabilities, Mapping) and len(probabilities) >= 2:
        ranked = sorted(((str(k), _float(v) or 0.0) for k, v in probabilities.items()), key=lambda kv: -kv[1])[:6]
        out.append(
            "".join(
                f'<div class="km-prob"><span class="km-mono">{_e(name)}</span><div class="km-meter"><i style="width:{max(value * 100, 0.5):.2f}%"></i></div>'
                f'<span class="km-val">{pct(value, 1)}</span></div>'
                for name, value in ranked
            )
        )
    inputs = _map(data.get("input"))
    if inputs:
        out.append(_details("Input", _kv_table((k, _e(_input_text(v))) for k, v in inputs.items()), open_=len(inputs) <= 12, note=f"{len(inputs)} features"))
    out.append(_raw_json("JSON", data, note="arrays elided; to_json() has everything"))
    out.append(_foot(f"Logged as <span class=km-mono>{_e(data.get('id') or '—')}</span>", ""))
    return "".join(out)


def _analysis(data: Mapping[str, Any]) -> str:
    status = str(data.get("status") or "")
    out = [_head("Analysis", data.get("id"), _when(data))]
    out.append(f'<div class="km-result"><span class="km-title">{_e(title_case(data.get("type") or "analysis"))} analysis</span></div>')
    badges = [
        _badge(title_case(status), STATUS_TONE.get(status, ""), dot=True) if status else "",
        _badge(data["model_id"], mono=True, title="Model") if data.get("model_id") else "",
    ]
    out.append(_badges(badges))
    progress = _float(data.get("progress"))
    if progress is not None and status in ("queued", "running"):
        share = progress if progress <= 1 else progress / 100
        out.append(f'<div class="km-meter" title="{pct(share, 0)}"><i style="width:{max(min(share, 1), 0) * 100:.1f}%"></i></div><div class="km-faint" style="margin:4px 0 10px">{pct(share, 0)} done</div>')
    insight_ids = data.get("insight_ids")
    insight_count = len(insight_ids) if isinstance(insight_ids, (list, tuple)) else 0
    tiles = [
        ("Insights", str(insight_count), ""),
        ("Model calls", compact(data.get("n_evals") or 0), ""),
        ("Started", _date_cell(data.get("started_at")), ""),
        ("Finished", _date_cell(data.get("completed_at")), duration_between(data.get("started_at"), data.get("completed_at")) if data.get("completed_at") else ""),
    ]
    out.append(_tiles(tiles))
    if data.get("error"):
        out.append(f'<div class="km-error">{_e(data["error"])}</div>')
    options = _map(data.get("options"))
    if options:
        out.append(_raw_json("Options", options))
    if insight_ids:
        out.append(_details("Insight ids", _kv_table((str(i + 1), f"<span class=km-mono>{_e(x)}</span>") for i, x in enumerate(insight_ids)), note=f"{len(insight_ids)}"))
    out.append(_raw_json("JSON", data))
    out.append(_foot(f"<span class=km-mono>{_e(data.get('id') or '—')}</span>", ""))
    return "".join(out)


def _page(data: Mapping[str, Any]) -> str:
    items = [item for item in (data.get("data") or []) if isinstance(item, Mapping)]
    kind = str(items[0].get("object") or "item") if items else "item"
    total = data.get("total")
    total_n = int(total) if isinstance(total, (int, float)) and not isinstance(total, bool) else len(items)
    offset = int(data.get("offset") or 0)
    noun = kind.replace("_", " ")
    plural = noun if total_n == 1 else noun + "s"
    out = [_head("List", None, f"{total_n:,} {plural}")]
    if not items:
        out.append('<div class="km-faint" style="margin:4px 0 8px">Nothing on this page.</div>')
    else:
        out.append(_page_table(kind, items))
    foot = f"{total_n:,} {plural}"
    if total_n > len(items):
        foot += f" · showing {offset + 1}–{offset + len(items)}"
    if data.get("has_more"):
        foot += " · more pages: <span class=km-mono>page.auto_paging_iter()</span>"
    out.append(_foot(foot, ""))
    return "".join(out)


def _page_table(kind: str, items: Sequence[Mapping[str, Any]]) -> str:
    if kind == "explanation":
        rows = []
        for e in items:
            main, sub = output_parts(e.get("output"))
            top = [f for f in (e.get("top_features") or []) if isinstance(f, Mapping)][:2]
            chips = " ".join(
                _badge(f"{f.get('feature')} {fmt_signed(_float(f.get('attribution')) if _float(f.get('attribution')) is not None else _float(f.get('mean_attribution')) or 0.0, 2)}", mono=True)
                for f in top
            )
            rows.append(
                [
                    _e(e.get("id") or ""),
                    _e(e.get("model_id") or ""),
                    f"<b>{_e(main)}</b> " + (f'<span class="km-faint">{_e(sub)}</span>' if sub else ""),
                    chips,
                    _badge(METHOD_LABEL.get(str(e.get("method")), str(e.get("method") or ""))),
                    _e(compact(e.get("n_evals"))),
                    f"{fmt_number(e['latency_ms'], 0)} ms" if _float(e.get("latency_ms")) is not None else "—",
                    _date_cell(e.get("created_at")),
                ]
            )
        return _table(
            [("ID", "mono"), ("Model", "mono muted"), ("Output", ""), ("Top features", "wrap"), ("Method", ""), ("Evals", "num"), ("Latency", "num"), ("When", "muted")],
            rows,
        )
    if kind == "insight":
        rows = [
            [
                _e(title_case(i.get("type"))),
                _badge(title_case(i.get("severity") or ""), SEVERITY_TONE.get(str(i.get("severity")), ""), dot=True),
                f"<b>{_e(i.get('title') or '')}</b>",
                _e(str(i.get("summary") or "").replace("`", "")),
                _e(fmt_number(i.get("score"), 3)) if _float(i.get("score")) is not None else "—",
                _date_cell(i.get("created_at")),
            ]
            for i in items
        ]
        return _table([("Type", "muted"), ("Severity", ""), ("Title", ""), ("Summary", "muted clip"), ("Score", "num mono"), ("When", "muted")], rows)
    if kind == "model":
        rows = [
            [
                f"<b>{_e(m.get('name') or '')}</b>",
                _e(m.get("slug") or ""),
                _e(f"v{m['version']}" if m.get("version") else ""),
                _e(TASK_SHORT.get(str(m.get("task")), str(m.get("task") or "").replace("_", " "))),
                _e(m.get("framework") or ""),
                _badge(title_case(m.get("status") or ""), STATUS_TONE.get(str(m.get("status")), ""), dot=True),
                _e(str(len(m.get("features") or []))),
                _e(compact(m.get("predictions_count") or 0)),
                _e(compact(m.get("explanations_count") or 0)),
            ]
            for m in items
        ]
        return _table(
            [("Model", ""), ("Slug", "mono muted"), ("Version", "mono"), ("Task", ""), ("Framework", "muted"), ("Status", ""), ("Features", "num"), ("Predictions", "num"), ("Explanations", "num")],
            rows,
        )
    if kind == "prediction":
        rows = []
        for p in items:
            main, sub = output_parts(p.get("output"))
            inputs = _map(p.get("input"))
            summary = "  ".join(f"{k}={_input_text(v)}" for k, v in list(inputs.items())[:6])
            rows.append(
                [
                    _e(p.get("id") or ""),
                    f"<b>{_e(main)}</b> " + (f'<span class="km-faint">{_e(sub)}</span>' if sub else ""),
                    f'<span title="{_e(summary)}">{_e(summary)}</span>',
                    f"{fmt_number(p['latency_ms'], 0)} ms" if _float(p.get("latency_ms")) is not None else "—",
                    _e(p.get("source") or ""),
                    _date_cell(p.get("created_at")),
                ]
            )
        return _table([("ID", "mono"), ("Output", ""), ("Input", "mono muted clip"), ("Latency", "num"), ("Source", "muted"), ("When", "muted")], rows)
    if kind == "analysis":
        rows = [
            [
                _e(a.get("id") or ""),
                _e(a.get("model_id") or ""),
                _e(a.get("type") or ""),
                _badge(title_case(a.get("status") or ""), STATUS_TONE.get(str(a.get("status")), ""), dot=True),
                _e(str(len(a.get("insight_ids") or []))),
                _e(compact(a.get("n_evals") or 0)),
                _date_cell(a.get("created_at")),
            ]
            for a in items
        ]
        return _table([("ID", "mono"), ("Model", "mono muted"), ("Type", ""), ("Status", ""), ("Insights", "num"), ("Evals", "num"), ("When", "muted")], rows)
    # Any other type: the scalar fields, id first, up to eight columns.
    columns: List[str] = []
    for item in items:
        for key, value in item.items():
            if key not in columns and not isinstance(value, (Mapping, list, tuple)) and key != "object":
                columns.append(key)
    columns = (["id"] if "id" in columns else []) + [c for c in columns if c != "id"]
    columns = columns[:8]
    rows = [[_e(fmt_value(item.get(c))) if not str(c).endswith("_at") else _date_cell(item.get(c)) for c in columns] for item in items]
    return _table([(title_case(c), "mono" if c == "id" else "") for c in columns], rows)


def _generic(data: Mapping[str, Any]) -> str:
    kind = str(data.get("object") or "record")
    title = data.get("name") or data.get("email") or data.get("slug") or data.get("title") or ""
    out = [_head(title_case(kind), data.get("id"), _when(data))]
    if title:
        out.append(f'<div class="km-result"><span class="km-title">{_e(title)}</span></div>')
    scalars = [
        (k, _e(long_date(v)) if str(k).endswith("_at") else _e(fmt_value(v)))
        for k, v in data.items()
        if k not in ("object",) and not isinstance(v, (Mapping, list, tuple))
    ]
    nested = {k: v for k, v in data.items() if isinstance(v, (Mapping, list, tuple))}
    if scalars:
        out.append(_kv_table(scalars))
    for key, value in nested.items():
        if isinstance(value, (list, tuple)) and value and all(isinstance(v, Mapping) for v in value):
            columns: List[str] = []
            for item in value:
                for k, v in item.items():
                    if k not in columns and not isinstance(v, (Mapping, list, tuple)):
                        columns.append(k)
            columns = columns[:8]
            rows = [[_e(fmt_value(item.get(c))) for c in columns] for item in value]
            out.append(_details(f"{title_case(key)}", _table([(title_case(c), "") for c in columns], rows), note=str(len(value))))
        else:
            out.append(_details(title_case(key), _json_block(value)))
    out.append(_raw_json("JSON", data))
    return "".join(out)
