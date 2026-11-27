"""Kaimei-themed HTML cards for notebooks.

``render(record)`` turns a record the API returned (an explanation, insight, model, prediction,
analysis, a page of them, ...) into one self-contained HTML card: the Kaimei mark, the dashboard's
colours and type, and the dashboard's numbers formatted by the same rules (its ``fmtSigned``,
``pct``, ``outputParts``, ``compact``, ... are ported below, name for name). What a model's log says
(a monitoring history, a population, a comparison of two models, similar cases, what Kaimei holds,
several explanation methods on one input) leads with the sentence a person would say about it.

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
from urllib.parse import quote

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
    "--track:rgba(18,20,28,.08);--midline:rgba(18,20,28,.18);--series-1:#4566e8;--series-2:#0891b2;--series-3:#f97316;"
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
    "--series-2:#3ec8ff;--series-3:#ffb04d;--hover:rgba(255,255,255,.035);--shadow:0 1px 2px rgba(0,0,0,.35)}"
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
    # The cards of what a model's log says (monitoring, population, comparison, similar cases, holdings, methods).
    ".km-say{font-size:15px;line-height:1.4;font-weight:500;letter-spacing:-.01em;color:var(--text);margin:2px 0 10px;max-width:680px}"
    ".km-win{position:relative;display:flex;align-items:flex-end;gap:2px;height:64px;border-bottom:1px solid var(--line2)}"
    ".km-win.dense{gap:0}"
    ".km-win-bar{flex:1 1 0%;min-width:1px;border-radius:2px 2px 0 0}"
    ".km-win-line{position:absolute;left:0;right:0;height:0;border-top:1px dashed var(--faint);opacity:.6}"
    ".km-tone-stable{background:var(--ok)}.km-tone-moderate{background:var(--warn)}.km-tone-significant{background:var(--bad)}"
    ".km-tone-chance{background:var(--faint)}.km-tone-none{background:var(--track)}.km-tone-traffic{background:var(--series-1)}"
    ".km-axis{display:flex;justify-content:space-between;gap:12px;font-size:11px;color:var(--faint);margin:3px 0 8px}"
    ".km .km-table td.bar,.km .km-table th.bar{width:30%}"
    ".km .km-table th.fold{white-space:normal;max-width:132px}"
    ".km .km-table td.said{white-space:normal;line-height:1.55}"
    ".km-track.cell{min-width:110px}"
    ".km-split{display:flex;gap:2px;height:10px;min-width:110px;border-radius:3px;overflow:hidden}"
    ".km-split i{display:block;height:100%;min-width:2px;border-radius:2px}"
    ".km-mix-0{background:var(--series-1)}.km-mix-1{background:var(--series-2)}.km-mix-2{background:var(--series-3)}"
    ".km-mix-x{background:var(--faint)}"
    ".km-gap{position:relative;height:14px;min-width:120px}"
    ".km-gap-track{position:absolute;left:0;right:0;top:6px;height:2px;border-radius:1px;background:var(--track)}"
    ".km-gap-line{position:absolute;top:6px;height:2px;background:var(--midline)}"
    # The sizes below count their borders: said here, because ``.km *{box-sizing}`` is not there without the stylesheet.
    ".km-pin{position:absolute;top:1px;width:12px;height:12px;margin-left:-6px;border-radius:50%;border:2px solid var(--bg);"
    "box-sizing:border-box}"
    ".km-sides{display:flex;gap:10px;flex-wrap:wrap;margin:0 0 4px}"
    ".km-side{flex:1 1 280px;min-width:0;background:var(--bg2);border:1px solid var(--line);border-radius:9px;padding:8px 12px 6px;"
    "box-sizing:border-box}"
    ".km-side-h{font-size:13px;font-weight:600;color:var(--text);margin:0 0 4px}"
    ".km-item{padding:5px 0;font-size:12.5px;border-top:1px solid var(--line)}"
    f".km-measure{{font-family:{_MONO};font-size:11.5px;color:var(--muted);margin-left:6px;white-space:nowrap}}"
    ".km-note{display:block;font-size:11.5px;color:var(--muted);margin-top:1px}"
    ".km-method{display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin:12px 0 6px;font-size:13px;font-weight:600;color:var(--text)}"
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
    "degraded": "bad",
    "improved": "ok",
}
METRIC_SHORT = {
    "auc": "AUC",
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "macro_precision": "macro precision",
    "macro_recall": "macro recall",
    "mae": "MAE",
}
#: Measures written as a share of rows.
SHARE_METRICS = frozenset({"accuracy", "precision", "recall", "macro_precision", "macro_recall"})
#: A JSON array longer than this, or any nested one, is elided in the card's JSON view.
_JSON_ARRAY_LIMIT = 12


# --- the dashboard's formatting (src/lib/format.ts), name for name ----------------------------------


def fmt_number(value: Any, digits: int = 2) -> str:
    """``fmtNumber``: up to ``digits`` decimals, thousands separators, trailing zeros dropped."""
    number = _float(value)
    if number is None:
        return "—"
    if digits > 0 and number != 0 and abs(number) < 10.0**-digits:
        # A small number isn't zero, and a model can work entirely at that scale.
        mantissa, exponent = f"{number:.1e}".split("e")
        return f"{mantissa}e{int(exponent)}"
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


def _diverging_legend(legend: Tuple[str, str]) -> str:
    return (
        '<div class="km-legend"><span><i style="background:linear-gradient(90deg,#3ec8ff,#6b8cff)"></i>'
        f"{_e(legend[0])}</span><span><i style=\"background:linear-gradient(90deg,#ff6bb5,#c86bff)\"></i>{_e(legend[1])}</span></div>"
    )


def _diverging_bars(
    rows: Sequence[Tuple[str, Optional[str], float]], *, legend: Optional[Tuple[str, str]] = None, scale: Optional[float] = None
) -> str:
    """The dashboard's ``AttributionBars``: ``rows`` are ``(name, input text or None, value)``.

    ``scale`` is the value a full bar stands for, for several sets of bars drawn on one scale; without it
    the largest of ``rows`` fills the bar.
    """
    if not rows:
        return '<div class="km-faint">No attributions yet.</div>'
    largest = max(max(abs(v) for _, _, v in rows), scale or 0.0, 1e-9)
    out = []
    if legend:
        out.append(_diverging_legend(legend))
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
        "decision_record": _decision_record,
        "reason_table": _reason_table,
        "evidence": _evidence,
        "retention": _retention,
        "model_report": _model_report,
        "insight": _insight,
        "model": _model,
        "prediction": _prediction,
        "analysis": _analysis,
        "model_summary": _model_summary,
        "model_case_study": _model_summary,  # its name in API 1.5.0
        "monitoring": _monitoring,
        "population": _population,
        "comparison": _comparison,
        "similar_cases": _similar,
        "model_holdings": _holdings,
        "method_comparison": _method_comparison,
        "list": _page,
    }.get(kind, _generic)
    card = f'<div class="{"km wide" if kind == "list" else "km"}">{builder(data)}</div>'
    # The stylesheet drives themes and hover; the inline copy keeps the layout where a viewer
    # drops <style> blocks (GitHub's notebook viewer does).
    return f"<style>{CSS}</style>{inline_styles(card, CSS)}"


def _model_summary(data: Mapping[str, Any]) -> str:
    """A model's summary, compact: what it is, the facts, what each section says, and what is not on file."""
    model = _map(data.get("model"))
    plain = _plain  # the API sets the names of inputs and classes in backticks; a card shows them as plain words
    out = [_head("Summary", model.get("slug"), _when(data, "generated ", key="generated_at"))]
    out.append(f'<div class="km-result"><span class="km-title">{_e(model.get("name") or "Model")}</span></div>')
    out.append(_say(data.get("headline")))
    facts = _dicts(data.get("facts"))
    out.append(_tiles((str(f.get("label") or ""), _e(f.get("value") or ""), str(f.get("note") or "")) for f in facts))
    tones = {"ready": "ok", "stale": "warn"}
    sections = []
    for number, section in enumerate(_dicts(data.get("sections")), start=1):
        status = str(section.get("status") or "")
        label = "does not apply" if status == "missing" and section.get("applies") is False else status
        said = f'<b>{_e(section.get("title") or section.get("key"))}</b><br><span class="km-muted">{plain(section.get("summary"))}</span>'
        sections.append([str(number), said, _badge(label, tones.get(status, ""))])
    if sections:
        out.append(_table([("", "num mono muted"), ("Section", "wrap"), ("Status", "")], sections))
    gaps = _dicts(data.get("gaps"))
    if gaps:
        listed = "".join(f"<li><b>{_e(gap.get('title'))}.</b> {plain(gap.get('how'))}</li>" for gap in gaps)
        out.append(_details(f"What is not on file ({len(gaps)})", f"<ul>{listed}</ul>", open_=True))
    link = _link(f"/models/{model['slug']}/summary", "Open in dashboard") if model.get("slug") else ""
    out.append(_foot("written from what is stored; nothing was run", link))
    return "".join(out)


# --- what a model's log says: monitoring, population, comparison, similar cases, holdings, methods ---
#
# Each leads with the sentence a person would say about the record, then the few numbers behind it, one
# small picture made of plain elements, and where to look next. A status is always written out.

#: The dashboard's ``VERDICT_LABEL``: what a window's drift check amounts to.
READING_LABEL = {
    "none": "Not enough rows",
    "chance": "Within chance",
    "stable": "Stable",
    "moderate": "Moderate drift",
    "significant": "Significant drift",
}
READING_TONE = {"stable": "ok", "moderate": "warn", "significant": "bad"}
#: The dashboard's ``runsLabel``: who runs a model.
RUNS_LABEL = {"kaimei": "Run by Kaimei", "your_endpoint": "Run by your endpoint", "your_agent": "Run by your agent", "nobody": "Run by nobody"}
_PLURAL = {"analysis": "analyses"}
#: Classes drawn in a colour of their own in a split bar; the rest are drawn together as "other".
_MIX = 3


def _plain(text: Any) -> str:
    """A sentence from the API as plain words: it sets the names of inputs and classes in backticks."""
    return _e(str(text or "").replace("`", ""))


def _say(text: Any) -> str:
    """The sentence a card leads with."""
    said = _plain(text).strip()
    return f'<div class="km-say">{said}</div>' if said else ""


def _items(value: Any) -> List[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _whole(value: Any) -> Optional[int]:
    number = _float(value)
    return int(number) if number is not None else None


def _counted(count: Any, noun: str) -> str:
    """``1 row``, ``600 rows``: a count with its noun."""
    n = _whole(count) or 0
    return f"{n:,} {noun if n == 1 else _PLURAL.get(noun, noun + 's')}"


def _size(value: Any) -> str:
    """``sizeText``: a size in plain units, base 1,000 (``734 bytes``, ``154 KB``, ``1.2 MB``); empty when not measured."""
    number = _float(value)
    if number is None or number < 0:
        return ""
    if number < 1000:
        return f"{int(number + 0.5)} {'byte' if int(number + 0.5) == 1 else 'bytes'}"
    units = ("KB", "MB", "GB", "TB")
    number /= 1000
    unit = 0
    while number >= 999.5 and unit < len(units) - 1:
        number /= 1000
        unit += 1
    return f"{int(number + 0.5) if number >= 100 else fmt_number(number, 1 if number >= 10 else 2)} {units[unit]}"


def _psi(value: Any) -> str:
    """``psiText``: a PSI to three decimals, to two from 1 up."""
    number = _float(value)
    if number is None:
        return "—"
    if 0 < number < 0.0005:
        return "under 0.001"
    return f"{number:.2f}" if number >= 1 else f"{number:.3f}"


def _day(value: Any) -> str:
    """A window's day: ``Nov 27``."""
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return str(value or "")
    return f"{moment:%b} {moment.day}"


def _model_link(ref: Any, tab: str = "", query: str = "") -> str:
    """A link to one of a model's pages in the dashboard, which takes a model's id where its slug goes."""
    return _link(f"/models/{quote(str(ref), safe='')}{tab}{query}", "Open in dashboard") if ref else ""


def _bar(value: float, largest: float, fill: str = "share", *, signed: bool = False) -> str:
    """One bar for a table cell: from zero, or either side of a midline when the column holds both signs."""
    size = min(abs(value) / largest, 1.0) * (50 if signed else 100) if largest > 0 else 0.0
    if signed:
        left = 50.0 if value >= 0 else 50.0 - size
        return (
            f'<div class="km-track cell"><span class="km-zero"></span><span class="km-fill {"pos" if value >= 0 else "neg"}" '
            f'style="left:{left:.2f}%;width:{size:.2f}%"></span></div>'
        )
    return f'<div class="km-track cell"><span class="km-fill {_e(fill)}" style="left:0;width:{size:.2f}%"></span></div>'


def _legend(items: Iterable[Tuple[str, str]], note: str = "") -> str:
    """Colour chips with what each stands for: ``items`` are ``(css class, words)``."""
    chips = "".join(f'<span><i class="{_e(cls)}"></i>{_e(words)}</span>' for cls, words in items)
    chips += f"<span>{_e(note)}</span>" if note else ""
    return f'<div class="km-legend">{chips}</div>' if chips else ""


def _reading(entry: Mapping[str, Any]) -> str:
    """``verdictOf``: a score past a threshold counts only when it is also past what sampling alone gives."""
    status = str(entry.get("status") or "")
    if status == "stable":
        return "stable"
    if status in ("moderate", "significant"):
        return "chance" if entry.get("beyond_chance") is False else status
    return "none"


def _monitoring_sentence(data: Mapping[str, Any], latest: Mapping[str, Any]) -> str:
    """What the latest window's check found: its reading, and the leading input's PSI against its own chance level."""
    logged, days = _whole(data.get("predictions")), _whole(data.get("days"))
    span = f" in the last {_counted(days, 'day')}" if days else ""
    if data.get("drift_unavailable"):
        traffic = f"{_counted(logged, 'prediction')} logged{span}. " if logged is not None else ""
        return f"{traffic}{data['drift_unavailable']}"
    if not latest:
        return f"No predictions were logged{span}." if logged == 0 else "This history has no windows to read."
    reading = _reading(latest)
    rows = _whole(latest.get("rows")) or 0
    if reading == "none" or _float(latest.get("overall_psi")) is None:
        return (
            "Not enough rows for a drift reading in the latest window: a check compares at least 20 logged rows with at "
            f"least 20 reference rows, and this one had {rows:,} and {_whole(data.get('reference_rows')) or 0:,}."
        )
    psi, chance = _psi(latest.get("overall_psi")), _float(latest.get("chance"))
    top = str(latest.get("top_feature") or "the leading input")
    drawn = f"{_counted(rows, 'row')} drawn from the reference data itself reach {_psi(chance)}"
    if reading == "chance":
        but = drawn if chance is not None else "as many rows drawn from the reference data itself score as high"
        return f"Within chance in the latest window: {top} scores PSI {psi}, past a drift threshold, but {but}."
    where = f", where {drawn}." if chance is not None else "."
    if reading == "stable":
        return f"Stable in the latest window: the highest PSI is {psi}, on {top}{where}"
    return f"{READING_LABEL[reading]} in the latest window: {top} scores PSI {psi}{where}"


def _monitoring(data: Mapping[str, Any]) -> str:
    """A monitoring history: the latest window's reading, the windows as bars, and the inputs by peak PSI."""
    series = _dicts(data.get("series"))
    latest: Mapping[str, Any] = series[-1] if series else {}
    reading = _reading(latest) if _float(latest.get("overall_psi")) is not None else "none"
    unavailable = bool(data.get("drift_unavailable"))
    window = str(data.get("window") or "window")
    days = _whole(data.get("days"))
    rows, reference = _whole(latest.get("rows")), _whole(data.get("reference_rows"))

    out = [_head("Monitoring", data.get("model_id"), _when(data, "read ", key="until"))]
    out.append(_say(_monitoring_sentence(data, latest)))
    badges = [_badge("No drift reading" if unavailable else READING_LABEL[reading], "" if unavailable else READING_TONE.get(reading, ""), dot=True)]
    if latest.get("start"):
        badges.append(_badge(f"latest {window}: {short_date(latest['start']) if window == 'hour' else _day(latest['start'])}", mono=True))
    if rows and reference and not unavailable:
        badges.append(
            _badge(f"{compact(reference)} reference vs {compact(rows)} logged rows", mono=True, title="What the latest window's check compares")
        )
    out.append(_badges(badges))

    tiles: List[Tuple[str, str, str]] = []
    if data.get("predictions") is not None:
        tiles.append(("Predictions", compact(data["predictions"]), f"last {_counted(days, 'day')}" if days else ""))
    if not unavailable and _float(latest.get("overall_psi")) is not None:
        tiles.append(("Latest PSI", _e(_psi(latest["overall_psi"])), f"on {latest['top_feature']}" if latest.get("top_feature") else ""))
        if _float(latest.get("chance")) is not None:
            tiles.append(("Chance level", _e(_psi(latest["chance"])), f"what {_counted(rows, 'reference row')} reach"))
    described = _map(_map(data.get("tape")).get("output"))
    answered = next((w for w in reversed(series) if _float(_map(w.get("output")).get("mean")) is not None), None)
    if answered is not None:
        mean = _map(answered.get("output")).get("mean")
        unit = described.get("unit") if isinstance(described.get("unit"), str) else None
        shown = fmt_value(mean, unit) if described.get("kind") == "value" else fmt_number(mean, 3)
        tiles.append(("Mean output", _e(shown), f"{described.get('name') or 'output'} · {_day(answered.get('start'))}"))
    out.append(_tiles(tiles))

    out.append(_window_bars(series, _map(data.get("thresholds")), window))
    if not unavailable:
        out.append(_monitoring_inputs(_dicts(data.get("features")), latest, _map(data.get("thresholds"))))
    out.append(_foot("read from the log; nothing was run or stored", _model_link(data.get("model_id"), "/monitoring")))
    return "".join(out)


def _window_bars(series: Sequence[Mapping[str, Any]], thresholds: Mapping[str, Any], window: str) -> str:
    """One small bar per window, oldest first: the highest PSI its check found, toned by what the check amounts to,
    with a line at each threshold. A history without a drift reading draws the predictions logged instead."""
    if not series:
        return ""

    def when(entry: Mapping[str, Any]) -> str:
        return short_date(entry.get("start")) if window == "hour" else _day(entry.get("start") or "")

    psis = [_float(entry.get("overall_psi")) for entry in series]
    known = [psi for psi in psis if psi is not None]
    bars: List[str] = []
    lines = ""
    if known:
        marks = sorted(v for v in (_float(x) for x in thresholds.values()) if v is not None and v > 0)
        top = max(known + [marks[-1] * 1.25 if marks else 0.0, 1e-9])
        counts: Dict[str, int] = {}
        for entry, psi in zip(series, psis):
            reading = _reading(entry) if psi is not None else "none"
            counts[reading] = counts.get(reading, 0) + 1
            said = f"{when(entry)}: {READING_LABEL[reading]}"
            if psi is not None:
                said += f", PSI {_psi(psi)}" + (f" on {entry['top_feature']}" if entry.get("top_feature") else "")
                said += f", chance reaches {_psi(entry['chance'])}" if _float(entry.get("chance")) is not None else ""
            if _whole(entry.get("predictions")) is not None:
                said += f", {_counted(entry['predictions'], 'prediction')}"
            height = f"{max(psi / top * 100, 3):.1f}%" if psi is not None else "2px"
            bars.append(f'<i class="km-win-bar km-tone-{reading}" style="height:{height}" title="{_e(said)}"></i>')
        lines = "".join(f'<span class="km-win-line" style="bottom:{mark / top * 100:.1f}%"></span>' for mark in marks if mark <= top)
        words = {"stable": "stable", "chance": "within chance", "moderate": "moderate drift", "significant": "significant drift", "none": "not enough rows"}
        legend = _legend((f"km-tone-{r}", f"{words[r]} · {counts[r]}") for r in ("stable", "chance", "moderate", "significant", "none") if counts.get(r))
        title = f"Highest PSI per {window}"
        middle = " and ".join(fmt_number(mark, 3) for mark in marks)
        middle = f"dashed lines at {middle} · tallest bar PSI {_psi(max(known))}" if marks else f"tallest bar PSI {_psi(max(known))}"
    else:
        logged = [_float(entry.get("predictions")) or 0.0 for entry in series]
        most = max(logged)
        if most <= 0:
            return ""
        for entry, count in zip(series, logged):
            height = f"{max(count / most * 100, 3):.1f}%" if count > 0 else "2px"
            said = f"{when(entry)}: {_counted(count, 'prediction')}"
            bars.append(f'<i class="km-win-bar km-tone-{"traffic" if count > 0 else "none"}" style="height:{height}" title="{_e(said)}"></i>')
        legend = ""
        title = f"Predictions per {window}"
        middle = f"busiest {window}: {_counted(most, 'prediction')}"
    dense = " dense" if len(series) > 150 else ""
    return (
        f'<div class="km-h">{_e(title)}</div>{legend}<div class="km-win{dense}">{lines}{"".join(bars)}</div>'
        f'<div class="km-axis"><span>{_e(when(series[0]))}</span><span>{_e(middle)}</span><span>{_e(when(series[-1]))}</span></div>'
    )


def _monitoring_inputs(features: Sequence[Mapping[str, Any]], latest: Mapping[str, Any], thresholds: Mapping[str, Any]) -> str:
    """The inputs ranked by their peak PSI, with each one's latest PSI and what it amounts to beside its own chance level."""
    features = [f for f in features if f.get("name")]
    if not features:
        return ""
    features = sorted(features, key=lambda f: (-(_float(f.get("peak_psi")) or 0.0), -(_float(f.get("latest_psi")) or 0.0)))
    moderate = _float(thresholds.get("moderate")) or 0.1
    significant = _float(thresholds.get("significant")) or 0.25
    levels = _map(latest.get("levels"))
    largest = max([_float(f.get("peak_psi")) or 0.0 for f in features])
    rows = []
    for feature in features[:8]:
        name = str(feature["name"])
        peak, now = _float(feature.get("peak_psi")), _float(feature.get("latest_psi"))
        status = str(feature.get("status") or "")
        level = _float(levels.get(name))
        if now is None or status not in ("stable", "moderate", "significant"):
            reading = "none"
        elif status != "stable" and level is not None and now <= level:
            reading = "chance"
        else:
            reading = status
        fill = "stable" if (peak or 0.0) < moderate else "moderate" if (peak or 0.0) < significant else "significant"
        unit = f' <span class="km-faint">{_e(feature["unit"])}</span>' if feature.get("unit") else ""
        rows.append(
            [
                f"<b>{_e(name)}</b>{unit}",
                _e(_psi(peak)) if peak is not None else '<span class="km-faint" title="It never scored above its own chance level">—</span>',
                _bar(peak, largest, fill) if peak else "",
                _e(_psi(now)),
                _badge(READING_LABEL[reading], READING_TONE.get(reading, "")),
            ]
        )
    more = f" {len(features) - 8} more in monitoring.features." if len(features) > 8 else ""
    note = (
        '<div class="km-faint" style="margin-top:6px">Peak PSI is an input\'s highest in the windows where it scored above its own '
        f"chance level; — means it never did.{_e(more)}</div>"
    )
    table = _table([("Input", "mono"), ("Peak PSI", "num mono"), ("", "bar"), ("Latest PSI", "num mono"), ("Latest reading", "")], rows)
    return f'<div class="km-h">Inputs by peak PSI</div>{table}{note}'


def _split_bar(counts: Mapping[str, Any], classes: Sequence[str]) -> str:
    """A group's class mix as one bar split by class, each class in the colour the legend gives it."""
    named = list(classes[:_MIX])
    parts = [(f"km-mix-{index}", _float(counts.get(name)) or 0.0) for index, name in enumerate(named)]
    other = sum(_float(value) or 0.0 for name, value in counts.items() if str(name) not in named)
    parts.append(("km-mix-x", other))
    said = ", ".join(f"{name}: {compact(value)}" for name, value in counts.items())
    segments = "".join(f'<i class="{cls}" style="flex:{amount:g} 1 0%"></i>' for cls, amount in parts if amount > 0)
    return f'<div class="km-split" title="{_e(said)}">{segments}</div>'


def _mix_legend(classes: Sequence[str]) -> str:
    chips = [(f"km-mix-{index}", str(name)) for index, name in enumerate(classes[:_MIX])]
    return _legend(chips + ([("km-mix-x", "other")] if len(classes) > _MIX else []))


def _population(data: Mapping[str, Any]) -> str:
    """A population: how many predictions, the field that separates the answers most, and that field's groups."""
    output = _map(data.get("output"))
    classes = [str(c) for c in _items(output.get("classes"))]
    fields = _dicts(data.get("fields"))
    ranked = sorted((f for f in fields if _float(f.get("separation")) is not None), key=lambda f: -(_float(f.get("separation")) or 0.0))
    top = ranked[0] if ranked and (_float(ranked[0].get("separation")) or 0.0) > 0 else None
    total, shown, days = _whole(data.get("predictions")), _whole(data.get("shown")), _whole(data.get("days"))
    values = [v for v in (_float(x) for x in _items(_map(data.get("rows")).get("value"))) if v is not None]
    if shown is None:
        shown = len(_items(_map(data.get("rows")).get("id"))) or total
    span = f" in the last {_counted(days, 'day')}" if days else ""

    def share(value: Any) -> str:
        return pct(value, 1 if (_float(value) or 0.0) < 0.1 else 0)

    def called(field: Mapping[str, Any]) -> str:
        return str(field.get("name") or field.get("key") or "an unnamed field")

    if not (total or shown):
        sentence = f"No predictions were logged{span}."
    else:
        if total is not None and shown is not None and shown < total:
            counted = f"the newest {shown:,} of {_counted(total, 'prediction')} logged{span}"
        else:
            counted = f"{_counted(total if total is not None else shown, 'prediction')} logged{span}"
        if top is not None:
            sentence = (
                f"Of {counted}, {called(top)} separates the answers most: "
                f"{share(top.get('separation'))} of their variation lies between its groups."
            )
        elif len(ranked) == 1:
            sentence = f"Of {counted}, the one field ranked by how far it separates the answers ({called(ranked[0])}) accounts for none of their variation."
        elif ranked:
            sentence = f"Of {counted}, no field separates the answers: none of the {len(ranked)} ranked for it accounts for any of their variation."
        else:
            sentence = f"{counted[:1].upper()}{counted[1:]}, with no field that ranks how they group."

    out = [_head("Population", data.get("model_id"), _when(data, "read ", key="until"))]
    out.append(_say(sentence))
    name = str(output.get("name") or "answer")
    unit = output.get("unit") if isinstance(output.get("unit"), str) else None

    def answer(value: Any) -> str:
        return _e(fmt_value(value, unit) if output.get("kind") == "value" else fmt_number(value, 3))

    tiles: List[Tuple[str, str, str]] = []
    if total is not None or shown is not None:
        tiles.append(("Predictions", compact(total if total is not None else shown), f"last {_counted(days, 'day')}" if days else ""))
    if values:
        tiles.append(("Mean answer", answer(sum(values) / len(values)), name))
    if data.get("outcomes") is not None:
        tiles.append(("Recorded outcomes", compact(data["outcomes"]), f"of {shown:,} shown" if shown else ""))
    if fields:
        tiles.append(("Fields", str(len(fields)), "they can be grouped by"))
    out.append(_tiles(tiles))

    field = top or next((f for f in fields if f.get("key") == "prediction"), None) or (fields[0] if fields else None)
    groups = _dicts(field.get("groups"))[:12] if field is not None else []
    if field is not None and groups:
        kind = ", ".join(str(part) for part in (field.get("source"), field.get("unit")) if part)
        said = f"{share(field.get('separation'))} of the variation in the answers" if _float(field.get("separation")) is not None else "the answers themselves"
        out.append(
            f'<div class="km-h">By {_e(called(field))} '
            f'<span class="km-faint">{_e(kind)}{" · " if kind else ""}{_e(said)}</span></div>'
        )
        mixed = bool(classes) and any(_map(g.get("predicted")) for g in groups)
        means = [_float(g.get("mean")) for g in groups]
        largest = max([abs(m) for m in means if m is not None] + [0.0])
        signed = any(m is not None and m < 0 for m in means)
        rows = []
        for group, mean in zip(groups, means):
            cells = [f"<b>{_e(group['label'] if group.get('label') is not None else '—')}</b>", _e(compact(group.get("rows")))]
            if mixed:
                counts = {str(k): _float(v) or 0.0 for k, v in _map(group.get("predicted")).items()}
                whole = sum(counts.values())
                first = max(counts, key=lambda k: counts[k]) if whole > 0 else None
                cells.append(_split_bar(counts, classes) if whole > 0 else "")
                cells.append(f"{pct(counts[first] / whole)} {_e(first)}" if first is not None else "—")
                cells.append(answer(mean) if mean is not None else "—")
            else:
                cells.append(_bar(mean, largest, signed=signed) if mean is not None else "")
                cells.append(answer(mean) if mean is not None else "—")
            rows.append(cells)
        if mixed:
            out.append(_mix_legend(classes))
            columns = [("Group", "mono"), ("Rows", "num mono"), ("Predicted", "bar"), ("Most predicted", "muted"), (f"Mean {name}", "num mono")]
        else:
            columns = [("Group", "mono"), ("Rows", "num mono"), (f"Mean {name}", "bar"), ("", "num mono")]
        out.append(_table(columns, rows))
    then = [f for f in ranked if f is not field and (_float(f.get("separation")) or 0.0) > 0][:4]
    if then:
        chips = [_badge(f"{called(f)} {share(f.get('separation'))}", mono=True) for f in then]
        out.append('<div class="km-h">Then</div>' + _badges(chips))
    out.append(_foot("read from the log; nothing was run", _model_link(data.get("model_id"), "/population")))
    return "".join(out)


def _comparison(data: Mapping[str, Any]) -> str:
    """Two models on the same rows: how far they agree, how each scores, and the rows they differ on most."""
    models = (_dicts(data.get("models")) + [{}, {}])[:2]
    names = [str(m.get("name") or m.get("slug") or unnamed) for m, unnamed in zip(models, ("The first model", "the second model"))]
    slugs = [str(m.get("slug") or m.get("id") or unnamed) for m, unnamed in zip(models, ("first", "second"))]
    output = _map(data.get("output"))
    classes = [str(c) for c in _items(output.get("classes"))]
    agreement = _map(data.get("agreement"))
    n = _whole(data.get("compared"))
    if n is None:
        n = _whole(agreement.get("rows"))
    noun = "logged prediction" if data.get("source") == "predictions" else "reference row"
    unit = output.get("unit") if isinstance(output.get("unit"), str) else None
    quantity = str(output.get("name") or "answer")
    rate, same = _float(agreement.get("rate")), _whole(agreement.get("same"))
    correlation, mean_gap = _float(agreement.get("correlation")), _float(agreement.get("mean_gap"))
    over = f" over {_counted(n, noun)}" if n else ""

    def gap(value: Any) -> str:
        return fmt_value(value, unit) if output.get("kind") == "value" else fmt_number(value, 3)

    if rate is not None and same is not None and n:
        sentence = f"{names[0]} and {names[1]} decide {same:,} of {_counted(n, noun)} the same way ({pct(rate, 1)})."
    elif rate is not None:
        sentence = f"{names[0]} and {names[1]} decide {pct(rate, 1)} of the rows the same way{over}."
    elif correlation is not None:
        apart = f", {gap(mean_gap)} apart on average" if mean_gap is not None else ""
        sentence = f"The answers of {names[0]} and {names[1]} correlate at {correlation:.2f}{over}{apart}."
    elif mean_gap is not None:
        sentence = f"The answers of {names[0]} and {names[1]} are {gap(mean_gap)} apart on average{over}."
    else:
        sentence = f"{names[0]} and {names[1]} were compared{over}."

    out = [_head("Comparison", f"{slugs[0]} vs {slugs[1]}" if any(models) else None, _e(_counted(n, noun)) if n else "")]
    out.append(_say(sentence))
    badges = [_badge(f"answer: {quantity}" + (f" ({unit})" if unit else ""), "info")] if output.get("name") else []
    if data.get("truncated"):
        badges.append(_badge("fewer rows than the source holds", title="Past the limit, or past the time one comparison may run for"))
    if _whole(data.get("skipped")):
        badges.append(_badge(f"{_counted(data['skipped'], 'row')} left out", title="One of the two could not read them, or gave no number for them"))
    out.append(_badges(badges))
    tiles: List[Tuple[str, str, str]] = []
    if rate is not None:
        tiles.append(("Agreement", pct(rate, 1), f"{same:,} of {n:,} rows" if same is not None and n else "of the rows"))
    if correlation is not None:
        tiles.append(("Correlation", fmt_number(correlation, 3), "of the two answers"))
    if mean_gap is not None:
        tiles.append(("Mean gap", _e(gap(mean_gap)), f"in {quantity}"))
    if _float(agreement.get("max_gap")) is not None:
        tiles.append(("Largest gap", _e(gap(agreement["max_gap"])), f"in {quantity}"))
    out.append(_tiles(tiles))

    scores = (_dicts(data.get("scores")) + [{}, {}])[:2]
    metrics = [key for key in ("accuracy", "auc", "mae", "r2") if any(key in score for score in scores)]
    if metrics:
        titles = {"accuracy": "Accuracy", "auc": "AUC", "mae": "MAE", "r2": "R²"}
        rows = []
        for index, (model, score) in enumerate(zip(models, scores)):
            decides = ""
            if _float(model.get("threshold")) is not None and model.get("positive") is not None:
                decides = f"{model['positive']} from {fmt_number(model['threshold'], 3)}"
            row = [
                f'<i class="km-pin km-mix-{index}" style="position:static;display:inline-block;margin:0 6px -2px 0"></i>'
                f"<b>{_e(names[index])}</b> <span class=\"km-faint km-mono\">{_e(slugs[index])}</span>"
            ]
            for key in metrics:
                value = _float(score.get(key))
                row.append("—" if value is None else pct(value, 1) if key == "accuracy" else _e(gap(value)) if key == "mae" else f"{value:.3f}")
            row.append(_e(compact(score.get("n"))) if score.get("n") is not None else "—")
            row.append(_e(decides))
            rows.append(row)
        columns = [("Model", "")] + [(titles[key], "num mono") for key in metrics] + [("Rows with truth", "num mono"), ("Decides", "muted")]
        out.append('<div class="km-h">How each scores <span class="km-faint">on the rows whose truth is known</span></div>')
        out.append(_table(columns, rows))
    out.append(_comparison_rows(data, slugs, classes, gap))
    link = _model_link(models[0].get("slug"), "/compare", f"?with={quote(str(models[1]['slug']), safe='')}" if models[1].get("slug") else "")
    runs = f"{_counted(data['runs'], 'model run')}; " if _whole(data.get("runs")) else ""
    out.append(_foot(f"{_e(runs)}both models were run and nothing was stored", link))
    return "".join(out)


def _comparison_rows(data: Mapping[str, Any], slugs: Sequence[str], classes: Sequence[str], gap: Callable[[Any], str]) -> str:
    """The three rows the two models are furthest apart on: both answers, drawn on one line, and what really happened."""
    columns = _map(data.get("rows"))
    values = [_items(side) for side in _items(columns.get("values"))]
    if len(values) != 2:
        return ""
    decisions = [_items(side) for side in _items(columns.get("decisions"))]
    ids, actual = _items(columns.get("id")), _items(columns.get("actual"))
    output = _map(data.get("output"))

    def at(column: Sequence[Any], index: int) -> Any:
        return column[index] if 0 <= index < len(column) else None

    def named(position: Any) -> Optional[str]:
        if position is None:
            return None
        if classes and isinstance(position, int) and not isinstance(position, bool):
            return classes[position] if 0 <= position < len(classes) else None
        return gap(position) if _float(position) is not None else str(position)

    places = [p for p in _items(data.get("differ_most")) if isinstance(p, int) and not isinstance(p, bool)]
    places = [p for p in places if _float(at(values[0], p)) is not None and _float(at(values[1], p)) is not None][:3]
    if not places:
        return ""
    if output.get("kind") == "probability":
        low, high = 0.0, 1.0
    else:
        every = [v for side in values for v in (_float(x) for x in side) if v is not None]
        low, high = min(every), max(every)
    span = (high - low) or 1.0

    def left(value: float) -> float:
        return min(max((value - low) / span * 100, 0.0), 100.0)

    threshold = _float(output.get("threshold"))
    mark = f'<span class="km-zero" style="left:{left(threshold):.1f}%"></span>' if threshold is not None and low <= threshold <= high else ""
    rows = []
    for place in places:
        first, second = float(at(values[0], place)), float(at(values[1], place))
        chose = [named(at(side, place)) for side in decisions] if len(decisions) == 2 else [None, None]
        cells = [_e(at(ids, place) if at(ids, place) is not None else place)]
        for value, choice in zip((first, second), chose):
            cells.append(_e(gap(value)) + (f' <span class="km-muted">{_e(choice)}</span>' if choice is not None else ""))
        a, b = left(first), left(second)
        cells.append(
            f'<div class="km-gap" title="{_e(slugs[0])}: {_e(gap(first))}, {_e(slugs[1])}: {_e(gap(second))}">'
            f'<span class="km-gap-track"></span>{mark}<span class="km-gap-line" style="left:{min(a, b):.1f}%;width:{abs(b - a):.1f}%"></span>'
            f'<i class="km-pin km-mix-0" style="left:{a:.1f}%"></i><i class="km-pin km-mix-1" style="left:{b:.1f}%"></i></div>'
        )
        cells.append(_e(gap(abs(second - first))))
        truth = named(at(actual, place))
        cells.append(_e(truth) if truth is not None else "—")
        rows.append(cells)
    quantity = str(output.get("name") or "answer")
    note = f"drawn from {gap(low)} to {gap(high)}" + (f", with a line at the threshold {fmt_number(threshold, 3)}" if mark else "")
    table = _table(
        [("Row", "mono"), (slugs[0], "num mono fold"), (slugs[1], "num mono fold"), (quantity, "bar"), ("Gap", "num mono"), ("Actual", "muted")],
        rows,
    )
    legend = _legend([("km-mix-0", slugs[0]), ("km-mix-1", slugs[1])], note)
    return f'<div class="km-h">Where they differ most <span class="km-faint">the {len(rows)} rows furthest apart</span></div>{legend}{table}'


def _similar(data: Mapping[str, Any]) -> str:
    """Similar cases: the summary sentence, then each case with how near it is, what is known of it and where it differs."""
    cases = _dicts(data.get("cases"))
    summary = _map(data.get("summary"))
    reference = str(data.get("source") or "reference") == "reference"
    noun = "reference row" if reference else "logged prediction"
    considered = _whole(data.get("considered"))
    right = f"{len(cases)} nearest of {_counted(considered, noun)}" if considered is not None and cases else ""
    out = [_head("Similar cases", data.get("model_id"), _e(right))]
    sentence = summary.get("sentence")
    if not sentence:
        sentence = f"The {_counted(len(cases), noun)} most like this case." if cases else f"No {noun}s came back as like this case."
    out.append(_say(sentence))

    def counted(key: str, prefix: str) -> List[str]:
        part = _map(summary.get(key))
        counts = _map(part.get("counts"))
        if counts:
            return [_badge(f"{prefix} {name} · {compact(count)}", mono=True) for name, count in list(counts.items())[:4]]
        if _float(part.get("mean")) is not None:
            return [_badge(f"{prefix} mean {fmt_number(part['mean'], 3)} ({fmt_number(part.get('min'), 3)} to {fmt_number(part.get('max'), 3)})", mono=True)]
        return []

    badges = counted("labels", "labelled") + counted("predicted", "predicted") + counted("outcomes", "outcome")
    nearest = _float(cases[0].get("similarity")) if cases else None
    if nearest is not None:
        badges.append(_badge(f"nearest {pct(nearest, 1)} alike", mono=True, title="Similarity: one less the distance, the mean gap over the inputs"))
    out.append(_badges(badges))
    asked = _map(data.get("input"))
    if data.get("prediction_id"):
        out.append(f'<div class="km-desc">Asked about prediction <span class="km-mono">{_e(data["prediction_id"])}</span>.</div>')
    elif asked:
        listed = " · ".join(f"{_e(k)} {_e(_input_text(v))}" for k, v in list(asked.items())[:10])
        more = f" · and {len(asked) - 10} more" if len(asked) > 10 else ""
        out.append(f'<div class="km-desc">Asked about <span class="km-mono">{listed}</span>{_e(more)}.</div>')

    if cases:
        rows = []
        for case in cases[:12]:
            similarity = _float(case.get("similarity"))
            cells = [_e(case.get("id") if case.get("id") is not None else "—")]
            width = f"{min(max(similarity, 0.0), 1.0) * 100:.1f}" if similarity is not None else "0"
            cells.append(f'<div class="km-meter" style="min-width:56px"><i style="width:{width}%"></i></div>')
            cells.append(pct(similarity, 1))
            if reference:
                cells.append(f"<b>{_e(_input_text(case['label']))}</b>" if case.get("label") is not None else "—")
            else:
                # A logged prediction is known by more, and by a long id: two lines a cell, so the table stays narrow.
                if case.get("time"):
                    cells[0] += f'<span class="km-note">{_date_cell(case["time"])}</span>'
                value = f' <span class="km-faint">{fmt_number(case["value"], 3)}</span>' if _float(case.get("value")) is not None else ""
                said = (f"<b>{_e(case['predicted'])}</b>" if case.get("predicted") is not None else "") + value or "—"
                outcome = f"outcome {_e(_input_text(case['outcome']))}" if case.get("outcome") is not None else "outcome not recorded"
                cells.append(f'{said}<span class="km-note">{outcome}</span>')
            differs = [
                f'<b>{_e(d.get("feature"))}</b> {_e(_input_text(d.get("value")))} <span class="km-faint">vs {_e(_input_text(d.get("asked")))}</span>'
                for d in _dicts(case.get("differs"))[:3]
                if d.get("feature") is not None
            ]
            same = '<span class="km-faint">the same on every input</span>' if _float(case.get("distance")) == 0 else "—"
            cells.append(" · ".join(differs) if differs else same)
            rows.append(cells)
        columns = [("Row" if reference else "Prediction", "mono"), ("Similarity", ""), ("", "num mono")]
        columns += [("Label", "")] if reference else [("Predicted", "")]
        columns.append(("Differs most on (its value vs the asked one)", "said"))
        out.append(_table(columns, rows))
        if len(cases) > 12:
            out.append(f'<div class="km-faint" style="margin-top:6px">{len(cases) - 12} more in similar.cases; similar.to_frame() has them all.</div>')
    tab = "/predictions" if data.get("prediction_id") else "/explain"
    out.append(_foot("compared input by input with what Kaimei holds; nothing was run", _model_link(data.get("model_id"), tab)))
    return "".join(out)


def _holdings(data: Mapping[str, Any]) -> str:
    """What Kaimei holds for a model: who runs it, then what stays with you beside what Kaimei holds."""
    runs = _map(data.get("runs"))
    items = [item for item in _dicts(data.get("items")) if item.get("label") or item.get("key")]
    yours = [item for item in items if item.get("held") == "no"]
    held = [item for item in items if item.get("held") in ("yes", "partly")]
    # An item that does not say is put on neither side: the card claims only what the record does.
    unsaid = [item for item in items if item not in yours and item not in held]
    out = [_head("What Kaimei holds", data.get("model_id"), _when(data, "read ", key="as_of"))]
    out.append(_say(runs.get("sentence") or (f"Kaimei holds {len(held)} of the {len(items)} things it could hold for this model." if items else "Nothing is listed for this model.")))
    badges = [_badge(RUNS_LABEL.get(str(runs.get("by")), title_case(runs.get("by"))), "info")] if runs.get("by") else []
    if items:
        badges.append(_badge(f"{len(held)} of {_counted(len(items), 'item')} held", mono=True))
    measured = [_float(item.get("bytes")) for item in held if _float(item.get("bytes")) is not None]
    if measured:
        badges.append(_badge(f"{_size(sum(measured))} in {_counted(len(measured), 'measured item')}", mono=True, title="What the database stores for them; the others are not measured"))
    out.append(_badges(badges))

    def listed(item: Mapping[str, Any]) -> str:
        count, unit = _whole(item.get("count")), item.get("unit")
        measure = [_counted(count, str(unit))] if count and unit else []
        if item.get("held") != "no" and _size(item.get("bytes")):
            measure.append(_size(item.get("bytes")))
        partly = item.get("held") == "partly"
        text = f"<b>{_e(item.get('label') or item.get('key'))}</b>"
        text += f'<span class="km-measure">{_e(" · ".join(measure))}</span>' if measure else ""
        text += f" {_badge('held in part', 'warn')}" if partly else ""
        # A whole holding is its count and size; one held in part, not at all, or without a measure needs its sentence.
        if item.get("detail") and (partly or item.get("held") == "no" or not measure):
            text += f'<span class="km-note">{_plain(item["detail"])}</span>'
        return f'<div class="km-item" title="{_plain(item.get("detail"))}">{text}</div>'

    if items:
        nothing_yours = '<div class="km-item"><span class="km-note">Nothing on this list: Kaimei holds at least part of every item.</span></div>'
        nothing_held = '<div class="km-item"><span class="km-note">Nothing on this list: Kaimei holds none of these.</span></div>'
        out.append(
            '<div class="km-sides">'
            f'<div class="km-side"><div class="km-side-h">Stays with you <span class="km-faint">Kaimei holds none of it</span></div>'
            f'{"".join(listed(item) for item in yours) or nothing_yours}</div>'
            f'<div class="km-side"><div class="km-side-h">Held by Kaimei <span class="km-faint">stored in Kaimei\'s database</span></div>'
            f'{"".join(listed(item) for item in held) or nothing_held}</div>'
            "</div>"
        )
    if unsaid:
        names = ", ".join(str(item.get("label") or item.get("key")) for item in unsaid)
        out.append(f'<div class="km-faint" style="margin-top:6px">The record does not say whether Kaimei holds: {_e(names)}.</div>')
    whens = [str(crossing.get("when")) for crossing in _dicts(data.get("crosses")) if crossing.get("when")]
    if whens:
        out.append(
            '<div class="km-faint" style="margin-top:6px">What travels between your network and Kaimei is in holdings.crosses, for '
            f"{_e(', '.join(whens))}.</div>"
        )
    out.append(_foot("read from what is stored; nothing was run", _model_link(data.get("model_id"), "/settings")))
    return "".join(out)


def _method_comparison(data: Mapping[str, Any]) -> str:
    """Several methods on one input: whether they agree, and each one's top three inputs on one scale with its cost."""
    explanations = _dicts(data.get("explanations"))
    agreement = _map(data.get("agreement"))
    costs = _dicts(data.get("costs"))
    methods = [str(m) for m in _items(data.get("methods"))] or [str(e.get("method")) for e in explanations]
    table = _dicts(data.get("rows"))
    first = explanations[0] if explanations else {}
    output = _map(first.get("output"))
    main, sub = output_parts(output)
    target = output.get("target_label")
    classification = isinstance(output.get("label"), str) or bool(output.get("probabilities"))

    out = [_head("Method comparison", data.get("model"), _e(_counted(len(methods), "method")) if methods else "")]
    sentence = agreement.get("sentence")
    if not sentence:
        sentence = f"{_counted(len(methods), 'method')} explained the same input." if len(methods) > 1 else "There is nothing to compare: fewer than two methods explained this input."
    out.append(_say(sentence))
    badges = []
    rank = _float(agreement.get("rank_correlation"))
    how = "Spearman's rank correlation of the inputs' absolute attributions, averaged over the pairs of methods: 1 is the same order"
    if rank is not None:
        badges.append(_badge(f"rank agreement {rank:.2f}", mono=True, title=how))
    elif "rank_correlation" in agreement:
        badges.append(_badge("rank agreement not defined", title="It needs two methods and three inputs they all rank"))
    if main != "—":
        # A class with its probability, or a number with its unit.
        tail = f" · {sub}" if sub and sub.endswith("%") else f" {sub}" if sub else ""
        badges.append(_badge(f"answer {main}{tail}", "info", title="The model's answer for this input"))
    if _whole(agreement.get("inputs")):
        badges.append(_badge(_counted(agreement["inputs"], "input"), mono=True))
    opposed = [str(name) for name in _items(agreement.get("opposed"))]
    if opposed:
        badges.append(_badge("pushed opposite ways: " + ", ".join(opposed[:3]), "warn"))
    out.append(_badges(badges))

    def top_three(index: int, method: str) -> List[Tuple[str, Optional[str], float]]:
        explanation = explanations[index] if index < len(explanations) else {}
        found = {str(k): _float(v) for k, v in _map(explanation.get("attributions")).items()}
        if not found:
            found = {str(f.get("feature")): _float(f.get("attribution")) for f in _dicts(explanation.get("top_features"))}
        if not found:
            found = {str(row.get("feature")): _float(_map(row.get("attributions")).get(method)) for row in table}
        ranked = sorted(((name, value) for name, value in found.items() if value is not None), key=lambda kv: -abs(kv[1]))[:3]
        inputs = _map(explanation.get("input"))
        return [(name, _bar_input(inputs.get(name)), value) for name, value in ranked]

    tops = [top_three(index, method) for index, method in enumerate(methods)]
    scale = max([abs(value) for rows in tops for _, _, value in rows] + [0.0])
    if any(tops):
        if classification and isinstance(target, str):
            legend = (f"pushes toward {target}", "pushes away")
        elif classification:
            legend = ("pushes toward the prediction", "pushes away")
        else:
            legend = ("pushes higher", "pushes lower")
        out.append('<div class="km-h">Each method\'s top three <span class="km-faint">bars on one scale</span></div>' + _diverging_legend(legend))
    for index, method in enumerate(methods):
        cost = next((c for c in costs if c.get("method") == method), costs[index] if index < len(costs) else {})
        chips = []
        if cost.get("exact") is True:
            chips.append(_badge("exact", "ok", title=str(cost.get("detail") or "")))
        elif cost.get("exact") is False:
            chips.append(_badge("approximate", title=str(cost.get("detail") or "")))
        if cost.get("n_evals") is not None:
            chips.append(_badge(f"{compact(cost['n_evals'])} model calls", mono=True))
        if _float(cost.get("latency_ms")) is not None:
            chips.append(_badge(f"{fmt_number(cost['latency_ms'], 0)} ms", mono=True, title="Server time"))
        detail = f'<span class="km-faint" style="font-weight:400">{_e(cost["detail"])}</span>' if cost.get("detail") else ""
        label = cost.get("label") or METHOD_LABEL.get(method, method)
        out.append(f'<div class="km-method"><span>{_e(label)}</span>{"".join(chips)}{detail}</div>')
        out.append(_diverging_bars(tops[index], scale=scale) if tops[index] else '<div class="km-faint">No attributions in this record.</div>')
    ids = [str(e.get("id")) for e in explanations if e.get("id")]
    left = f"{_counted(len(ids), 'explanation')} stored, one per method" if ids else ""
    out.append(_foot(_e(left), _link(f"/explanations/{quote(ids[0], safe='')}", "Open in dashboard") if ids else ""))
    return "".join(out)


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
        name = "score" if (data.get("output") or {}).get("uncalibrated") else "P"
        head_sub = f"{name}({_e(label)}) = {_e(sub)}" if classification and isinstance(label, str) and sub.endswith("%") else _e(sub)
    out.append(f'<div class="km-result"><span class="km-value">{_e(main)}</span>' + (f'<span class="km-sub">{head_sub}</span>' if head_sub else "") + "</div>")
    badges = [_badge(METHOD_LABEL.get(method, method or "attributions"), "ok")]
    if (data.get("output") or {}).get("uncalibrated"):
        badges.append(_badge("score, not a probability", "info", title="This model only ranks or votes"))
    if isinstance(target, str) and target != label:
        badges.append(_badge(f"explains {target}", "info"))
    if _float(data.get("base_value")) is not None:
        meaning = "The intercept of a local fit" if method == "lime" else "Average prediction over the rows it was compared with"
        badges.append(_badge(f"base {fmt_number(data['base_value'], 3)}", mono=True, title=meaning))
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
        name = "score" if (data.get("output") or {}).get("uncalibrated") else "P"
        quantity = f"{name}({target})" if classification and isinstance(target, str) else None
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


def _decision_record(data: Mapping[str, Any]) -> str:
    """A decision record. Every sentence is the API's own; the card only lays them out."""
    outcome = _map(data.get("outcome"))
    model = _map(data.get("model"))
    explanation = _map(data.get("explanation"))
    factors = [f for f in (data.get("factors") or []) if isinstance(f, Mapping)]
    decided = _e(long_date(data["decided_at"])) if data.get("decided_at") else ""

    out = [_head("Decision record", data.get("id"), decided)]
    kind = outcome.get("kind")
    main, sub = "", ""
    if kind == "number" and _float(outcome.get("value")) is not None:
        main = fmt_value(outcome.get("value"), outcome.get("unit") or None)
        sub = str(outcome.get("name") or "")
    elif outcome.get("label") is not None:
        main = str(outcome["label"])
        if kind == "binary" and _float(outcome.get("probability")) is not None:
            sub = f"P({outcome.get('positive_label')}) = {pct(outcome['probability'], 1)} · threshold {pct(outcome.get('threshold'), 0)}"
        elif _float(outcome.get("probability")) is not None:
            sub = f"P({outcome['label']}) = {pct(outcome['probability'], 1)}"
    if main:
        tail = f'<span class="km-sub">{_e(sub)}</span>' if sub else ""
        out.append(f'<div class="km-result"><span class="km-value">{_e(main)}</span>{tail}</div>')
    badges = []
    if model.get("name"):
        badges.append(_badge(f"{model['name']} · v{model.get('version')}", title="The model that made this decision"))
    if explanation:
        why = str(explanation.get("reproducible_sentence") or "")
        badges.append(_badge(explanation.get("method_name") or explanation.get("method") or "explained", "ok"))
        badges.append(_badge("reproducible", "ok", title=why) if explanation.get("reproducible") else _badge("not exactly reproducible", "warn", title=why))
    else:
        badges.append(_badge("not explained", "warn", title="client.decisions.explain(id) adds how the model reached it"))
    out.append(_badges(badges))
    if outcome.get("sentence"):
        out.append(f'<div class="km-lead" style="color:var(--text);font-size:14px">{_e(outcome["sentence"])}</div>')
    for note in data.get("notes") or []:
        out.append(f'<div class="km-lead">{_badge("note", "warn")} {_e(note)}</div>')

    view = _map(data.get("view"))
    if explanation:
        out.append('<div class="km-h">How the model reached it</div>')
        reference = _map(explanation.get("reference")).get("sentence")
        # A picture, a text or a series: the record's own account of what the model was looking at.
        for sentence in view.get("sentences") or []:
            out.append(f'<div class="km-lead" style="margin-bottom:6px">{_e(sentence)}</div>')
        if reference and not view:
            out.append(f'<div class="km-lead">{_e(reference)}</div>')
        if view.get("kind") == "text" and view.get("parts"):
            out.append(_token_strip([part for part in view["parts"] if isinstance(part, Mapping)]))
        elif view.get("kind") == "image":
            out.append(
                '<div class="km-faint" style="margin:0 0 10px">The picture, with the model\'s evidence drawn on it, is in the '
                "PDF (<span class=km-mono>client.decisions.save_pdf</span>) and on the record in the dashboard.</div>"
            )
        rows = [
            (str(f.get("label") or f.get("feature")), None if f.get("display") is None else str(f.get("display")), _float(f.get("attribution")) or 0.0)
            for f in factors
        ]
        legend = ("toward", "against") if explanation.get("measure") == "evidence" else ("raises it", "lowers it")
        out.append(_diverging_bars(rows[:14], legend=legend))
        if len(rows) > 14:
            out.append(_details(f"All {len(rows)} effects", _diverging_bars(rows[14:])))
        if explanation.get("summary"):
            out.append(f'<div class="km-lead" style="color:var(--text);margin-top:10px">{_e(explanation["summary"])}</div>')
        moved = [f for f in factors if f.get("direction") != "none" and f.get("sentence")]
        # An effect too small to print has a sentence that says so; a list in words leaves those out.
        stated = [f for f in moved if not f.get("negligible")] or moved
        if stated:
            lines = "".join(f'<div>{_e(f.get("rank"))}. {_e(f["sentence"])}</div>' for f in stated)
            out.append(_details("In words", f'<div class="km-cf">{lines}</div>', open_=len(stated) <= 6, note=f"{len(stated)} effects"))

    counterfactual = _map(data.get("counterfactual"))
    if counterfactual.get("sentence"):
        body = f'<div>{_e(counterfactual["sentence"])}</div><span class="km-faint">{_e(counterfactual.get("note") or "")}</span>'
        out.append(_details("What would have changed the outcome", f'<div class="km-cf">{body}</div>', open_=True))

    drafted = _map(data.get("reasons"))
    if drafted:
        cited = _map(drafted.get("table"))
        which = "the draft reason table" if cited.get("status") == "draft" else f"reason table version {cited.get('version')}"
        said = [f'<div>{_e(drafted.get("sentence") or "")}</div>']
        for number, item in enumerate([r for r in (drafted.get("reasons") or []) if isinstance(r, Mapping)], start=1):
            source = str(item.get("label") or item.get("feature") or "")
            if item.get("display") is not None:
                source += f" ({item.get('display')})"
            said.append(f'<div>{number}. <b>{_e(item.get("reason"))}</b> <span class="km-faint">from {_e(source)}</span></div>')
        for item in drafted.get("held") or []:
            if isinstance(item, Mapping):
                said.append(f'<div>{_badge("held for review", "warn")} {_e(item.get("sentence"))}</div>')
        for note in drafted.get("notes") or []:
            said.append(f'<div class="km-faint">{_e(note)}</div>')
        said.append(f'<span class="km-faint">In the wording of {_e(which)}. {_e(drafted.get("notice") or "")}</span>')
        out.append(_details("Draft reasons", f'<div class="km-cf">{"".join(said)}</div>', open_=bool(drafted.get("applies"))))

    policy = _map(data.get("policy"))
    if policy:
        said = [f"<div>{_e(sentence)}</div>" for sentence in policy.get("sentences") or []]
        for item in policy.get("reasons") or []:
            if isinstance(item, Mapping):
                said.append(
                    f'<div>{_badge("reason given by a rule")} <b>{_e(item.get("reason"))}</b> '
                    f'<span class="km-faint">{_e(item.get("rule"))}</span></div>'
                )
        if policy.get("note"):
            said.append(f"<div>Note: {_e(policy.get('note'))}</div>")
        by = _map(policy.get("recorded_by")).get("name")
        when = long_date(policy["recorded_at"]) if policy.get("recorded_at") else ""
        signed = f"Recorded {when}" + (f" by {by}" if by else "")
        if policy.get("amended_at"):
            signed += f"; revision {policy.get('revision')}"
        said.append(f'<span class="km-faint">{_e(signed)}. {_e(policy.get("stated") or "")}</span>')
        out.append(_details("What was decided", f'<div class="km-cf">{"".join(said)}</div>', open_=True))
    held = [h for h in (data.get("holds") or []) if isinstance(h, Mapping)]
    history = [h for h in (data.get("history") or []) if isinstance(h, Mapping)]
    if held or history:
        custody = []
        for hold in held:
            covers = "this decision" if hold.get("scope") == "decision" else "every record of the model"
            matter = f" ({_e(hold.get('matter'))})" if hold.get("matter") else ""
            custody.append(f'<div>{_badge("on hold", "warn")} <b>{_e(hold.get("reason"))}</b>{matter} <span class="km-faint">covers {covers}</span></div>')
        for entry in history[-12:]:
            when = long_date(entry["at"]) if entry.get("at") else ""
            custody.append(f'<div><span class="km-faint">{_e(when)} · {_e(entry.get("by"))}</span> {_e(entry.get("summary"))}</div>')
        out.append(_details("Custody of this record", f'<div class="km-cf">{"".join(custody)}</div>', open_=bool(held)))

    inputs = [i for i in (data.get("inputs") or []) if isinstance(i, Mapping)]
    if inputs:
        table = _table(
            [("Input", ""), ("Name", "mono muted"), ("Value as stored", "mono"), ("Unit", "muted")],
            (
                [
                    _e(i.get("label") or ""),
                    _e(i.get("name") or ""),
                    _e(i.get("exact") if i.get("exact") is not None else i.get("display") or ""),
                    _e(i.get("unit") or ""),
                ]
                for i in inputs
            ),
        )
        out.append(_details("Inputs as recorded", table, note=f"{len(inputs)} inputs"))
    if explanation:
        compared = _map(explanation.get("reference"))
        source = str(compared.get("source_words") or "")
        if source and compared.get("rows"):
            source = f"{int(compared['rows']):,} rows from {source}"
        how = [
            ("method", _e(f"{explanation.get('method_name')}. {explanation.get('method_sentence') or ''}".strip())),
            ("reproducible", _e(explanation.get("reproducible_sentence") or "")),
            ("compared with", _e(source)),
            ("seed", _e("none" if explanation.get("seed") is None else explanation.get("seed"))),
            ("explanation", _e(explanation.get("id") or "")),
            ("computed", _e(long_date(explanation["computed_at"])) if explanation.get("computed_at") else ""),
        ]
        out.append(_details("How this was computed", _kv_table(how)))
    limits = [item for item in (data.get("limits") or []) if isinstance(item, Mapping)]
    if limits:
        lines = "".join(f'<div><b>{_e(item.get("title"))}</b> {_e(item.get("text"))}</div>' for item in limits)
        notice = f'<span class="km-faint">{_e(data.get("notice") or "")}</span>'
        out.append(_details("What this record does not show", f'<div class="km-cf">{lines}{notice}</div>'))
    out.append(_raw_json("JSON", data, note="to_json() has everything"))
    mark = str(_map(data.get("fingerprint")).get("value") or "")
    left = f'Fingerprint <span class=km-mono title="sha256:{_e(mark)}">sha256:{_e(mark[:16])}…</span>' if mark else ""
    where = "/decisions/" + "/".join(quote(part, safe="") for part in str(data.get("id") or "").split("/"))
    out.append(_foot(left, _link(where, "Open in dashboard") if data.get("id") else ""))
    return "".join(out)


def _period(days: Any) -> str:
    return "until deleted" if days is None else f"{int(days):,} days"


def _retention(data: Mapping[str, Any]) -> str:
    """An organization's retention schedule: its period, then each model's, what is due and what is held."""
    default = _map(data.get("default"))
    models = [m for m in (data.get("models") or []) if isinstance(m, Mapping)]
    out = [_head("Retention", None, "")]
    if default:
        main, sub = f"{int(default.get('keep_days') or 0):,} days", "after a record is logged, it is deleted"
    else:
        main, sub = "Kept until deleted", "no period is set"
    out.append(f'<div class="km-result"><span class="km-value">{_e(main)}</span><span class="km-sub">{_e(sub)}</span></div>')
    due_d = sum(int(_map(m.get("due")).get("decisions") or 0) for m in models)
    due_e = sum(int(_map(m.get("due")).get("explanations") or 0) for m in models)
    holds = int(_map(data.get("holds")).get("active") or 0)
    badges = [_badge(f"{due_d:,} decisions and {due_e:,} explanations due", "warn" if due_d or due_e else "ok")]
    badges.append(_badge(f"{holds:,} hold{'' if holds == 1 else 's'} in force", "warn" if holds else ""))
    if data.get("last_run_at"):
        badges.append(_badge(f"last applied {long_date(data['last_run_at'])}"))
    out.append(_badges(badges))
    rows = []
    for item in models:
        model, held, due = _map(item.get("model")), _map(item.get("held")), _map(item.get("due"))
        whose = {"model": " (its own)", "organization": "", "none": ""}.get(str(item.get("source")), "")
        hold = "every record" if held.get("model") else (f"{int(held.get('decisions') or 0):,} decisions" if held.get("decisions") else "—")
        rows.append(
            [
                f"<b>{_e(model.get('name'))}</b>",
                _e(_period(item.get("keep_days")) + whose),
                _e(f"{int(item.get('decisions') or 0):,}"),
                _date_cell(item.get("oldest_at")),
                _e("held" if held.get("model") else f"{int(due.get('decisions') or 0):,}"),
                _e(hold),
            ]
        )
    if rows:
        out.append(
            _table(
                [("Model", ""), ("Records are kept", ""), ("Decisions", "num"), ("Oldest", "muted"), ("Due now", "num"), ("Holds", "")],
                rows,
            )
        )
    out.append(f'<div class="km-lead"><span class="km-faint">{_e(data.get("covers") or "")}</span></div>')
    out.append(_raw_json("JSON", data, note="to_json() has everything"))
    out.append(_foot(f"Shortest period {int(data.get('minimum_days') or 0)} days", _link("/retention", "Open in dashboard")))
    return "".join(out)


_EVIDENCE_WORDS = {
    "held": ("held", "ok"),
    "partial": ("partly held", "warn"),
    "stale": ("out of date", "warn"),
    "missing": ("not held yet", ""),
    "not_applicable": ("does not apply", ""),
}


def _evidence_summary(frameworks: Iterable[Any]) -> str:
    """A table of rules: how many requirements of each have evidence held, and the sentence that says it."""
    rows = []
    for framework in frameworks:
        framework = _map(framework)
        summary = _map(framework.get("summary"))
        sentence = framework.get("sentence") or summary.get("sentence") or ""
        rows.append(
            [
                f'<b>{_e(framework.get("name"))}</b><br><span class="km-faint">{_e(framework.get("citation") or "")}</span>',
                _e(sentence),
            ]
        )
    return _table([("Rule", "wrap"), ("Evidence Kaimei holds", "wrap")], rows) if rows else ""


def _evidence(data: Mapping[str, Any]) -> str:
    """A model's evidence by requirement: each rule in a sentence, the requirements under it, and what to add."""
    model = _map(data.get("model"))
    frameworks = [f for f in (data.get("frameworks") or []) if isinstance(f, Mapping)]
    held = sum(int(_map(f.get("summary")).get("held") or 0) for f in frameworks)
    total = sum(int(_map(f.get("summary")).get("requirements") or 0) for f in frameworks)
    out = [_head("Evidence", model.get("slug"), _when(data, key="checked_at"))]
    out.append(
        f'<div class="km-result"><span class="km-value">{held} of {total}</span>'
        f'<span class="km-sub">requirements with evidence held · {_e(model.get("name") or "")}</span></div>'
    )
    out.append(_evidence_summary(frameworks))
    for framework in frameworks:
        rows = []
        for requirement in framework.get("requirements") or []:
            requirement = _map(requirement)
            words, tone = _EVIDENCE_WORDS.get(str(requirement.get("status")), (str(requirement.get("status")), ""))
            parts = [_map(part) for part in requirement.get("evidence") or []]
            listed = "; ".join(
                f"{_e(part.get('label'))} ({_e(_EVIDENCE_WORDS.get(str(part.get('status')), ('', ''))[0])})" for part in parts
            )
            rows.append(
                [
                    f'<b>{_e(requirement.get("title"))}</b><br><span class="km-faint">{_e(requirement.get("reference"))}</span>',
                    _badge(words, tone),
                    listed,
                ]
            )
        if rows:
            body = _table([("Requirement", "wrap"), ("Evidence", ""), ("What is checked", "wrap")], rows)
            out.append(_details(_e(framework.get("name")), body, note=f"{len(rows)} requirements"))
    todo = [_map(entry) for entry in data.get("todo") or []]
    if todo:
        lines = "".join(f'<div><b>{_e(entry.get("label"))}.</b> {_e(entry.get("todo"))}</div>' for entry in todo)
        out.append(_details("What would add evidence", f'<div class="km-cf">{lines}</div>', open_=True, note=str(len(todo))))
    out.append(_raw_json("JSON", data))
    link = _link(f"/models/{quote(str(model.get('slug')), safe='')}/compliance", "Open in dashboard") if model.get("slug") else ""
    out.append(_foot(f'<span class="km-faint">{_e(data.get("notice") or "")}</span>', link))
    return "".join(out)


def _model_report(data: Mapping[str, Any]) -> str:
    """A model documentation report: what the model is, what its analyses found, and the evidence held."""
    model = _map(data.get("model"))
    out = [_head("Model documentation", model.get("slug"), _when(data, key="generated_at"))]
    out.append(
        f'<div class="km-result"><span class="km-title">{_e(model.get("name") or "")}</span>'
        f'<span class="km-sub">version {_e(model.get("version"))} · {_e(model.get("task_words") or model.get("task") or "")}</span></div>'
    )
    if model.get("description"):
        out.append(f'<div class="km-lead" style="color:var(--text)">{_e(model["description"])}</div>')
    said = " ".join(str(part) for part in (_map(data.get("output")).get("sentence"), _map(data.get("runs")).get("sentence")) if part)
    if said:
        out.append(f'<div class="km-lead">{_e(said)}</div>')
    findings = [_map(f) for f in data.get("findings") or []]
    if findings:
        out.append('<div class="km-h">What the analyses found</div>')
        lines = "".join(
            f'<div><b>{_e(f.get("name"))}.</b> {_e(f.get("summary") or f.get("title") or "")} '
            f'<span class="km-faint">{_e(short_date(f.get("created_at")))}</span></div>'
            for f in findings
        )
        out.append(f'<div class="km-cf">{lines}</div>')
    not_run = [str(name) for name in data.get("not_run") or []]
    if not_run:
        out.append(f'<div class="km-faint" style="margin-top:6px">Not run: {_e(", ".join(not_run))}.</div>')
    holdings = [_map(h) for h in data.get("holdings") or []]
    if holdings:
        rows = [[f'<b>{_e(h.get("label"))}</b>', _e(h.get("detail") or "")] for h in holdings]
        out.append(_details("What Kaimei holds about it", _table([("What", ""), ("Held", "wrap")], rows), note=str(len(rows))))
    frameworks = [f for f in (data.get("evidence") or []) if isinstance(f, Mapping)]
    if frameworks:
        out.append(_details("Evidence held, by rule", _evidence_summary(frameworks), open_=True))
    limits = "".join(f"<div>{_e(line)}</div>" for line in data.get("limits") or [])
    if limits:
        out.append(_details("What this report does not show", f'<div class="km-cf">{limits}</div>'))
    out.append(_raw_json("JSON", data, note="to_json() has everything"))
    mark = str(_map(data.get("fingerprint")).get("value") or "")
    left = f'Fingerprint <span class=km-mono title="sha256:{_e(mark)}">sha256:{_e(mark[:16])}…</span>' if mark else ""
    link = _link(f"/models/{quote(str(model.get('slug')), safe='')}/compliance", "Open in dashboard") if model.get("slug") else ""
    out.append(_foot(left, link))
    return "".join(out)


def _reason_table(data: Mapping[str, Any]) -> str:
    """A reason table: which version it is, what it gives reasons for, and its wording input by input."""
    model = _map(data.get("model"))
    adverse = _map(data.get("adverse"))
    coverage = _map(data.get("coverage"))
    status = str(data.get("status") or "")
    out = [_head("Reason table", data.get("id"), _when(data, key="approved_at") or _when(data, key="updated_at"))]
    title = "Draft" if status == "draft" else f"Version {data.get('version')}"
    out.append(f'<div class="km-result"><span class="km-title">{_e(title)}</span><span class="km-sub">{_e(model.get("name") or "")}</span></div>')
    badges = [_badge("in force", "ok") if status == "approved" else _badge(status or "draft", "warn" if status == "draft" else "")]
    against = adverse.get("outcome") or (f"a {adverse.get('direction')} prediction" if adverse.get("direction") else "")
    if against:
        badges.append(_badge(f"reasons for {against}", "info"))
    badges.append(_badge(f"at most {data.get('max_reasons')} reasons", mono=True))
    undecided = [str(name) for name in coverage.get("unmapped") or []]
    badges.append(_badge("every input decided", "ok") if not undecided else _badge(f"{len(undecided)} inputs undecided", "warn"))
    out.append(_badges(badges))
    who = _map(data.get("approved_by")).get("name")
    if status != "draft" and who:
        out.append(f'<div class="km-lead">Approved by {_e(who)}.</div>')
    elif status == "draft":
        out.append('<div class="km-lead">A draft changes nothing until a reviewer, admin or owner approves it.</div>')
    rows = [
        [_e(name), _badge("reason", "ok"), _e(_map(entry).get("reason") or ""), _e(_map(entry).get("code") or "")]
        for name, entry in _map(data.get("entries")).items()
    ]
    rows += [[_e(name), _badge("never state", "warn"), _e(why or ""), ""] for name, why in _map(data.get("never_state")).items()]
    rows += [[_e(name), _badge("undecided"), '<span class="km-faint">held for review when it pushes toward the adverse outcome</span>', ""] for name in undecided]
    if rows:
        out.append(_table([("Input", "mono"), ("Treatment", ""), ("Approved wording, or why never stated", "wrap"), ("Code", "mono muted")], rows))
    if data.get("note"):
        out.append(f'<div class="km-faint" style="margin-top:8px">What changed: {_e(data.get("note"))}</div>')
    out.append(_raw_json("JSON", data))
    mark = str(data.get("content_hash") or "")
    left = f'Content <span class=km-mono title="sha256:{_e(mark)}">sha256:{_e(mark[:16])}…</span>' if mark else ""
    link = _link(f"/models/{quote(str(model.get('slug')), safe='')}/compliance?view=reasons", "Open in dashboard") if model.get("slug") else ""
    out.append(_foot(left, link))
    return "".join(out)


def _token_strip(parts: Sequence[Mapping[str, Any]]) -> str:
    """A text as the model read it: each token on the colour of its effect, stronger for a larger one."""
    chips = []
    for part in parts:
        strength = max(0.0, min(1.0, _float(part.get("strength")) or 0.0))
        direction = part.get("direction")
        style = "display:inline-block;padding:1px 5px;margin:2px 1px;border-radius:5px;border:1px solid var(--line)"
        if direction in ("raises", "lowers") and strength >= 0.02:
            hue = "107,140,255" if direction == "raises" else "255,107,181"
            style += f";background:rgba({hue},{0.14 + 0.5 * strength:.2f})"
        value = _float(part.get("attribution"))
        title = f' title="{_e(fmt_signed(value, 4))}"' if value is not None else ""
        chips.append(f'<span style="{style}"{title}>{_e(part.get("label"))}</span>')
    key = (
        '<div class="km-legend" style="margin-top:6px"><span><i style="background:rgba(107,140,255,.6)"></i>raised it</span>'
        '<span><i style="background:rgba(255,107,181,.6)"></i>lowered it</span></div>'
    )
    return f'<div style="margin:2px 0 12px;line-height:2">{"".join(chips)}{key}</div>'


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
        # A score that rows drawn from the reference data itself can reach is not shown as drift (API 1.5.0).
        features = [
            (
                str(f.get("feature")),
                _float(f.get("psi")) or 0.0,
                "" if f.get("beyond_chance") is False and f.get("status") != "stable" else str(f.get("status") or ""),
            )
            for f in _dicts(payload.get("features"))
            if f.get("feature")
        ][:limit]
        chips = []
        if _float(payload.get("overall_psi")) is not None:
            chips.append(_badge(f"PSI {fmt_number(payload['overall_psi'], 3)}", mono=True, title="Population stability index over every input"))
        if payload.get("beyond_chance") is False and payload.get("status") in ("moderate", "significant"):
            reach = f" (up to {fmt_number(payload['chance'], 2)})" if _float(payload.get("chance")) is not None else ""
            chips.append(_badge("within chance", title=f"As many rows drawn from the reference data itself score as high{reach}"))
        elif isinstance(payload.get("status"), str):
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
    if kind == "performance":
        metric = str(payload.get("metric") or "")
        current, baseline, comparison = _map(payload.get("current")), _map(payload.get("baseline")), _map(payload.get("comparison"))

        def figure(value: Any) -> str:
            return pct(value, 1) if metric in SHARE_METRICS else fmt_number(value, 3)

        name = METRIC_SHORT.get(metric, metric)
        chips = []
        if _float(current.get(metric)) is not None:
            chips.append(_badge(f"{name} {figure(current[metric])} · {compact(current.get('n'))} recent", mono=True))
        if _float(baseline.get(metric)) is not None:
            chips.append(_badge(f"baseline {figure(baseline[metric])} · {compact(baseline.get('n'))}", mono=True))
        # Precision and recall beside whatever the check watches: each says what the other hides.
        for key in ("precision", "recall", "macro_precision", "macro_recall"):
            if key != metric and _float(current.get(key)) is not None:
                chips.append(_badge(f"{METRIC_SHORT[key]} {pct(current[key], 1)}", mono=True))
        status = payload.get("status")
        if isinstance(status, str):
            chips.append(_badge(status.replace("_", " "), STATUS_TONE.get(status, "")))
        if _float(comparison.get("z")) is not None:
            chips.append(_badge(f"z {fmt_number(comparison['z'], 2)}", mono=True, title="The change in standard errors; beyond ±1.96 it is more than chance"))
        values = [v for v in (_float(w.get(metric)) for w in _dicts(payload.get("series"))) if v is not None]
        return _badges(chips) + _sparkline(values, width=300, title=f"{name} per week")
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
    actual = _map(data.get("actual"))
    if actual:
        if actual.get("label") is not None:
            happened, tone = str(actual["label"]), ("ok" if str(actual["label"]) == label else "bad")
        else:
            happened, tone = fmt_number(actual.get("value"), 3), ""
        badges.append(_badge(f"actual {happened}", tone, title="What really happened"))
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
    if kind == "decision":
        rows = []
        for d in items:
            outcome = _map(d.get("outcome"))
            if outcome.get("label") is not None:
                chance = _float(outcome.get("probability"))
                shown = f"<b>{_e(outcome['label'])}</b>" + (f' <span class="km-faint">{pct(chance, 1)}</span>' if chance is not None else "")
            elif _float(outcome.get("value")) is not None:
                shown = f"<b>{_e(fmt_value(outcome.get('value'), outcome.get('unit') or None))}</b>"
            else:
                shown = '<span class="km-faint">not logged</span>'
            rows.append(
                [
                    _e(d.get("id") or ""),
                    _e(d.get("model_id") or ""),
                    shown,
                    _badge("explained", "ok") if d.get("explained") else _badge("not explained"),
                    _date_cell(d.get("decided_at")),
                ]
            )
        return _table([("Decision id", "mono"), ("Model", "mono muted"), ("Outcome", ""), ("Record", ""), ("Decided", "muted")], rows)
    if kind == "audit_event":
        rows = [
            [
                _date_cell(e.get("at")),
                _e(_map(e.get("actor")).get("name") or ""),
                f"{_e(e.get('summary') or '')}<br><span class=\"km-faint km-mono\">{_e(e.get('action') or '')}</span>",
            ]
            for e in items
        ]
        return _table([("When", "muted"), ("Who", ""), ("What happened", "wrap")], rows)
    if kind == "hold":
        rows = [
            [
                _e(h.get("decision_id") or f"every record of {_map(h.get('model')).get('name') or 'a model'}"),
                _e(h.get("reason") or ""),
                _e(h.get("matter") or "—"),
                _badge("in force", "warn") if h.get("active") else _badge("released"),
                _date_cell(h.get("placed_at")),
            ]
            for h in items
        ]
        return _table([("Holds", "mono"), ("Why", "wrap"), ("Matter", "mono"), ("State", ""), ("Placed", "muted")], rows)
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
