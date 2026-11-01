"""Repeat a card's stylesheet in ``style`` attributes, for viewers that drop ``<style>`` blocks.

GitHub's notebook viewer, and other renderers that sanitize HTML outputs, remove ``<style>``
elements but keep inline ``style`` attributes. Without the stylesheet a card would lose its layout
(and its logo would grow to the full width of the page). So each card keeps its stylesheet, which
still drives themes, hover and open/closed sections, and every element also carries its resting
style inline:

- Custom properties stay in the stylesheet. Inline values read them with the light theme's value
  as a fallback (``color:var(--text,#12141c)``), so a notebook's dark theme still applies and a
  viewer without the stylesheet shows the light theme.
- A property that a ``:hover`` / ``[open]`` / ``:last-child`` rule or an ``@supports`` block changes
  is left to the stylesheet (with any shorthand that covers it), because an inline value always
  wins over a stylesheet rule.
- Pseudo-element rules (``::before``) and interaction-only properties are not copied, and a
  ``<summary>`` keeps its native disclosure marker.

Only the selector forms the card stylesheet uses are understood: tags, classes, descendant and
child combinators, with pseudo-classes and attribute selectors treated as states.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

__all__ = ["inline_styles"]

_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
_RAW = {"style", "script"}
#: Properties that only matter for interaction, or that would hide a ``<summary>``'s marker.
_NEVER = {"cursor", "user-select", "transition", "list-style"}
_NEVER_ON = {"summary": {"display"}}

_TAG = re.compile(r"""<(/?)([A-Za-z][A-Za-z0-9-]*)((?:\s+[^\s=>/]+(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>"']+))?)*)\s*(/?)>""")
_CLASS_ATTR = re.compile(r"""\sclass\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>"']+))""", re.IGNORECASE)
_STYLE_ATTR = re.compile(r"""\sstyle\s*=\s*"([^"]*)\"""", re.IGNORECASE)
_COMPOUND = re.compile(r"([A-Za-z][A-Za-z0-9]*)?((?:\.[A-Za-z0-9_-]+)*)((?:\[[^\]]*\]|::?[A-Za-z-]+(?:\([^)]*\))?)*)")
_VAR = re.compile(r"var\((--[A-Za-z0-9-]+)\)")

# (combinator, tag or None, classes) from the leftmost compound to the rightmost.
_Part = Tuple[str, Optional[str], FrozenSet[str]]


class _Node:
    __slots__ = ("tag", "classes", "parent")

    def __init__(self, tag: str, classes: FrozenSet[str], parent: Optional["_Node"]) -> None:
        self.tag, self.classes, self.parent = tag, classes, parent


class _Sheet:
    def __init__(self, css: str) -> None:
        #: (parts, specificity, order, declarations) of the rules copied inline.
        self.rules: List[Tuple[List[_Part], Tuple[int, int], int, List[Tuple[str, str]]]] = []
        #: (parts, properties) of the rules that depend on a state the inline copy can't follow.
        self.states: List[Tuple[List[_Part], Set[str]]] = []
        self.variables: Dict[str, str] = {}
        order = 0
        for prelude, body in _blocks(css):
            if prelude.startswith("@"):
                for inner_prelude, inner_body in _blocks(body):
                    for selector in inner_prelude.split(","):
                        parsed = _parse(selector)
                        if parsed is not None:
                            self.states.append((parsed[0], {p for p, _ in _declarations(inner_body)}))
                continue
            declarations = _declarations(body)
            for selector in prelude.split(","):
                selector = selector.strip()
                if selector == ".km":
                    self.variables.update({p: v for p, v in declarations if p.startswith("--")})
                if selector.startswith("body") or "::" in selector:
                    continue  # theme blocks (custom properties only) and pseudo-elements
                parsed = _parse(selector)
                if parsed is None:
                    continue
                parts, specificity, stateful = parsed
                if stateful:
                    self.states.append((parts, {p for p, _ in declarations}))
                    continue
                kept = [(p, v) for p, v in declarations if not p.startswith("--") and p not in _NEVER]
                if kept:
                    self.rules.append((parts, specificity, order, kept))
                    order += 1

    def style_for(self, node: _Node) -> str:
        matched = sorted((spec, order, decls) for parts, spec, order, decls in self.rules if _matches(parts, node))
        if not matched:
            return ""
        held = {p for parts, props in self.states if _matches(parts, node) for p in props}
        held |= _NEVER_ON.get(node.tag, set())
        values: Dict[str, str] = {}
        for _, _, declarations in matched:
            for prop, value in declarations:
                if any(_related(prop, h) for h in held):
                    continue
                values.pop(prop, None)  # keep the cascade order: a later value moves to the end
                values[prop] = value
        return ";".join(f"{p}:{self._fallback(v)}" for p, v in values.items())

    def _fallback(self, value: str) -> str:
        def resolve(match: "re.Match[str]") -> str:
            name = match.group(1)
            light = self.variables.get(name)
            return f"var({name},{light})" if light is not None else match.group(0)

        return _VAR.sub(resolve, value).replace('"', "'")


def _related(prop: str, held: str) -> bool:
    """``prop`` sets ``held`` or part of it (``border`` covers ``border-bottom``), or the reverse."""
    return prop == held or held.startswith(prop + "-") or prop.startswith(held + "-")


def _blocks(css: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    i = 0
    while True:
        start = css.find("{", i)
        if start < 0:
            return out
        depth, j = 1, start + 1
        while depth and j < len(css):
            depth += {"{": 1, "}": -1}.get(css[j], 0)
            j += 1
        out.append((css[i:start].strip(), css[start + 1 : j - 1]))
        i = j


def _declarations(body: str) -> List[Tuple[str, str]]:
    out = []
    for item in body.split(";"):
        prop, sep, value = item.partition(":")
        if sep and prop.strip() and value.strip():
            out.append((prop.strip().lower(), value.strip()))
    return out


def _parse(selector: str) -> Optional[Tuple[List[_Part], Tuple[int, int], bool]]:
    parts: List[_Part] = []
    classes_n = tags_n = 0
    stateful = False
    combinator = " "
    for token in re.findall(r">|[^\s>]+", selector.strip()):
        if token == ">":
            combinator = ">"
            continue
        match = _COMPOUND.fullmatch(token)
        if not match or not (match.group(1) or match.group(2)):
            return None  # "*" and other forms the card stylesheet doesn't need inline
        tag, classes, extra = match.group(1), match.group(2), match.group(3)
        names = frozenset(c for c in classes.split(".") if c)
        parts.append((combinator, tag.lower() if tag else None, names))
        classes_n += len(names) + (extra.count(":") + extra.count("[") if extra else 0)
        tags_n += 1 if tag else 0
        stateful = stateful or bool(extra)
        combinator = " "
    return (parts, (classes_n, tags_n), stateful) if parts else None


def _matches(parts: List[_Part], node: _Node) -> bool:
    def fits(part: _Part, candidate: _Node) -> bool:
        return (part[1] is None or part[1] == candidate.tag) and part[2] <= candidate.classes

    def walk(i: int, candidate: _Node) -> bool:
        if not fits(parts[i], candidate):
            return False
        if i == 0:
            return True
        ancestor = candidate.parent
        if parts[i][0] == ">":
            return ancestor is not None and walk(i - 1, ancestor)
        while ancestor is not None:
            if walk(i - 1, ancestor):
                return True
            ancestor = ancestor.parent
        return False

    return walk(len(parts) - 1, node)


@lru_cache(maxsize=4)
def _sheet(css: str) -> _Sheet:
    return _Sheet(css)


def inline_styles(html: str, css: str) -> str:
    """``html`` with each element's resting style from ``css`` repeated in its ``style`` attribute.

    An element's own ``style`` attribute comes last, so it still wins. ``<style>`` and ``<script>``
    contents are copied as they are.
    """
    sheet = _sheet(css)
    out: List[str] = []
    stack: List[_Node] = []
    position = 0
    raw: Optional[str] = None
    for match in _TAG.finditer(html):
        closing, name, attributes, self_closing = match.groups()
        tag = name.lower()
        if raw is not None:
            if closing and tag == raw:
                raw = None
            continue
        if closing:
            for k in range(len(stack) - 1, -1, -1):
                if stack[k].tag == tag:
                    del stack[k:]
                    break
            continue
        if tag in _RAW and not self_closing:
            raw = tag
            continue
        found = _CLASS_ATTR.search(attributes)
        classes = frozenset((next(g for g in found.groups() if g is not None) if found else "").split())
        node = _Node(tag, classes, stack[-1] if stack else None)
        style = sheet.style_for(node)
        if style:
            text = match.group(0)
            own = _STYLE_ATTR.search(text)
            if own:
                merged = f"{style};{own.group(1)}" if own.group(1).strip() else style
                text = text[: own.start(1)] + merged + text[own.end(1) :]
            else:
                cut = 1 + len(name)
                text = f'{text[:cut]} style="{style}"{text[cut:]}'
            out.append(html[position : match.start()])
            out.append(text)
            position = match.end()
        if not self_closing and tag not in _VOID:
            stack.append(node)
    out.append(html[position:])
    return "".join(out)
