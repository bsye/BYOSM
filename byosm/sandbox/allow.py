from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

HUB_HOSTS = ("huggingface.co", "hf.co")
SCHEME_PORTS = {"http": 80, "https": 443}
DEFAULT_PROVIDER = "anthropic"


def _matches(host: str, port: int, rule: str) -> bool:
    name, sep, only = rule.rpartition(":")
    if sep and only.isdigit():
        return host == name and port == int(only)
    return host == rule or host.endswith("." + rule)


def allowed(host: str, port: int, rules: list[str]) -> bool:
    return any(_matches(host, port, rule) for rule in rules)


def connections(ws: Path, llm: dict[str, Any] | None) -> list[dict[str, Any]]:
    from ..config import load_config
    from ..llm import role_config

    roles = {"default": llm} if llm else load_config(ws)["llm"]
    return [role_config(roles, role) for role in roles]


def endpoint(conf: dict[str, Any]) -> str | None:
    from ..llm.detect import ANTHROPIC_URL, OPENAI_URL

    defaults = {DEFAULT_PROVIDER: ANTHROPIC_URL, "openai": OPENAI_URL}
    url = urlsplit(conf.get("base_url") or defaults.get(conf.get("provider", DEFAULT_PROVIDER), ""))
    if not url.hostname:
        return None
    return f"{url.hostname}:{url.port or SCHEME_PORTS.get(url.scheme, SCHEME_PORTS['https'])}"


def allow_list(ws: Path, llms: list[dict[str, Any]]) -> list[str]:
    from ..config import job_settings

    found = filter(None, map(endpoint, llms))
    return list(dict.fromkeys([*HUB_HOSTS, *found, *job_settings(ws)["network"]]))
