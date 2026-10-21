def normalize(text: str) -> str:
    return " ".join("".join(c if c.isalnum() else " " for c in str(text).lower()).split())


def match_label(name: str, candidates: list[str]) -> str | None:
    words = normalize(name).split()
    exact = [c for c in candidates if normalize(c).split() == words]
    hits = exact or [c for c in candidates if _abbreviates(words, normalize(c).split())]
    return hits[0] if len(hits) == 1 else None


def _abbreviates(a: list[str], b: list[str]) -> bool:
    return len(a) == len(b) and all(x.startswith(y) or y.startswith(x) for x, y in zip(a, b, strict=True))
