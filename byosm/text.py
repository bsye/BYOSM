from .rt import OUTSIDE, bio


def normalize(text: str) -> str:
    return " ".join("".join(c if c.isalnum() else " " for c in str(text).lower()).split())


def slug(text: str, max_words: int = 3) -> str:
    return "-".join(normalize(text).split()[:max_words])


def match_label(name: str, candidates: list[str]) -> str | None:
    words = normalize(name).split()
    exact = [c for c in candidates if normalize(c).split() == words]
    hits = exact or [c for c in candidates if _abbreviates(words, normalize(c).split())]
    return hits[0] if len(hits) == 1 else None


def _abbreviates(a: list[str], b: list[str]) -> bool:
    return len(a) == len(b) and all(x.startswith(y) or y.startswith(x) for x, y in zip(a, b, strict=True))


MARKUP_OPEN, MARKUP_CLOSE, MARKUP_SEP = "[", "]", ":"


def markup_tags(text: str) -> tuple[list[str], list[str]]:
    words, tags, rest = [], [], text
    while rest:
        before, sep, after = rest.partition(MARKUP_OPEN)
        words += before.split()
        tags += [OUTSIDE] * len(before.split())
        if not sep:
            break
        inside, _, rest = after.partition(MARKUP_CLOSE)
        kind, _, value = (k.strip() for k in inside.partition(MARKUP_SEP))
        words += value.split()
        tags += bio(kind, len(value.split()))
    return words, tags
