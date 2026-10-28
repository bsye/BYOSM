import random

from byosm.options.situations import Situation

WORDS = ["refund", "flight", "weather", "invoice", "stock", "recipe", "route", "movie", "salary", "lyrics", "visa"]


def tools(n=120):
    return [
        {
            "name": f"tool{i:03d}",
            "description": f"Looks up {WORDS[i % len(WORDS)]} data, variant {i}.",
            "parameters": {
                "query": {"type": "str", "description": "what to look up"},
                "limit": {"type": "int, optional"},
            },
        }
        for i in range(n)
    ]


def pool(n=600, seed=3):
    rng, offer = random.Random(seed), tools()
    out = []
    for i in range(n):
        offered = rng.sample(offer, rng.randint(2, 4))
        chosen = {o["name"] for o in offered[: rng.randint(1, 2)]}
        words = " ".join(rng.sample(WORDS, 4))
        out.append(Situation(f"could you check {words} for case {i}", offered, chosen))
    return out
