def normalize(text: str) -> str:
    return " ".join("".join(c if c.isalnum() else " " for c in str(text).lower()).split())
