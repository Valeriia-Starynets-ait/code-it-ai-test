"""Greeting helpers."""


def greet(name: str) -> str:
    """Return a greeting for ``name``.

    If ``name`` is empty or only whitespace, greet the world instead.
    """
    cleaned = name.strip()
    if not cleaned:
        return "Hello, world!"
    return f"Hello, {cleaned}!"
