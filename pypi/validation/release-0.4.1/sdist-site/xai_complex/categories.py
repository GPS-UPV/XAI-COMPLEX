"""Five ordered empirical difficulty labels."""
CATEGORIES = ("toy", "easy", "medium", "hard", "open")


def validate_protocol(protocol):
    if not isinstance(protocol, dict) or not isinstance(protocol.get("id"), str) or not protocol["id"].strip():
        raise ValueError("A named labeling protocol is required")
    definitions = protocol.get("class_definitions", {})
    if set(definitions) != set(CATEGORIES) or any(not isinstance(v, str) or not v.strip() for v in definitions.values()):
        raise ValueError("Define toy, easy, medium, hard and open explicitly in the labeling protocol")
    return protocol
