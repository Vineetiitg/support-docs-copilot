import re


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip()


def query_variants(query: str) -> list[str]:
    normalized = normalize_query(query)
    variants = [normalized]
    if "error" in normalized.lower() and "troubleshoot" not in normalized.lower():
        variants.append(f"troubleshoot {normalized}")
    if "how" in normalized.lower() and "steps" not in normalized.lower():
        variants.append(f"{normalized} steps")
    return list(dict.fromkeys(variants))
