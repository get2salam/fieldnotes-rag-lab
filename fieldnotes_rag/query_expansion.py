"""Query expansion — synonym substitution and morphological variants for field-research terms."""

from __future__ import annotations

from typing import Dict, List

SYNONYMS: Dict[str, List[str]] = {
    # Birds
    "bird": ["avian", "passerine", "raptor", "waterfowl", "songbird", "shorebird", "species"],
    "birds": ["avian", "passerines", "raptors", "waterfowl", "songbirds", "species"],
    "dipper": ["cinclus", "americanus", "streamside", "aquatic"],
    "kingfisher": ["megaceryle", "alcyon", "riparian", "hover"],
    "duck": ["waterfowl", "mergus", "merganser", "anatidae", "aquatic"],
    "warbler": ["setophaga", "passerine", "songbird", "breeding"],
    "raptor": ["hawk", "eagle", "osprey", "falcon", "accipiter"],
    "osprey": ["pandion", "haliaetus", "fish-hawk", "piscivore"],

    # Water and stream terms
    "stream": ["creek", "river", "riffle", "brook", "watercourse", "drainage"],
    "streams": ["creeks", "rivers", "riffles", "brooks"],
    "streamside": ["riparian", "bank", "waterside", "riverside", "creek-side"],
    "riffle": ["fast-water", "turbulent", "cascade", "rapid", "whitewater"],
    "pool": ["slow-water", "reach", "pond", "backwater"],
    "crossing": ["ford", "traverse", "wade", "wading", "stream-crossing"],
    "creek": ["stream", "brook", "burn", "run", "drainage"],
    "flood": ["high-water", "inundation", "surge", "spate"],

    # Habitat
    "habitat": ["environment", "microhabitat", "ecosystem", "zone", "biotope"],
    "riparian": ["streamside", "bank", "waterside", "floodplain"],
    "canopy": ["overstory", "crown", "treetop", "forest-cover"],
    "bank": ["shore", "margin", "edge", "riverside"],
    "forest": ["woodland", "conifer", "stand", "timber"],

    # Plants
    "plant": ["vegetation", "flora", "shrub", "herb", "species", "growth"],
    "plants": ["vegetation", "flora", "shrubs", "species"],
    "alder": ["alnus", "nitrogen-fixer", "riparian-tree"],
    "willow": ["salix", "bank-stabiliser", "riparian-shrub"],
    "invasive": ["exotic", "non-native", "introduced", "weed"],

    # Wildlife actions
    "observe": ["watch", "detect", "spot", "sight", "record", "monitor", "survey"],
    "record": ["log", "note", "document", "note", "register", "report"],
    "identify": ["id", "confirm", "determine", "distinguish"],
    "forage": ["feed", "hunt", "search", "prey"],

    # Temporal
    "dusk": ["twilight", "evening", "sunset", "crepuscular", "civil-twilight"],
    "dawn": ["morning", "sunrise", "crepuscular", "early", "first-light"],
    "night": ["nocturnal", "darkness", "after-dark"],
    "season": ["spring", "summer", "autumn", "fall", "winter", "breeding"],

    # Safety
    "safety": ["hazard", "risk", "protocol", "precaution", "danger"],
    "hazard": ["danger", "risk", "threat", "caution"],
    "emergency": ["rescue", "incident", "accident", "sos"],

    # Weather
    "weather": ["precipitation", "temperature", "wind", "conditions", "climate"],
    "rain": ["precipitation", "downpour", "drizzle", "showers"],
    "wind": ["breeze", "gust", "airflow", "draught"],
    "cold": ["temperature", "frost", "chill", "cool"],

    # Survey methods
    "survey": ["count", "census", "monitoring", "assessment", "transect"],
    "observation": ["sighting", "record", "detection", "encounter"],
    "detection": ["observation", "sighting", "discovery", "encounter"],
    "species": ["animal", "organism", "taxa", "vertebrate", "wildlife"],
}

# Morphological suffix variants for common terms
MORPHOLOGICAL_VARIANTS: Dict[str, List[str]] = {
    "observe": ["observed", "observing", "observation"],
    "record": ["recorded", "recording", "records"],
    "forage": ["foraging", "forages", "forager"],
    "survey": ["surveying", "surveyed", "surveys"],
    "breed": ["breeding", "breeds", "breeder"],
    "nest": ["nesting", "nests", "nested"],
    "migrate": ["migrating", "migration", "migratory"],
}


def expand_query(query: str) -> str:
    """Expand a query with synonyms and morphological variants to improve recall."""
    tokens = query.lower().split()
    extra: List[str] = []

    for token in tokens:
        # Direct synonym expansion
        syns = SYNONYMS.get(token, [])
        extra.extend(syns)

        # Morphological variant expansion
        morphs = MORPHOLOGICAL_VARIANTS.get(token, [])
        extra.extend(morphs)

        # Strip trailing 's' for plural/singular coverage
        if token.endswith("s") and len(token) > 3:
            root = token[:-1]
            syns_root = SYNONYMS.get(root, [])
            if syns_root:
                extra.extend(syns_root)

    if extra:
        # Deduplicate while preserving order
        seen = set(tokens)
        unique_extra = []
        for t in extra:
            if t not in seen:
                unique_extra.append(t)
                seen.add(t)
        return query + " " + " ".join(unique_extra)

    return query
