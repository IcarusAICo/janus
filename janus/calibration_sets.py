"""Per-task temperatures: one per (family, cardinality bucket) with family and global fallbacks (Phase 4 production).

The calibration file keeps the keys the server reads (`temperature`, `by_cardinality`, `histogram`, checkpoint and
data hashes) and adds `by_family`: {family: {"temperature": t, "by_cardinality": {bucket: t}}}. A file without
`by_family`, or a family or bucket that is missing, falls back exactly as before, so old files load unchanged and the
server (which reads only `temperature`) accepts the new file. Families are the group_id prefix, as `jev evaluate`
groups them; buckets are powers of two ("2", "4" for 3-4 options, "8" for 5-8, ...) so sparse cardinalities share a fit.
"""

from .metrics import fit_temperature


def bucket(cardinality):
    return str(2 ** max(1, (int(cardinality) - 1).bit_length()))


def family_of_request(request):
    return request.group_id.split(":", 1)[0]


def fit_temperature_by_family(logits, targets, families, minimum=20):
    """{family: {temperature, by_cardinality}} for every family with at least `minimum` questions; a bucket inside a
    family gets its own fit only with `minimum` questions, otherwise the family temperature applies."""
    if not len(logits) == len(targets) == len(families):
        raise ValueError("logits, targets and families must align")
    out = {}
    for family in sorted(set(families)):
        rows = [(z, y) for z, y, f in zip(logits, targets, families) if f == family]
        if len(rows) < minimum:
            continue
        by_bucket = {}
        for name in sorted({bucket(len(z)) for z, _ in rows}, key=int):
            subset = [(z, y) for z, y in rows if bucket(len(z)) == name]
            if len(subset) >= minimum:
                by_bucket[name] = fit_temperature([z for z, _ in subset], [y for _, y in subset])
        out[family] = {"temperature": fit_temperature([z for z, _ in rows], [y for _, y in rows]), "by_cardinality": by_bucket}
    return out


def temperature_for(calibration, family, cardinality, kind=None):
    """The most specific temperature the file carries: (family, bucket), then family, then the question kind's
    fallback (`by_kind`, for requests without a calibrated family), then the global entry."""
    entry = (calibration.get("by_family") or {}).get(family)
    if entry is None:
        return (calibration.get("by_kind") or {}).get(kind, calibration["temperature"])
    return entry.get("by_cardinality", {}).get(bucket(cardinality), entry["temperature"])


def by_family_logits(logits, families, calibration):
    return [z / temperature_for(calibration, f, len(z)) for z, f in zip(logits, families)]
