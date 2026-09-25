"""Phase 1 states rendered as PNG images (JEV_PHASE1_IMAGE_V1). The menu, incident record, message features or free text
moves into the picture; the state text keeps only the request sentence or a one-line framing. Gold, questions and
group ids are those of the text twin, so accuracy can be compared row for row through `paired.jsonl`."""

import argparse
import json
from pathlib import Path
import random
import re
import textwrap

import matplotlib
from PIL import Image, ImageDraw, ImageFont

from ..data import file_hash, state_hash, write_json, write_jsonl

DATASET = "JEV_PHASE1_IMAGE_V1"
SPLITS = ("train", "dev", "calibration", "test", "test_post_unseen")
# Fonts shipped with matplotlib, so the rendering is identical on every machine.
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
FONTS = {"sans": "DejaVuSans.ttf", "bold": "DejaVuSans-Bold.ttf", "mono": "DejaVuSansMono.ttf",
         "monobold": "DejaVuSansMono-Bold.ttf", "serif": "DejaVuSerif.ttf", "serifbold": "DejaVuSerif-Bold.ttf"}
WIDTHS = (448, 512, 576, 640)
MIN_SIDE, MAX_SIDE = 448, 672
FRAMING = {"ord": "The incident record is shown in the image.",
           "post": "The message's features are shown in the image.",
           "study": "The user's message is shown in the image."}
POST_LABELS = {"uses_emoji": "Uses emoji", "formal_greeting": "Formal greeting",
               "mentions_deadline": "Mentions deadline", "long_message": "Long message"}
POST_PHRASES = {"uses emoji": ("uses_emoji", True), "uses no emoji": ("uses_emoji", False),
                "opens with a formal greeting": ("formal_greeting", True), "opens without a greeting": ("formal_greeting", False),
                "mentions a deadline": ("mentions_deadline", True), "mentions no deadline": ("mentions_deadline", False),
                "is long": ("long_message", True), "is short": ("long_message", False)}
ITEM = re.compile(r"^(?P<name>\w+): \$(?P<price>\d+), rated (?P<rating>[\d.]+), (?P<distance_km>\d+) km away$")
ORD = re.compile(r"^Incident (?P<id>\d+): (?P<users>\d+) users affected\. Duration: (?P<minutes>\d+) minutes\. "
                 r"Data loss: (?P<loss>yes|no)\. Workaround available: (?P<workaround>yes|no)\.$")
POST = re.compile(r"^Message (?P<id>\d+): the message (?P<features>.*)\.$")


def _load_json(state):
    try:
        value = json.loads(state)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def document(row):
    """(text state, title, header or None, rows) for a Phase 1 row. A `header` of None with one single-cell row is
    a free-text note (study utterances and Luna paraphrases, whose fields are not parsed back out)."""
    state, family = row["state"], row["family"]
    data = _load_json(state)
    if family == "rel":
        items = []
        for option in row["questions"]["rel:pick"]["criteria"].values():
            item = _load_json(option) or ITEM.match(option).groupdict()
            items.append([str(item["name"]), f"${item['price']}", f"{float(item['rating']):.1f}", f"{item['distance_km']} km"])
        if data:
            order, request = data["order_id"], data["customer_request"]
        else:
            order, request = state.split(". ", 1)
            order = order.split()[1]
        return request, f"Order {order}", ["Option", "Price", "Rating", "Distance"], items
    if family == "ord":
        if data:
            fields = {"id": data["incident_id"], "users": data["users_affected"], "minutes": data["duration_minutes"],
                      "loss": "yes" if data["data_loss"] else "no", "workaround": "yes" if data["workaround_available"] else "no"}
        elif (match := ORD.match(state)):
            fields = match.groupdict()
        else:  # paraphrased prose: render the sentence itself
            return FRAMING["ord"], "Incident report", None, [[state]]
        return FRAMING["ord"], f"Incident {fields['id']}", None, [
            ["Users affected", str(fields["users"])], ["Duration", f"{fields['minutes']} minutes"],
            ["Data loss", fields["loss"]], ["Workaround available", fields["workaround"]]]
    if family == "post":
        if data:
            message, features = data["message_id"], data["message_features"]
        elif (match := POST.match(state)) and all(p.strip() in POST_PHRASES for p in match["features"].split(",")):
            message = match["id"]
            features = dict(POST_PHRASES[p.strip()] for p in match["features"].split(","))
        else:
            return FRAMING["post"], "Message features", None, [[state]]
        return FRAMING["post"], f"Message {message}", None, [[label, "yes" if features[key] else "no"] for key, label in POST_LABELS.items()]
    return FRAMING["study"], "Message", None, [[state]]


def _font(name, size):
    return ImageFont.truetype(str(FONT_DIR / FONTS[name]), size)


def _layout(rng, note):
    """Deterministic style choices: renderer, font family, size and canvas width."""
    if note:
        return {"style": "note", "font": rng.choice(("sans", "serif", "mono")), "size": rng.choice((18, 20, 22)), "width": rng.choice(WIDTHS)}
    style = rng.choice(("table", "receipt", "list"))
    font = "mono" if style == "receipt" else rng.choice(("sans", "serif"))
    return {"style": style, "font": font, "size": rng.choice((17, 19, 21)), "width": rng.choice(WIDTHS)}


def render(title, header, rows, layout):
    """Draw the document; returns a PIL image whose long side lies in [MIN_SIDE, MAX_SIDE]."""
    size, width, style = layout["size"], layout["width"], layout["style"]
    while True:
        image = _draw(title, header, rows, style, layout["font"], size, width)
        if image is not None and image.height <= MAX_SIDE:
            return image
        size -= 1  # ponytail: shrink the type until a long note or wide table fits; Phase 1 needs a step or two at most


def _draw(title, header, rows, style, font_name, size, width):
    bold = {"sans": "bold", "serif": "serifbold", "mono": "monobold"}[font_name]
    body, head = _font(font_name, size), _font(bold, size + 2)
    margin, line = 24, int(size * 1.7)
    columns = len(rows[0]) if header or len(rows[0]) > 1 else 1
    if style == "note" or columns == 1:
        chars = max(20, int((width - 2 * margin) / (size * 0.58)))
        lines = [wrapped for row in rows for wrapped in textwrap.wrap(row[0], chars) or [""]]
        height = 2 * margin + line * (len(lines) + 2)
        image = Image.new("RGB", (width, max(height, MIN_SIDE if width < MIN_SIDE else 0)), "white")
        draw = ImageDraw.Draw(image)
        draw.text((margin, margin), title, font=head, fill="black")
        for i, text in enumerate(lines):
            draw.text((margin, margin + line * (i + 2)), text, font=body, fill=(30, 30, 30))
        return image
    table = ([header] if header else []) + rows
    height = 2 * margin + line * (len(table) + 2) + (8 if style == "table" else 0)
    image = Image.new("RGB", (width, height), "white" if style != "receipt" else (250, 248, 240))
    draw = ImageDraw.Draw(image)
    y = margin
    if style == "receipt":
        draw.text((width / 2, y), title, font=head, fill="black", anchor="ma")
        y += line
        draw.text((margin, y), "-" * int((width - 2 * margin) / (size * 0.6)), font=body, fill="black")
        y += line
        for cells in table:
            left, right = cells[0], "  ".join(cells[1:])
            draw.text((margin, y), left, font=body, fill="black")
            draw.text((width - margin, y), right, font=body, fill="black", anchor="ra")
            y += line
        return image
    draw.text((margin, y), title, font=head, fill="black")
    y += line + 8
    if style == "list":
        for cells in rows:
            text = " - ".join(cells) if header is None else f"{cells[0]}: " + ", ".join(f"{h.lower()} {c}" for h, c in zip(header[1:], cells[1:]))
            draw.text((margin, y), "• " + text, font=body, fill=(30, 30, 30))
            y += line
        return image
    inner = width - 2 * margin
    widths = [max(head.getlength(cells[c]) for cells in table) + 20 for c in range(columns)]
    if sum(widths) > inner:
        return None  # too wide for this type size; the caller shrinks it
    widths = [w + (inner - sum(widths)) / columns for w in widths]  # spread the slack evenly
    x_edges = [margin + int(sum(widths[:i])) for i in range(columns + 1)]
    for r, cells in enumerate(table):
        top = y + line * r
        if header and r == 0:
            draw.rectangle((margin, top, width - margin, top + line), fill=(225, 230, 240))
        draw.line((margin, top, width - margin, top), fill=(120, 120, 120))
        for c, cell in enumerate(cells):
            font = head if header and r == 0 else body
            draw.text((x_edges[c] + 8, top + (line - size) / 2), cell, font=font, fill="black")
    bottom = y + line * len(table)
    draw.line((margin, bottom, width - margin, bottom), fill=(120, 120, 120))
    for x in x_edges:
        draw.line((x, y, x, bottom), fill=(120, 120, 120))
    return image


def image_name(group_id):
    return "images/" + group_id.replace(":", "_") + ".png"  # colons are not portable file-name characters


def render_row(row, output):
    """Render one Phase 1 row; returns the image-state row. Deterministic in the group id."""
    text, title, header, rows = document(row)
    rng = random.Random(f"render:{row['group_id']}")
    layout = _layout(rng, header is None and len(rows) == 1 and len(rows[0]) == 1)
    image = render(title, header, rows, layout).convert("P", palette=Image.Palette.ADAPTIVE, colors=16)
    name = image_name(row["group_id"])
    path = Path(output) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, optimize=True)
    return {"state": {"text": text, "images": [{"path": name}]}, "questions": row["questions"], "group_id": row["group_id"],
            "tier": row["tier"], "family": row["family"], "style": row["style"], "render_style": layout["style"],
            "width": image.width, "height": image.height}


def prepare_images(phase1, output, splits=SPLITS):
    phase1, output = Path(phase1), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"dataset": DATASET, "source": str(phase1), "source_files": json.loads((phase1 / "manifest.json").read_text())["files"],
                "counts": {}, "render_styles": {}, "megabytes": {}}
    paired, total = [], 0
    for split in splits:
        rows = [json.loads(line) for line in (phase1 / f"{split}.jsonl").read_text().splitlines() if line.strip()]
        out = [render_row(row, output) for row in rows]
        write_jsonl(output / f"{split}.jsonl", out)
        counts, styles, size = {}, {}, 0
        for index, (row, image) in enumerate(zip(rows, out)):
            counts[row["family"]] = counts.get(row["family"], 0) + 1
            styles[image["render_style"]] = styles.get(image["render_style"], 0) + 1
            size += (output / image["state"]["images"][0]["path"]).stat().st_size
            paired.append({"split": split, "index": index, "group_id": row["group_id"], "family": row["family"],
                           "text_state_sha256": state_hash(row["state"]), "image": image["state"]["images"][0]["path"],
                           "render_style": image["render_style"], "width": image["width"], "height": image["height"]})
        manifest["counts"][split], manifest["render_styles"][split] = counts, styles
        manifest["megabytes"][split] = round(size / 1e6, 2)
        total += size
    write_jsonl(output / "paired.jsonl", paired)
    manifest["megabytes"]["total"] = round(total / 1e6, 2)
    manifest["files"] = {p.name: file_hash(p) for p in sorted(output.glob("*.jsonl"))}
    write_json(output / "manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phase1", default="data/phase1-v1")
    parser.add_argument("--output", default="data/phase1-image-v1")
    args = parser.parse_args(argv)
    manifest = prepare_images(args.phase1, args.output)
    print(json.dumps({k: manifest[k] for k in ("counts", "render_styles", "megabytes")}, indent=2))


if __name__ == "__main__":
    main()
