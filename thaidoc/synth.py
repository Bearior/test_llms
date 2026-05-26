"""Synthetic sample generator.

Renders simple, clearly-synthetic document cards for a representative subset of
the canonical types, with augmentation and a deliberately *adversarial* subset.

IMPORTANT (honesty): these are renderer-generated mock documents, NOT real Thai
government documents. Any accuracy measured on them is RENDERER-BOUNDED and not
predictive of production performance. The adversarial subset (marker removed +
heavy degradation) exists specifically so the pipeline must *fail gracefully*
and route to human review — proving the routing mechanism rather than faking
reading skill.
"""
from __future__ import annotations

import csv
import random
from dataclasses import dataclass, asdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from . import labels
from .config import DATA_DIR, SEED, ensure_dirs
from .fonts import resolve_thai_font

# Demo subset: >=6 types across families, INCLUDING >=2 same-family visa
# subtypes (the near-identical hard case the whole project hinges on).
DEMO_TYPE_NAMES = [
    'Non-Immigrant VISA "B"',                       # VISA subtype 1  (same family)
    'Non-Immigrant VISA "F"',                       # VISA subtype 2  (same family)
    "หนังสือเดินทาง (Passport)",                      # PASSPORT
    "บัตรประจำตัวประชาชน (ID Card) เจ้าของสัญชาติ",     # ID_CARD
    "ทะเบียนบ้าน เล่มปกสีน้ำเงิน (ทร.14)",             # HOUSEHOLD
    "ใบอนุญาตทำงาน (Work Permit)",                    # PERMIT
    "หนังสือรับรองเงินเดือน",                          # CERTIFICATE_LETTER
]

_FAMILY_BG = {
    labels.FAMILY_VISA: (210, 225, 245),
    labels.FAMILY_PASSPORT: (220, 210, 195),
    labels.FAMILY_ID_CARD: (215, 240, 220),
    labels.FAMILY_HOUSEHOLD: (245, 235, 205),
    labels.FAMILY_PERMIT: (235, 215, 235),
    labels.FAMILY_CERT_LETTER: (235, 235, 235),
    labels.FAMILY_OTHER: (225, 225, 225),
}

IMG_W, IMG_H = 520, 360


@dataclass
class SampleRecord:
    filename: str
    physical_type: str
    family: str
    subtype_marker: str
    is_adversarial: bool
    split: str          # train | calib | test
    mock_text: str      # what a MockReader "reads" (marker stripped if adversarial)


def _wrap(text: str, width: int = 30) -> list[str]:
    out, line = [], ""
    for ch in text:
        line += ch
        if len(line) >= width and ch == " ":
            out.append(line.rstrip())
            line = ""
    if line:
        out.append(line)
    return out


def _render(dt: labels.DocumentType, font_path: str, adversarial: bool,
            rng: random.Random) -> Image.Image:
    bg = _FAMILY_BG.get(dt.family, (225, 225, 225))
    img = Image.new("RGB", (IMG_W, IMG_H), bg)
    draw = ImageDraw.Draw(img)
    title_font = ImageFont.truetype(font_path, 22)
    body_font = ImageFont.truetype(font_path, 18)
    code_font = ImageFont.truetype(font_path, 40)

    draw.rectangle([8, 8, IMG_W - 8, IMG_H - 8], outline=(90, 90, 90), width=2)
    draw.text((20, 16), f"[{dt.family}]", font=body_font, fill=(70, 70, 70))

    y = 60
    for line in _wrap(dt.display_name, 28):
        draw.text((20, y), line, font=title_font, fill=(15, 15, 15))
        y += 30

    # The distinguishing subtype code, rendered large. On adversarial samples
    # we *degrade* the whole card so this becomes unreadable.
    if dt.subtype_marker:
        draw.text((20, IMG_H - 90), "CODE", font=body_font, fill=(60, 60, 60))
        draw.text((20, IMG_H - 70), dt.subtype_marker, font=code_font, fill=(120, 0, 0))

    # Synthetic (fake) field values — no real PII.
    draw.text((230, IMG_H - 70), f"No. {rng.randint(1000000, 9999999)}",
              font=body_font, fill=(40, 40, 40))

    # --- augmentation ---
    angle = rng.uniform(-3, 3)
    img = img.rotate(angle, expand=False, fillcolor=bg)
    if adversarial:
        # Heavy degradation: blur + downscale/upscale so the CODE is lost.
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(2.2, 3.5)))
        small = img.resize((IMG_W // 4, IMG_H // 4))
        img = small.resize((IMG_W, IMG_H))
    return img


def generate(n_per_type: int = 12, seed: int = SEED) -> Path:
    """Generate the synthetic dataset and write images + manifest.csv.

    Splits per type: 60% train, ~17% calib, ~23% test. ~1/3 of the *test*
    images are adversarial (degraded + mock reader blinded to the marker).
    Returns the manifest path.
    """
    ensure_dirs()
    rng = random.Random(seed)
    font_path = resolve_thai_font()
    img_dir = DATA_DIR / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    records: list[SampleRecord] = []
    for name in DEMO_TYPE_NAMES:
        dt = labels.get_type(labels._normalize(name))
        if dt is None:
            raise ValueError(f"Demo type not found in catalog: {name!r}")
        for i in range(n_per_type):
            # split assignment
            if i < int(n_per_type * 0.60):
                split = "train"
            elif i < int(n_per_type * 0.77):
                split = "calib"
            else:
                split = "test"
            adversarial = split == "test" and (i % 3 == 0)

            img = _render(dt, font_path, adversarial, rng)
            safe = "".join(c if c.isalnum() else "_" for c in dt.physical_type)[:40]
            fname = f"{safe}_{i:03d}{'_adv' if adversarial else ''}.jpg"
            img.save(img_dir / fname, "JPEG",
                     quality=rng.randint(45, 92) if adversarial else rng.randint(75, 95))

            # MockReader text: full label normally; marker STRIPPED if adversarial.
            if adversarial and dt.subtype_marker:
                mock_text = dt.display_name.replace(f'"{dt.subtype_marker}"', '""')
            else:
                mock_text = dt.display_name + (
                    f" CODE {dt.subtype_marker}" if dt.subtype_marker else "")

            records.append(SampleRecord(
                filename=fname,
                physical_type=dt.physical_type,
                family=dt.family,
                subtype_marker=dt.subtype_marker or "",
                is_adversarial=adversarial,
                split=split,
                mock_text=mock_text,
            ))

    manifest = DATA_DIR / "manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
        w.writeheader()
        for r in records:
            w.writerow(asdict(r))
    return manifest


def load_manifest() -> list[dict]:
    manifest = DATA_DIR / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(
            f"{manifest} not found — run `python -m thaidoc.synth` first.")
    with open(manifest, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["is_adversarial"] = r["is_adversarial"] in ("True", "true", "1")
    return rows


def image_path(filename: str) -> Path:
    """Resolve an image path, confined to the images dir.

    A real manifest is an external CSV; a crafted ``filename`` (``..\\..`` or an
    absolute path) must not escape the data directory and read arbitrary files.
    """
    base = (DATA_DIR / "images").resolve()
    p = (base / filename).resolve()
    if p != base and base not in p.parents:
        raise ValueError(f"Unsafe image path outside data dir: {filename!r}")
    return p


if __name__ == "__main__":
    import io
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    p = generate()
    rows = load_manifest()
    n_adv = sum(1 for r in rows if r["is_adversarial"])
    print(f"Generated {len(rows)} images across {len(DEMO_TYPE_NAMES)} types "
          f"({n_adv} adversarial). Manifest: {p}")
