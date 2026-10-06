"""Подбор набора «обликов» (рамка ± наклон) под конкретный материал.

Облик = (приближение z, окно x, y, наклон t). Кадр увеличивается до W·z × H·z,
поворачивается на t вокруг центра и из него вырезается окно 1080×1920 в (x, y).

Набор подбирается перебором ПО КАДРАМ САМИХ ИСХОДНИКОВ: два облика совместимы,
если ни один кадр выборки, показанный в одном облике, не ближе 31 бита PDQ к
любому кадру выборки в другом. Жадный поиск максимального набора попарно
совместимых обликов — от этого числа зависит, сколько роликов можно развести.

Почему не константа: на «бот работ» без мемов из партии 100 набор был 33 облика,
с одним её роликом в выборке — 24. Чем больше однотонных кадров (экран, мем на
тёмном фоне), тем меньше их сдвиг меняет хэш и тем меньше обликов.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

W, H = 1080, 1920
# Геометрия считается на половинном кадре: PDQ всё равно сводит кадр к 64×64,
# а перебор так идёт вчетверо быстрее.
SCALE = 0.5
ZOOMS = (1.05, 1.06, 1.07, 1.08, 1.09, 1.10, 1.11, 1.12, 1.13, 1.14)
X_STEP, Y_STEP = 12, 24
TILTS = (-0.5, 0.5)
PALETTE_VERSION = 2     # меняется вместе с правилами перебора — сбрасывает кэш


@dataclass(frozen=True)
class Limits:
    text_box: tuple[int, int, int, int] = (95, 1306, 994, 1619)   # титры: l, t, r, b
    text_margin: int = 8
    max_top: int = 150          # сколько можно срезать сверху, px
    max_bottom: int = 230


def scaled_size(z: float) -> tuple[int, int]:
    return round(W * z / 2) * 2, round(H * z / 2) * 2


def to_out(px: float, py: float, look: tuple) -> tuple[float, float]:
    """Точка исходного кадра → где она окажется в готовом кадре."""
    z, x, y, t = look
    w, h = scaled_size(z)
    sx, sy = px * w / W, py * h / H
    cx, cy = w / 2, h / 2
    th = math.radians(t)
    dx, dy = sx - cx, sy - cy
    return (cx + dx * math.cos(th) - dy * math.sin(th) - x,
            cy + dx * math.sin(th) + dy * math.cos(th) - y)


def no_black_corners(look: tuple) -> bool:
    """Углы окна лежат внутри повёрнутого увеличенного кадра."""
    z, x, y, t = look
    w, h = scaled_size(z)
    if x < 0 or y < 0 or x + W > w or y + H > h:
        return False
    cx, cy = w / 2, h / 2
    th = math.radians(-t)
    for qx, qy in ((x, y), (x + W, y), (x, y + H), (x + W, y + H)):
        dx, dy = qx - cx, qy - cy
        ux = dx * math.cos(th) - dy * math.sin(th)
        uy = dx * math.sin(th) + dy * math.cos(th)
        if abs(ux) > w / 2 - 1 or abs(uy) > h / 2 - 1:
            return False
    return True


def is_safe(look: tuple, lim: Limits) -> bool:
    if not no_black_corners(look):
        return False
    l, tp, r, b = lim.text_box
    for px in (l, r):
        for py in (tp, b):
            u, _ = to_out(px, py, look)
            if not lim.text_margin <= u <= W - lim.text_margin:
                return False
    top = -to_out(W / 2, 0, look)[1]
    bottom = -(H - to_out(W / 2, H, look)[1])
    return top <= lim.max_top and bottom <= lim.max_bottom


def candidates(tilt: bool, lim: Limits) -> list[tuple]:
    tilts = (0.0,) + (TILTS if tilt else ())
    out = []
    for t in tilts:
        for z in ZOOMS:
            w, h = scaled_size(z)
            for x in range(0, w - W + 1, X_STEP):
                for y in range(0, h - H + 1, Y_STEP):
                    look = (z, x, y, t)
                    if is_safe(look, lim):
                        out.append(look)
    return out


# ── хэши облика на выборке кадров ─────────────────────────────────────────────
_FRAMES: list = []


def sample_frames(videos: list[Path], per_video: int = 12) -> list[np.ndarray]:
    """По per_video кадров из каждого ролика, равномерно по длине, в половинном размере."""
    frames = []
    w, h = int(W * SCALE), int(H * SCALE)
    for v in videos:
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(v), "-vf", f"fps=1,scale={w}:{h}",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
        n = w * h * 3
        all_ = [np.frombuffer(raw, np.uint8, n, o).reshape(h, w, 3)
                for o in range(0, len(raw) - n + 1, n)]
        if all_:
            step = max(1, len(all_) // per_video)
            frames += all_[::step][:per_video]
    # Как в гейте (и в vPDQ): почти однотонные кадры — чёрный экран, размытый
    # переход — не сравниваются. Они совпадают при любой рамке, и без фильтра
    # один такой кадр запрещает облики, которые гейт на деле пропустил бы.
    import pdqhash
    from qa_dedup import QUALITY_MIN

    return [f for f in frames if pdqhash.compute(np.ascontiguousarray(f))[1] >= QUALITY_MIN]


def _init(frames):
    global _FRAMES
    _FRAMES = frames


def _hash_look(look: tuple) -> np.ndarray:
    import pdqhash
    from PIL import Image

    z, x, y, t = look
    w, h = (round(v * SCALE) for v in scaled_size(z))
    out = []
    for a in _FRAMES:
        im = Image.fromarray(a).resize((w, h), Image.BILINEAR)
        if t:
            # PIL крутит против часовой, ffmpeg rotate — по часовой.
            im = im.rotate(-t, resample=Image.BILINEAR, center=(w / 2, h / 2))
        sx, sy = round(x * SCALE), round(y * SCALE)
        im = im.crop((sx, sy, sx + int(W * SCALE), sy + int(H * SCALE)))
        hb, _ = pdqhash.compute(np.asarray(im.resize((270, 480), Image.BILINEAR)))
        out.append(hb.astype(np.int32))
    return np.array(out)


def search(videos: list[Path], tilt: bool, lim: Limits, cache_dir: Path,
           jobs: int = 12, tries: int = 3000, threshold: int = 31) -> list[tuple]:
    """Максимальный найденный набор попарно совместимых обликов (кэшируется)."""
    key = hashlib.sha1(json.dumps([PALETTE_VERSION, sorted(map(str, videos)), tilt, lim.__dict__,
                                   ZOOMS, X_STEP, Y_STEP, TILTS, threshold]).encode()).hexdigest()
    cache = cache_dir / f"palette_{key}.json"
    if cache.exists():
        return [tuple(x) for x in json.loads(cache.read_text())]

    cands = [(1.0, 0, 0, 0.0)] + candidates(tilt, lim)   # первый — оригинал
    frames = sample_frames(videos)
    with ProcessPoolExecutor(jobs, initializer=_init, initargs=(frames,)) as ex:
        hashes = np.stack(list(ex.map(_hash_look, cands, chunksize=4)))
    sums = hashes.sum(2)
    k = len(cands)
    bad = np.zeros((k, k), bool)
    for i in range(k):
        d = sums[i][None, :, None] + sums[:, None, :] - 2 * np.einsum("nb,kmb->knm", hashes[i], hashes)
        bad[i] = (d <= threshold).any((1, 2))
    np.fill_diagonal(bad, False)
    pool = [i for i in range(1, k) if not bad[0, i]]     # далеко от оригинала
    best: list[int] = []
    for seed in range(tries):
        rnd = random.Random(seed)
        order = pool[:]
        rnd.shuffle(order)
        chosen: list[int] = []
        for i in order:
            if not bad[i, chosen].any():
                chosen.append(i)
        if len(chosen) > len(best):
            best = chosen
    result = sorted(cands[i] for i in best)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result))
    return result
