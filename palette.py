"""Подбор набора «обликов» (рамка ± наклон) под конкретный материал.

Облик = (приближение z, сдвиг ox, oy, наклон t) — в ДОЛЯХ кадра, поэтому один
и тот же облик годится для любого размера и соотношения сторон. Кадр w×h
увеличивается до w·z × h·z (пропорции сохраняются), поворачивается на t вокруг
центра, и из него вырезается окно w×h с левым верхним углом в (ox·w, oy·h).
Выход — того же размера, что и исходник: вертикальный остаётся вертикальным,
горизонтальный — горизонтальным.

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

ZOOMS = (1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.10, 1.11, 1.12, 1.13, 1.14)
# Шаг сдвига окна в долях кадра (на 1080×1920 — 12 и 24 px, как прежде).
X_STEP, Y_STEP = 12 / 1080, 24 / 1920
TILTS = (-0.5, 0.5)
SAMPLE_WIDTH = 540        # кадры выборки — в этой ширине, пропорции свои
PALETTE_VERSION = 3       # меняется вместе с правилами перебора — сбрасывает кэш


@dataclass(frozen=True)
class Limits:
    """Что нельзя срезать — в долях кадра.

    ``text_box`` — прямоугольник титров (l, t, r, b), None — титры не защищаются.
    ``max_top``/``max_bottom``/``max_side`` — сколько кадра можно потерять с
    каждой стороны.
    """
    text_box: tuple[float, float, float, float] | None = None
    text_margin: float = 0.0
    max_top: float = 1.0
    max_bottom: float = 1.0
    max_side: float = 1.0


def even(v: float) -> int:
    return max(2, round(v / 2) * 2)


def to_out(u: float, v: float, look: tuple, aspect: float) -> tuple[float, float]:
    """Точка исходного кадра (доли u, v) → где она окажется в готовом кадре (доли).

    Считается в единицах «высота кадра = 1, ширина = aspect».
    """
    z, ox, oy, t = look
    w, h = aspect, 1.0
    sx, sy = u * w * z, v * h * z
    cx, cy = w * z / 2, h * z / 2
    th = math.radians(t)
    dx, dy = sx - cx, sy - cy
    x = cx + dx * math.cos(th) - dy * math.sin(th) - ox * w
    y = cy + dx * math.sin(th) + dy * math.cos(th) - oy * h
    return x / w, y / h


def no_black_corners(look: tuple, aspect: float) -> bool:
    """Углы окна лежат внутри повёрнутого увеличенного кадра."""
    z, ox, oy, t = look
    w, h = aspect, 1.0
    if ox < 0 or oy < 0 or ox + 1 > z + 1e-9 or oy + 1 > z + 1e-9:
        return False
    cx, cy = w * z / 2, h * z / 2
    th = math.radians(-t)
    eps = 1e-3 * h
    for qx, qy in ((ox * w, oy * h), ((ox + 1) * w, oy * h),
                   (ox * w, (oy + 1) * h), ((ox + 1) * w, (oy + 1) * h)):
        dx, dy = qx - cx, qy - cy
        ux = dx * math.cos(th) - dy * math.sin(th)
        uy = dx * math.sin(th) + dy * math.cos(th)
        if abs(ux) > w * z / 2 - eps or abs(uy) > h * z / 2 - eps:
            return False
    return True


def is_safe(look: tuple, lim: Limits, aspect: float) -> bool:
    if not no_black_corners(look, aspect):
        return False
    if lim.text_box is not None:
        l, tp, r, b = lim.text_box
        for u in (l, r):
            for v in (tp, b):
                x, _ = to_out(u, v, look, aspect)
                if not lim.text_margin <= x <= 1 - lim.text_margin:
                    return False
    top = -to_out(0.5, 0, look, aspect)[1]
    bottom = to_out(0.5, 1, look, aspect)[1] - 1
    left = -to_out(0, 0.5, look, aspect)[0]
    right = to_out(1, 0.5, look, aspect)[0] - 1
    return (top <= lim.max_top + 1e-9 and bottom <= lim.max_bottom + 1e-9
            and left <= lim.max_side + 1e-9 and right <= lim.max_side + 1e-9)


def candidates(tilt: bool, lim: Limits, aspects: list[float]) -> list[tuple]:
    """Облики, безопасные для ВСЕХ соотношений сторон партии."""
    tilts = (0.0,) + (TILTS if tilt else ())
    out = []
    for t in tilts:
        for z in ZOOMS:
            nx = int((z - 1) / X_STEP + 1e-9)
            ny = int((z - 1) / Y_STEP + 1e-9)
            for i in range(nx + 1):
                for j in range(ny + 1):
                    look = (z, round(i * X_STEP, 5), round(j * Y_STEP, 5), t)
                    if all(is_safe(look, lim, a) for a in aspects):
                        out.append(look)
    return out


# ── хэши облика на выборке кадров ─────────────────────────────────────────────
_FRAMES: list = []


def video_dims(path: Path) -> tuple[int, int]:
    """Размер кадра с учётом поворота из метаданных (как его покажет плеер)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height:stream_side_data=rotation", "-of", "json", str(path)],
        capture_output=True, text=True).stdout
    st = json.loads(out or "{}").get("streams", [{}])[0]
    w, h = int(st.get("width", 0)), int(st.get("height", 0))
    rot = next((abs(int(sd.get("rotation", 0))) for sd in st.get("side_data_list", [])
                if "rotation" in sd), 0)
    return (h, w) if rot % 180 == 90 else (w, h)


def sample_frames(videos: list[Path], per_video: int = 12) -> list[np.ndarray]:
    """По per_video кадров из каждого ролика, равномерно по длине, ширина 540."""
    frames = []
    for v in videos:
        vw, vh = video_dims(v)
        if not vw or not vh:
            continue
        w, h = SAMPLE_WIDTH, even(SAMPLE_WIDTH * vh / vw)
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


def apply_look(im, look: tuple):
    """Облик на PIL-картинке — та же геометрия, что у ffmpeg в рендере."""
    from PIL import Image

    z, ox, oy, t = look
    w0, h0 = im.size
    w, h = even(w0 * z), even(h0 * z)
    im = im.resize((w, h), Image.BILINEAR)
    if t:
        # PIL крутит против часовой, ffmpeg rotate — по часовой.
        im = im.rotate(-t, resample=Image.BILINEAR, center=(w / 2, h / 2))
    x, y = round(ox * w0), round(oy * h0)
    return im.crop((x, y, x + w0, y + h0))


def _hash_look(look: tuple) -> np.ndarray:
    import pdqhash
    from PIL import Image

    out = []
    for a in _FRAMES:
        im = apply_look(Image.fromarray(a), look)
        hb, _ = pdqhash.compute(np.asarray(im))
        out.append(hb.astype(np.int32))
    return np.array(out)


def search(videos: list[Path], tilt: bool, lim: Limits, aspects: list[float], cache_dir: Path,
           jobs: int = 12, tries: int = 3000, threshold: int = 31) -> list[tuple]:
    """Максимальный найденный набор попарно совместимых обликов (кэшируется)."""
    key = hashlib.sha1(json.dumps([PALETTE_VERSION, sorted(map(str, videos)), tilt, lim.__dict__,
                                   sorted(round(a, 4) for a in aspects), ZOOMS, X_STEP, Y_STEP,
                                   TILTS, threshold]).encode()).hexdigest()
    cache = cache_dir / f"palette_{key}.json"
    if cache.exists():
        return [tuple(x) for x in json.loads(cache.read_text())]

    cands = [(1.0, 0.0, 0.0, 0.0)] + candidates(tilt, lim, aspects)   # первый — оригинал
    if len(cands) == 1:
        return []
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
