#!/usr/bin/env python3
"""Гейт уникальности: насколько готовые ролики похожи друг на друга для Meta.

Считает то же, что видео-хэш Meta vPDQ (ThreatExchange): из ролика берётся по
кадру в секунду, каждый кадр сворачивается в 256-битный PDQ-хэш, и два ролика
сравниваются по доле кадров, у которых в другом ролике нашёлся «двойник».

Пороги — из открытой документации Meta (vpdq/README.md, PDQ hashing.pdf):
  * кадры совпадают, если расстояние Хэмминга между хэшами ≤ 31 бита из 256
    (у случайной пары в среднем 128);
  * кадры с качеством PDQ < 50 (почти однотонные: чёрный экран, заливка) в
    сравнение не идут — у них хэш неустойчив;
  * Meta в примере объявляет ролики копией при ≥ 80% совпавших кадров.

Наш порог жёстче: пара НЕ ПРОХОДИТ, если совпал хотя бы один кадр (``--fail-pct
0``). Как именно Instagram взвешивает частичные совпадения, неизвестно, поэтому
целимся в ноль, а не в «меньше, чем у Meta».

Официальный пакет ``vpdq`` на macOS не собирается, поэтому сравнение повторено
здесь поверх ``pdqhash`` (тот же алгоритм PDQ, обёртка над эталонным C++).

Usage:
  python qa_dedup.py output/bot_уник                # все пары в папке
  python qa_dedup.py output/bot_уник --against output/bot_тираж
  python qa_dedup.py output/bot_уник --fail-pct 80  # порог Meta как есть
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "output" / "_dedup_cache"

MATCH_DISTANCE = 31     # Meta: PDQ DefaultMatchThreshold
QUALITY_MIN = 50        # Meta vPDQ: quality filter tolerance
META_COPY_PCT = 80.0    # Meta vPDQ: пример порога «это копия»
SAMPLE_FPS = 1.0        # Meta vPDQ: «кадр раз в секунду»
FRAME_W, FRAME_H = 270, 480   # PDQ всё равно сводит кадр к 64×64


@dataclass
class VideoHashes:
    path: Path
    bits: np.ndarray        # (n, 256) uint8 — только кадры с качеством ≥ QUALITY_MIN
    total: int              # сколько кадров было до фильтра качества


def _cache_file(path: Path) -> Path:
    st = path.stat()
    key = f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}|{SAMPLE_FPS}"
    return CACHE / (hashlib.sha1(key.encode()).hexdigest() + ".json")


def hash_video(path: Path) -> tuple[list[list[int]], list[int]]:
    """PDQ каждого секундного кадра: (хэши как списки бит, качества)."""
    import pdqhash

    cache = _cache_file(path)
    if cache.exists():
        data = json.loads(cache.read_text())
        return data["bits"], data["quality"]
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-vf", f"fps={SAMPLE_FPS},scale={FRAME_W}:{FRAME_H}",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True)
    frame = FRAME_W * FRAME_H * 3
    raw = proc.stdout
    bits, quality = [], []
    for off in range(0, len(raw) - frame + 1, frame):
        img = np.frombuffer(raw, np.uint8, frame, off).reshape(FRAME_H, FRAME_W, 3)
        h, q = pdqhash.compute(np.ascontiguousarray(img))
        bits.append([int(b) for b in h])
        quality.append(int(q))
    CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"path": str(path), "bits": bits, "quality": quality}))
    return bits, quality


def load(path: Path) -> VideoHashes:
    bits, quality = hash_video(path)
    arr = np.array(bits, dtype=np.uint8).reshape(-1, 256)
    keep = np.array(quality) >= QUALITY_MIN if quality else np.zeros(0, bool)
    return VideoHashes(path, arr[keep], len(bits))


def load_many(paths: list[Path], jobs: int = 8) -> list[VideoHashes]:
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        return list(ex.map(load, paths))


def distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Матрица расстояний Хэмминга между кадрами a и b."""
    if not len(a) or not len(b):
        return np.full((len(a), len(b)), 256, dtype=np.int32)
    a = a.astype(np.int32)
    b = b.astype(np.int32)
    # Число различающихся бит = |a| + |b| − 2·(a·b) для 0/1-векторов.
    return a.sum(1)[:, None] + b.sum(1)[None, :] - 2 * (a @ b.T)


@dataclass
class PairResult:
    a: Path
    b: Path
    pct_a: float        # доля кадров A, у которых есть двойник в B
    pct_b: float
    min_bits: int       # ближайшая пара кадров
    median_bits: float  # медиана «расстояния до ближайшего» по кадрам A

    @property
    def worst(self) -> float:
        return max(self.pct_a, self.pct_b)


def compare(a: VideoHashes, b: VideoHashes,
            threshold: int = MATCH_DISTANCE) -> PairResult:
    d = distances(a.bits, b.bits)
    if d.size == 0:
        return PairResult(a.path, b.path, 0.0, 0.0, 256, 256.0)
    hit = d <= threshold
    pct_a = 100.0 * hit.any(1).mean()
    pct_b = 100.0 * hit.any(0).mean()
    nearest = d.min(1)
    return PairResult(a.path, b.path, pct_a, pct_b, int(d.min()), float(np.median(nearest)))


def all_pairs(videos: list[VideoHashes], against: list[VideoHashes] | None = None,
              threshold: int = MATCH_DISTANCE) -> list[PairResult]:
    out = []
    if against is None:
        for i in range(len(videos)):
            for j in range(i + 1, len(videos)):
                out.append(compare(videos[i], videos[j], threshold))
    else:
        for v in videos:
            for r in against:
                out.append(compare(v, r, threshold))
    return out


def videos_in(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.suffix.lower() in {".mp4", ".mov"} and not p.name.startswith("."))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="папка с готовыми роликами")
    ap.add_argument("--against", type=Path, action="append",
                    help="сравнивать с роликами из этой папки (можно несколько раз)")
    ap.add_argument("--fail-pct", type=float, default=0.0,
                    help="пара не проходит, если совпало больше этой доли кадров, %% "
                         "(по умолчанию 0 — ни одного общего кадра)")
    ap.add_argument("--distance", type=int, default=MATCH_DISTANCE,
                    help="кадры совпадают при расстоянии ≤ N бит (Meta: 31)")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--report", type=Path, help="записать все пары в JSON")
    ap.add_argument("--show", type=int, default=20, help="сколько худших пар показать")
    args = ap.parse_args()

    paths = videos_in(args.folder)
    if not paths:
        print(f"В {args.folder} нет роликов")
        return 1
    print(f"Хэширую {len(paths)} роликов (кадр в секунду, PDQ)…", flush=True)
    videos = load_many(paths, args.jobs)
    against = None
    if args.against:
        ref_paths = [p for f in args.against for p in videos_in(f)]
        print(f"…и {len(ref_paths)} роликов для сравнения", flush=True)
        against = load_many(ref_paths, args.jobs)

    pairs = all_pairs(videos, against, args.distance)
    pairs.sort(key=lambda p: (-p.worst, p.min_bits))
    failed = [p for p in pairs if p.worst > args.fail_pct]
    copies = [p for p in pairs if p.worst >= META_COPY_PCT]

    print(f"\nПар проверено: {len(pairs)}")
    print(f"Порог кадра: ≤{args.distance} бит · порог пары: >{args.fail_pct:g}% совпавших кадров")
    print(f"Meta сочла бы копией (≥{META_COPY_PCT:g}%): {len(copies)}")
    print(f"НЕ ПРОШЛИ: {len(failed)}")
    if pairs:
        worst = np.array([p.worst for p in pairs])
        mins = np.array([p.min_bits for p in pairs])
        print(f"Совпавших кадров: медиана {np.median(worst):.0f}%, максимум {worst.max():.0f}%")
        print(f"Ближайшие кадры: минимум {mins.min()} бит, медиана {np.median(mins):.0f} бит")
    for p in pairs[:args.show]:
        mark = "FAIL" if p.worst > args.fail_pct else " ok "
        print(f"  {mark} {p.a.name:<22} ↔ {p.b.name:<22} "
              f"{p.pct_a:5.1f}% / {p.pct_b:5.1f}%  мин {p.min_bits:3d} бит")

    if args.report:
        args.report.write_text(json.dumps([
            {"a": p.a.name, "b": p.b.name, "pct_a": round(p.pct_a, 1),
             "pct_b": round(p.pct_b, 1), "min_bits": p.min_bits,
             "median_bits": p.median_bits} for p in pairs], ensure_ascii=False, indent=1))
        print(f"\nОтчёт: {args.report}")
    print("\nИТОГ:", "PASS" if not failed else f"FAIL ({len(failed)} пар)")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
