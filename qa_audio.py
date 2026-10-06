#!/usr/bin/env python3
"""Гейт звука: насколько похожи звуковые дорожки роликов.

Меряет открытым звуковым отпечатком Chromaprint (AcoustID; ``fpcalc`` из
``brew install chromaprint``). Как звук сравнивает Instagram, не публикуется —
это лучший доступный открытый аналог, как PDQ для картинки.

Отпечаток — 32-битное слово примерно на каждые 0.12 с. Два ролика
выравниваются по лучшему сдвигу (±15 с), и считается доля слов, отличающихся
не больше чем на 6 бит из 32, — «совпавшие отрезки». Замер на «бот работ»:
два разных ролика — ~1%, ролики с общим хуком и советом — ~90%.

Что проверяется:
  * каждый новый ролик против СВОЕГО оригинала (``--originals``, по имени файла);
  * новые ролики между собой — пары с общим куском в имени (h3_…, …_t7_…, …_c4).
    Все пары подряд дороги (100 тыс. на 450 роликов), а звук делят как раз они.

Порог — ``config.yaml → gate.audio_fail_pct``; разово ``--fail-pct``.

Usage:
  python qa_audio.py output/bot_уник/v2/с\\ мемами --originals "raw/с мемами"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "output" / "_audio_cache"
MAX_SHIFT = 120          # слов ≈ 15 с
BIT_TOLERANCE = 6        # слово «совпало», если отличается ≤ 6 бит из 32
MIN_OVERLAP = 40

_POP = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.int32)


def popcount(x: np.ndarray) -> np.ndarray:
    return _POP[x & 0xFFFF] + _POP[(x >> 16) & 0xFFFF]


def fingerprint(path: Path) -> np.ndarray:
    st = path.stat()
    key = hashlib.sha1(f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()
    cache = CACHE / f"{key}.json"
    if cache.exists():
        return np.array(json.loads(cache.read_text()), dtype=np.int64)
    out = subprocess.run(["fpcalc", "-raw", "-length", "600", str(path)],
                         capture_output=True, text=True).stdout
    line = next((ln for ln in out.splitlines() if ln.startswith("FINGERPRINT=")), "FINGERPRINT=")
    values = [int(x) & 0xFFFFFFFF for x in line.split("=", 1)[1].split(",") if x]
    CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(values))
    return np.array(values, dtype=np.int64)


def matched_pct(a: np.ndarray, b: np.ndarray) -> float:
    """Доля совпавших отрезков при лучшем сдвиге (от более короткого ролика)."""
    if len(a) < MIN_OVERLAP or len(b) < MIN_OVERLAP:
        return 0.0
    best = 0.0
    for off in range(-MAX_SHIFT, MAX_SHIFT + 1):
        x, y = (a[off:], b) if off >= 0 else (a, b[-off:])
        n = min(len(x), len(y))
        if n < MIN_OVERLAP:
            continue
        hit = (popcount(x[:n] ^ y[:n]) <= BIT_TOLERANCE).sum()
        best = max(best, hit / min(len(a), len(b)))
    return 100.0 * best


def default_fail_pct() -> float:
    import yaml

    try:
        return float(yaml.safe_load((ROOT / "config.yaml").read_text())["gate"]["audio_fail_pct"])
    except (OSError, KeyError, TypeError, ValueError):
        return 30.0


BLOCK = re.compile(r"^[a-zA-Z]+\d+(?:_[a-zA-Z]+\d+)+$")


def shares_block(a: str, b: str) -> bool:
    if not (BLOCK.match(a) and BLOCK.match(b)):
        return False
    return bool(set(a.lower().split("_")) & set(b.lower().split("_")))


def videos_in(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.suffix.lower() in {".mp4", ".mov"} and not p.name.startswith("."))


def label(p: Path) -> str:
    return f"{p.parent.name}/{p.name}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, nargs="+", help="папки с новыми роликами")
    ap.add_argument("--originals", type=Path, action="append", default=[],
                    help="папки с оригиналами — новый ролик сравнивается со своим по имени")
    ap.add_argument("--fail-pct", type=float, default=default_fail_pct(),
                    help="пара не проходит при > N%% совпавших отрезков (по умолчанию %(default)g)")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--show", type=int, default=15)
    args = ap.parse_args()

    new = [p for f in args.folder for p in videos_in(f)]
    originals = {p.name: p for f in args.originals for p in videos_in(f)}
    print(f"Звуковые отпечатки: {len(new)} новых" +
          (f" + {len(originals)} оригиналов" if originals else "") + "…", flush=True)
    with ThreadPoolExecutor(args.jobs) as ex:
        fps = dict(zip(new, ex.map(fingerprint, new)))
        ofps = dict(zip(originals.values(), ex.map(fingerprint, originals.values())))

    # Ролик без звуковой дорожки (у сырых файлов это обычное дело) звуком ни на
    # кого не похож — в сравнение не идёт, но перечисляется.
    silent = [p for p in new if len(fps[p]) == 0]
    if silent:
        print(f"Без звука (пропущены): {len(silent)}: " + ", ".join(label(p) for p in silent[:10])
              + (" …" if len(silent) > 10 else ""))
    new = [p for p in new if len(fps[p])]
    rows = []
    for p in new:
        o = originals.get(p.name)
        if o is not None and len(ofps[o]):
            rows.append(("оригинал", label(p), label(o), matched_pct(fps[p], ofps[o])))
    pairs = [(a, b) for i, a in enumerate(new) for b in new[i + 1:] if shares_block(a.stem, b.stem)]
    print(f"Пар с общим куском: {len(pairs)}", flush=True)
    with ThreadPoolExecutor(args.jobs) as ex:
        pcts = list(ex.map(lambda ab: matched_pct(fps[ab[0]], fps[ab[1]]), pairs))
    rows += [("между собой", label(a), label(b), pct) for (a, b), pct in zip(pairs, pcts)]

    rc = 0
    for kind in ("оригинал", "между собой"):
        sel = sorted((r for r in rows if r[0] == kind), key=lambda r: -r[3])
        if not sel:
            continue
        vals = np.array([r[3] for r in sel])
        failed = (vals > args.fail_pct).sum()
        rc |= int(failed > 0)
        title = "против своего оригинала" if kind == "оригинал" else "новые между собой (общий кусок)"
        print(f"\n── звук: {title} ──")
        print(f"Пар: {len(sel)} · совпавших отрезков: медиана {np.median(vals):.0f}%, "
              f"максимум {vals.max():.0f}% · порог >{args.fail_pct:g}% · НЕ ПРОШЛИ: {failed}")
        for r in sel[:args.show]:
            mark = "FAIL" if r[3] > args.fail_pct else " ok "
            print(f"  {mark} {r[1]:<34} ↔ {r[2]:<34} {r[3]:5.1f}%")
    if args.report:
        args.report.write_text(json.dumps(
            [{"kind": k, "a": a, "b": b, "pct": round(v, 1)} for k, a, b, v in rows],
            ensure_ascii=False, indent=1))
    print("\nИТОГ (звук):", "PASS" if rc == 0 else "FAIL")
    return 2 if rc else 0


if __name__ == "__main__":
    raise SystemExit(main())
