#!/usr/bin/env python3
"""Уникализатор: из папки готовых рилсов делает версии, которые Meta не сочтёт
одним и тем же роликом, и сам проверяет результат гейтом vPDQ.

Каждому ролику достаются:
  * своя РАМКА кадра — приближение 5–14% со сдвигом окна. Это главный рычаг:
    замер на «бот работ» — сдвиг окна на 22–40 px даёт 0% совпавших кадров, а
    приближение 2% по центру — 95%;
  * своя СКОРОСТЬ — ±1–3%, голос не меняет высоту. Сама по себе даёт 26–33%
    совпавших кадров, то есть только помогает рамке;
  * свой ШУМ — нарезка из noise/pack, наложена еле заметно. На хэш не влияет
    (100% совпавших кадров при шуме 25%), это косметика;
  * свой GOP и чистые метаданные.

Рамки раздаются так, чтобы любые два ролика с общим куском (имена вида
``h3_t7_c4``: общий ``h3``, ``t7`` или ``c4``) получили РАЗНЫЕ рамки — это
раскраска графа «у кого общий кусок». Ролики без общего куска могут делить
рамку: у них и так разные кадры.

Команды:
  python uniquify.py pack                               # нарезать noise/sources → noise/pack
  python uniquify.py run SRC_DIR --name bot_уник        # уникализировать + гейт
  python uniquify.py run SRC_DIR --name bot_уник --against output/bot_тираж
  python uniquify.py run SRC_DIR --name test --limit 6 --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
W, H, FPS = 1080, 1920, 30
GOLDEN = 0.6180339887498949

# ── рамки ─────────────────────────────────────────────────────────────────────
# (приближение, x, y): кадр увеличивается до W·z × H·z, и из него вырезается
# окно 1080×1920 с левым верхним углом в (x, y). Набор подобран перебором 361
# кандидата на кадрах трёх роликов «бот работ» (говорящая голова, экран демо,
# мем): у ЛЮБОЙ пары рамок ниже нет ни одного кадра ближе 31 бита PDQ, и ни одна
# не совпадает с оригиналом. Ограничения при переборе: титры (95–994 px) не
# срезаются, сверху уходит ≤150 px, снизу ≤230 px.
FRAMINGS: tuple[tuple[float, int, int], ...] = (
    (1.05, 12, 24), (1.05, 12, 96), (1.05, 36, 24), (1.06, 0, 0),
    (1.06, 0, 72), (1.06, 24, 0), (1.06, 24, 72), (1.06, 48, 0),
    (1.07, 0, 120), (1.07, 60, 72), (1.08, 48, 144), (1.08, 60, 120),
    (1.08, 72, 48), (1.08, 84, 24), (1.08, 84, 144), (1.09, 12, 0),
    (1.09, 48, 72), (1.09, 72, 0), (1.10, 24, 72), (1.10, 36, 144),
    (1.10, 60, 48), (1.10, 96, 0), (1.10, 96, 96), (1.11, 36, 0),
    (1.12, 48, 72), (1.12, 72, 0), (1.12, 72, 144), (1.12, 96, 144),
    (1.13, 60, 48), (1.13, 60, 144), (1.14, 84, 48), (1.14, 84, 120),
    (1.14, 96, 96),
)


def scaled_size(z: float) -> tuple[int, int]:
    return round(W * z / 2) * 2, round(H * z / 2) * 2


def framing_is_safe(f: tuple[float, int, int], text_left: int, text_right: int,
                    margin: int) -> bool:
    """Рамка не срезает титры и не вылезает за увеличенный кадр."""
    z, x, y = f
    w, h = scaled_size(z)
    if x < 0 or y < 0 or x + W > w or y + H > h:
        return False
    return text_left * z - x >= margin and text_right * z - x <= W - margin


# ── кто с кем делит куски ─────────────────────────────────────────────────────
BLOCK_NAME = re.compile(r"^[a-zA-Z]+\d+(?:_[a-zA-Z]+\d+)+$")


def blocks_of(stem: str) -> frozenset[str] | None:
    """Куски ролика по имени ``h3_t7_c4`` → {h3, t7, c4}; None — имя не по схеме."""
    if not BLOCK_NAME.match(stem):
        return None
    return frozenset(stem.lower().split("_"))


Weights = list[dict[int, float]]


def conflicts(stems: list[str]) -> list[set[int]]:
    """Для каждого ролика — номера роликов, с которыми у него общий кусок.

    Ролик с именем не по схеме считается похожим на все: что внутри, неизвестно.
    """
    blocks = [blocks_of(s) for s in stems]
    adj: list[set[int]] = [set() for _ in stems]
    for i in range(len(stems)):
        for j in range(i + 1, len(stems)):
            a, b = blocks[i], blocks[j]
            if a is None or b is None or a & b:
                adj[i].add(j)
                adj[j].add(i)
    return adj


# Пара с общим куском по имени, у которой кадры исходников случайно не совпали
# (секундная выборка легла между похожими кадрами), — всё равно сосед, но слабый.
NAME_ONLY_WEIGHT = 0.5


def similarity(srcs: list[Path], jobs: int) -> Weights:
    """Кто на кого похож В ИСХОДНИКАХ: вес = сколько секундных кадров совпало.

    Имени мало: на «бот работ» 4491 пара без общего куска в имени делит кадры —
    это общие мемы и кадр продукта, которые в имя не попадают. Поэтому граф
    строится по самим кадрам (тот же PDQ, что и в гейте), а имя только добавляет
    слабые рёбра там, где кадры не совпали.
    """
    from qa_dedup import MATCH_DISTANCE, distances, load_many

    videos = load_many(srcs, jobs)
    # Имя без схемы ничего не говорит о содержимом — для него хватает кадров.
    named = [blocks_of(p.stem) is not None for p in srcs]
    adj_names = conflicts([p.stem if ok else f"x{i}_y{i}" for i, (p, ok) in enumerate(zip(srcs, named))])
    w: Weights = [dict() for _ in srcs]
    for i in range(len(srcs)):
        for j in range(i + 1, len(srcs)):
            d = distances(videos[i].bits, videos[j].bits)
            hit = d <= MATCH_DISTANCE
            weight = float(max(hit.any(1).sum(), hit.any(0).sum())) if d.size else 0.0
            if not weight and j in adj_names[i]:
                weight = NAME_ONLY_WEIGHT
            if weight:
                w[i][j] = w[j][i] = weight
    return w


def assign_framings(weights: Weights, k: int) -> list[int]:
    """Раскраска DSatur в k рамок: похожие ролики получают разные рамки.

    Пока свободная рамка есть, соседи не делят рамку вовсе. Когда все k заняты
    соседями (на «бот работ» нужно 59 при 33 безопасных), ролику достаётся
    рамка, у владельцев которой с ним меньше всего общих кадров, — самые
    похожие пары разводятся, слабые совпадения остаются гейту.
    """
    n = len(weights)
    color: dict[int, int] = {}
    while len(color) < n:
        u = max((i for i in range(n) if i not in color),
                key=lambda i: (len({color[j] for j in weights[i] if j in color}),
                               sum(weights[i].values()), -i))
        cost = [0.0] * k
        for j, wt in weights[u].items():
            if j in color:
                cost[color[j]] += wt
        color[u] = min(range(k), key=lambda c: (cost[c], c))
    return [color[i] for i in range(n)]


def shared_framing_pairs(weights: Weights, colors: list[int]) -> list[tuple[int, int, float]]:
    """Похожие пары, которым досталась одна рамка, — их разводит только скорость."""
    return sorted(((i, j, wt) for i in range(len(weights)) for j, wt in weights[i].items()
                   if i < j and colors[i] == colors[j]), key=lambda t: -t[2])


# ── скорость ──────────────────────────────────────────────────────────────────

def speed_for(index: int, lo: float, hi: float, dead_zone: float) -> float:
    """Скорость из коридора по золотому сечению — ровно, без кучкования.

    Середина коридора (|v−1| < dead_zone) выкидывается: половина коридора
    отображается на [lo, 1−dz], половина на [1+dz, hi].
    """
    u = (index * GOLDEN) % 1.0
    if u < 0.5:
        v = lo + (1 - dead_zone - lo) * (u / 0.5)
    else:
        v = 1 + dead_zone + (hi - 1 - dead_zone) * ((u - 0.5) / 0.5)
    return round(v, 3)


# ── шум ───────────────────────────────────────────────────────────────────────
SLICE_SEC = 6.0


def sh(cmd: list, what: str) -> str:
    r = subprocess.run([str(c) for c in cmd], text=True, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"{what} не удался:\n{r.stderr[-2000:]}")
    return r.stdout


def duration(path: Path) -> float:
    out = sh(["ffprobe", "-v", "error", "-show_entries", "format=duration",
              "-of", "default=nw=1:nk=1", path], f"ffprobe {path.name}")
    try:
        return float(out.strip())
    except ValueError:
        return 0.0


def video_size(path: Path) -> tuple[int, int]:
    out = sh(["ffprobe", "-v", "error", "-select_streams", "v:0",
              "-show_entries", "stream=width,height", "-of", "csv=p=0", path],
             f"ffprobe {path.name}")
    w, h = out.strip().split(",")[:2]
    return int(w), int(h)


def slice_plan(dur: float) -> int:
    """Сколько нарезок брать из исходника: длинный даёт больше, но не больше 12."""
    return max(3, min(12, int(dur // 5)))


def noise_slice_filter(src_w: int, src_h: int, pos: float) -> str:
    """Вертикальное окно из горизонтального шума → серый слой «вокруг 128».

    Высокочастотная часть (кадр минус его размытие + 128) оставляет только
    зерно/помехи и убирает общий свет исходника — поэтому любой шум одинаково
    ложится поверх кадра режимом overlay и не красит его.
    """
    cw = min(src_w, round(src_h * W / H / 2) * 2)
    x = round((src_w - cw) * pos)
    return (f"crop={cw}:{src_h}:{x}:0,scale={W}:{H}:flags=bicubic,fps={FPS},"
            f"format=gray,split[a][b];[b]gblur=sigma=12[bl];"
            f"[a][bl]blend=all_mode=grainextract,format=gray")


def noise_std(path: Path) -> float:
    """Сила шума — СКО отклонения от 128 по нескольким кадрам."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf",
                          "fps=2,scale=270:480", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True).stdout
    a = np.frombuffer(raw, np.uint8).astype(np.float32)
    return float(np.sqrt(np.mean((a - 128.0) ** 2))) if a.size else 0.0


def cmd_pack(args: argparse.Namespace, cfg: dict) -> int:
    src_dir = ROOT / "noise" / "sources"
    pack = ROOT / cfg["noise"]["pack"]
    sources = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in {".mp4", ".webm", ".mov", ".ogv"})
    if not sources:
        print(f"В {src_dir} нет исходников — запустите noise/fetch_sources.sh")
        return 1
    if pack.exists():
        shutil.rmtree(pack)
    pack.mkdir(parents=True)
    jobs = []
    for src in sources:
        dur = duration(src)
        sw, sh_ = video_size(src)
        k = slice_plan(dur)
        for j in range(k):
            start = 0.0 if dur <= SLICE_SEC else (dur - SLICE_SEC) * j / max(1, k - 1)
            pos = (j * GOLDEN + 0.5) % 1.0
            jobs.append((src, start, sw, sh_, pos, j))
    index = []

    def make(job, n):
        src, start, sw, sh_, pos, j = job
        out = pack / f"noise_{n:03d}.mp4"
        sh(["ffmpeg", "-y", "-v", "error", "-stream_loop", "-1", "-ss", f"{start:.2f}",
            "-i", src, "-t", SLICE_SEC, "-an",
            "-filter_complex", noise_slice_filter(sw, sh_, pos),
            "-c:v", "libx264", "-preset", "medium", "-crf", 16, "-pix_fmt", "yuv420p",
            "-g", FPS, out], f"нарезка {src.name}")
        return {"file": out.name, "source": src.name, "start": round(start, 2),
                "window": round(pos, 3), "std": round(noise_std(out), 2)}

    with ThreadPoolExecutor(args.jobs) as ex:
        index = list(ex.map(make, jobs, range(len(jobs))))
    weak = [e for e in index if e["std"] < 1.0]
    index = [e for e in index if e["std"] >= 1.0]
    for e in weak:
        (pack / e["file"]).unlink(missing_ok=True)
    (pack / "_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1))
    print(f"Нарезок шума: {len(index)} из {len(sources)} исходников → {pack}")
    if weak:
        print(f"Выброшено почти пустых нарезок: {len(weak)}")
    return 0


# ── рендер ────────────────────────────────────────────────────────────────────

@dataclass
class Look:
    framing: int
    zoom: float
    x: int
    y: int
    speed: float
    noise: str
    noise_opacity: float
    gop: int

    def describe(self) -> str:
        return (f"рамка #{self.framing:02d} (×{self.zoom:.2f} @ {self.x},{self.y}), "
                f"скорость {self.speed:.3f}, шум {self.noise} ×{self.noise_opacity:.2f}")


def build_filter(look: Look) -> str:
    w, h = scaled_size(look.zoom)
    video = (f"[0:v]setpts=PTS/{look.speed},fps={FPS},"
             f"scale={w}:{h}:flags=bicubic,crop={W}:{H}:{look.x}:{look.y},"
             f"setsar=1,format=yuv420p[m];"
             f"[1:v]scale={W}:{H},format=yuv420p[n];"
             # Шум только по яркости. В режиме normal ffmpeg считает A·op + B·(1−op),
             # поэтому цвет кадра сохраняет opacity 1, а не 0 (0 отдаёт серый
             # цвет слоя шума — ролик становится чёрно-белым).
             f"[m][n]blend=c0_mode=overlay:c0_opacity={look.noise_opacity:.3f}:"
             f"c1_mode=normal:c1_opacity=1:c2_mode=normal:c2_opacity=1:shortest=1[v]")
    audio = f"[0:a]atempo={look.speed},aresample=48000[a]"
    return f"{video};{audio}"


def render(src: Path, out: Path, look: Look, noise_dir: Path, cfg: dict) -> None:
    sh(["ffmpeg", "-y", "-v", "error", "-i", src,
        "-stream_loop", "-1", "-i", noise_dir / look.noise,
        "-filter_complex", build_filter(look), "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", cfg["render"]["preset"], "-crf", cfg["render"]["crf"],
        "-r", FPS, "-g", look.gop, "-pix_fmt", "yuv420p",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-map_metadata", "-1", "-movflags", "+faststart", out], f"рендер {src.name}")


def next_version(base: Path) -> Path:
    nums = [int(p.name[1:]) for p in base.glob("v*") if p.is_dir() and p.name[1:].isdigit()]
    out = base / f"v{max(nums, default=0) + 1}"
    out.mkdir(parents=True)
    latest = base / "latest"
    if latest.is_symlink():
        latest.unlink()
    if not latest.exists():
        latest.symlink_to(out.name, target_is_directory=True)
    return out


def plan_looks(srcs: list[Path], cfg: dict, jobs: int) -> tuple[list[Look], list[tuple], list]:
    fr = cfg["framing"]
    palette = [f for f in FRAMINGS
               if framing_is_safe(f, fr["text_left"], fr["text_right"], fr["text_margin"])]
    weights = similarity(srcs, jobs)
    colors = assign_framings(weights, len(palette))
    index = json.loads((ROOT / cfg["noise"]["pack"] / "_index.json").read_text())
    sp = cfg["speed"]
    # Скорость считается по номеру ролика ВНУТРИ его рамки: у роликов с одной
    # рамкой (а это единственные, кого рамка не разводит) скорости разные.
    rank: dict[int, int] = {}
    looks = []
    for i, c in enumerate(colors):
        z, x, y = palette[c]
        r = rank[c] = rank.get(c, -1) + 1
        # Сдвиг на 3·рамку: внутри рамки скорости по-прежнему разные, а по
        # партии не собираются в одно значение у первых роликов каждой рамки.
        nz = index[(i * 7) % len(index)]
        opacity = min(0.6, cfg["noise"]["strength"] / max(nz["std"], 1e-3))
        looks.append(Look(framing=c, zoom=z, x=x, y=y,
                          speed=speed_for(r + 3 * c, sp["min"], sp["max"], sp["dead_zone"]),
                          noise=nz["file"], noise_opacity=round(opacity, 3),
                          gop=15 + (i * 5) % 11))
    return looks, palette, shared_framing_pairs(weights, colors)


def cmd_run(args: argparse.Namespace, cfg: dict) -> int:
    noise_dir = ROOT / cfg["noise"]["pack"]
    if not (noise_dir / "_index.json").exists():
        print("Нет нарезок шума — сначала: python uniquify.py pack")
        return 1
    srcs = sorted(p for p in args.src.iterdir()
                  if p.suffix.lower() in {".mp4", ".mov"} and not p.name.startswith("."))
    if args.limit:
        srcs = srcs[:args.limit]
    if not srcs:
        print(f"В {args.src} нет роликов")
        return 1
    stems = [p.stem for p in srcs]
    print(f"Сравниваю исходники между собой ({len(srcs)} роликов)…", flush=True)
    looks, palette, shared = plan_looks(srcs, cfg, args.jobs)
    print(f"Роликов: {len(srcs)} · рамок задействовано: {len({l.framing for l in looks})} "
          f"из {len(palette)} безопасных")
    if shared:
        print(f"⚠ Рамок не хватило: {len(shared)} похожих пар делят рамку, их разводит "
              f"только скорость. Самые похожие:")
        for i, j, wt in shared[:5]:
            print(f"    {stems[i]} ↔ {stems[j]}: {wt:g} общих кадров в исходниках")
    else:
        print("Все похожие пары получили разные рамки.")
    if args.dry_run:
        for s, l in zip(stems, looks):
            print(f"  {s:<16} {l.describe()}")
        return 0

    out = next_version(ROOT / "output" / args.name)
    plan = {s: asdict(l) for s, l in zip(stems, looks)}
    (out / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1))
    print(f"Пишу в {out}", flush=True)

    done = 0

    def one(pair):
        nonlocal done
        src, look = pair
        render(src, out / src.name, look, noise_dir, cfg)
        done += 1
        print(f"  [{done}/{len(srcs)}] {src.name}: {look.describe()}", flush=True)

    with ThreadPoolExecutor(cfg["render"]["jobs"]) as ex:
        list(ex.map(one, zip(srcs, looks)))

    gate = [sys.executable, str(ROOT / "qa_dedup.py"), str(out),
            "--distance", str(cfg["gate"]["distance"]),
            "--fail-pct", str(cfg["gate"]["fail_pct"]),
            "--report", str(out / "dedup_report.json")]
    rc = subprocess.run(gate).returncode
    if args.against:
        for ref in args.against:
            print(f"\n── против {ref} ──")
            rc |= subprocess.run(gate[:3] + ["--against", str(ref)] + gate[3:-2]
                                 + ["--report", str(out / f"dedup_vs_{ref.name}.json")]).returncode
    print(f"\nГотово: {out}")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack", help="нарезать шумы из noise/sources в noise/pack")
    p.add_argument("--jobs", type=int, default=4)
    r = sub.add_parser("run", help="уникализировать папку роликов и проверить гейтом")
    r.add_argument("src", type=Path)
    r.add_argument("--name", required=True, help="папка в output/ (версии v1, v2…)")
    r.add_argument("--against", type=Path, action="append",
                   help="ещё сравнить с роликами из этой папки (например, уже выложенными)")
    r.add_argument("--limit", type=int)
    r.add_argument("--jobs", type=int, default=8, help="процессов для хэширования")
    r.add_argument("--dry-run", action="store_true", help="только показать план")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    return cmd_pack(args, cfg) if args.cmd == "pack" else cmd_run(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
