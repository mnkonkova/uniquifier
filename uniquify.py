#!/usr/bin/env python3
"""Уникализатор: из готовых рилсов делает версии, которые Meta не сочтёт одним и
тем же роликом, и сам проверяет результат гейтом vPDQ.

Подача — три вопроса:
  СКОЛЬКО   ``--count N``: сколько роликов выдать. Уникализатор сначала считает,
            сколько роликов из папок вообще можно развести до порога (команда
            ``capacity`` показывает это заранее таблицей), и берёт до N из них —
            в первую очередь тех, у кого меньше всего похожих соседей.
  ПОРОГ     ``config.yaml → gate.fail_pct`` (по умолчанию 30) или разово
            ``--threshold P``: пара не проходит гейт при > P% общих кадров.
            0 — ни одного общего кадра (строго; на «бот работ» это ~150 из 450),
            30 — все 450; Meta в примере vPDQ зовёт копией от 80%.
            ``--allow-shared`` добирает до N остальными, честно помечая пары,
            которые делят облик.
  КАК       ``--levers``: какие рычаги включить (через запятую):
              рамка   приближение 5–14% со сдвигом окна — главный рычаг;
              наклон  ±0.5° — расширяет набор обликов, титры чуть завалены;
              скорость ±1–3%, голос не меняет высоту — помогает, но не спасает;
              шум     еле заметный слой из noise/pack — на хэш не влияет, косметика.
            По умолчанию: рамка,скорость,шум.

Облики (рамка ± наклон) подбираются под материал при каждом запуске: перебором
по кадрам самих исходников (``palette.py``), результат кэшируется.

Команды:
  python uniquify.py pack                                  # шумы → noise/pack (один раз)
  python uniquify.py capacity SRC [SRC…]                   # сколько можно развести
  python uniquify.py run SRC [SRC…] --name X --count 100 --threshold 10 --levers рамка,скорость,шум
  python uniquify.py run SRC --name X --dry-run            # только план
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

import palette

ROOT = Path(__file__).resolve().parent
W, H, FPS = 1080, 1920, 30
GOLDEN = 0.6180339887498949
CACHE = ROOT / "output" / "_cache"

LEVERS = {"рамка": "frame", "наклон": "tilt", "скорость": "speed", "шум": "noise",
          "frame": "frame", "tilt": "tilt", "speed": "speed", "noise": "noise"}
DEFAULT_LEVERS = "рамка,скорость,шум"
IDENTITY = (1.0, 0, 0, 0.0)


def parse_levers(text: str) -> set[str]:
    out = set()
    for part in text.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part not in LEVERS:
            raise SystemExit(f"Неизвестный рычаг «{part}». Есть: рамка, наклон, скорость, шум.")
        out.add(LEVERS[part])
    if "tilt" in out:
        out.add("frame")        # наклон крутит кадр внутри той же рамки
    return out


# ── кто на кого похож ─────────────────────────────────────────────────────────
BLOCK_NAME = re.compile(r"^[a-zA-Z]+\d+(?:_[a-zA-Z]+\d+)+$")
Weights = list[dict[int, float]]


def blocks_of(stem: str) -> frozenset[str] | None:
    """Куски ролика по имени ``h3_t7_c4`` → {h3, t7, c4}; None — имя не по схеме."""
    if not BLOCK_NAME.match(stem):
        return None
    return frozenset(stem.lower().split("_"))


def conflicts(stems: list[str]) -> list[set[int]]:
    """Для каждого ролика — номера роликов, с которыми у него общий кусок по имени."""
    blocks = [blocks_of(s) for s in stems]
    adj: list[set[int]] = [set() for _ in stems]
    for i in range(len(stems)):
        for j in range(i + 1, len(stems)):
            a, b = blocks[i], blocks[j]
            if a is not None and b is not None and a & b:
                adj[i].add(j)
                adj[j].add(i)
    return adj


# Пара с общим куском по имени, у которой кадры исходников случайно не совпали
# (секундная выборка легла между похожими кадрами), — всё равно сосед, но слабый.
NAME_ONLY_PCT = 1.0


def similarity(srcs: list[Path], jobs: int) -> Weights:
    """Кто на кого похож В ИСХОДНИКАХ: вес = % совпавших кадров (как в гейте).

    Имени мало: на «бот работ» тысячи пар без общего куска в имени делят кадры —
    общие мемы, кадр продукта, тот же человек в той же комнате. Поэтому граф
    строится по самим кадрам (тот же PDQ, что и в гейте), а имя добавляет слабые
    рёбра там, где кадры не совпали.
    """
    from qa_dedup import MATCH_DISTANCE, distances, load_many

    videos = load_many(srcs, jobs)
    adj_names = conflicts([p.stem for p in srcs])
    w: Weights = [dict() for _ in srcs]
    for i in range(len(srcs)):
        for j in range(i + 1, len(srcs)):
            d = distances(videos[i].bits, videos[j].bits)
            hit = d <= MATCH_DISTANCE
            pct = 100.0 * max(hit.any(1).mean(), hit.any(0).mean()) if d.size else 0.0
            if not pct and j in adj_names[i]:
                pct = NAME_ONLY_PCT
            if pct:
                w[i][j] = w[j][i] = float(pct)
    return w


def over(weights: Weights, threshold: float) -> Weights:
    """Только пары, которые без разных обликов не прошли бы гейт (> threshold %).

    Пара с общим обликом сохраняет примерно то сходство, что было в исходниках
    (скорость его только снижает), а пара с разными обликами уходит в ноль, —
    поэтому развести обликами нужно лишь пары выше порога.
    """
    return [{j: v for j, v in w.items() if v > threshold} for w in weights]


# ── отбор: сколько и кому какой облик ─────────────────────────────────────────

def select_strict(weights: Weights, k: int, count: int | None) -> dict[int, int]:
    """Ролики, которые можно развести до нуля: {ролик: облик}.

    Похожие ролики (вес > 0) не делят облик. Порядок — сначала ролики с самым
    малым числом похожих соседей: так в отбор попадает больше роликов (это
    эвристика максимального набора). Ролик, которому не осталось свободного
    облика, пропускается.
    """
    order = sorted(range(len(weights)), key=lambda i: (len(weights[i]), i))
    chosen: dict[int, int] = {}
    for u in order:
        if count is not None and len(chosen) >= count:
            break
        used = {chosen[j] for j in weights[u] if j in chosen}
        free = [c for c in range(k) if c not in used]
        if free:
            # Облик, которым пока пользуются меньше всего, — запас на потом.
            load = [0] * k
            for c in chosen.values():
                load[c] += 1
            chosen[u] = min(free, key=lambda c: (load[c], c))
    return chosen


def fill_shared(weights: Weights, k: int, chosen: dict[int, int], count: int) -> dict[int, int]:
    """Добрать до count остальными: облик, у владельцев которого меньше общих кадров."""
    out = dict(chosen)
    rest = sorted((i for i in range(len(weights)) if i not in out),
                  key=lambda i: (sum(weights[i].values()), i))
    for u in rest:
        if len(out) >= count:
            break
        cost = [0.0] * k
        for j, wt in weights[u].items():
            if j in out:
                cost[out[j]] += wt
        out[u] = min(range(k), key=lambda c: (cost[c], c))
    return out


def shared_pairs(weights: Weights, assign: dict[int, int]) -> list[tuple[int, int, float]]:
    """Похожие пары с одним обликом — их разводит только скорость."""
    return sorted(((i, j, wt) for i in assign for j, wt in weights[i].items()
                   if j in assign and i < j and assign[i] == assign[j]), key=lambda t: -t[2])


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
    look: int           # номер облика в наборе; −1 — без рамки
    zoom: float
    x: int
    y: int
    tilt: float
    speed: float
    noise: str | None
    noise_opacity: float
    gop: int

    def describe(self) -> str:
        parts = []
        if self.look >= 0:
            tilt = f", наклон {self.tilt:+.1f}°" if self.tilt else ""
            parts.append(f"облик #{self.look:02d} (×{self.zoom:.2f} @ {self.x},{self.y}{tilt})")
        if self.speed != 1.0:
            parts.append(f"скорость {self.speed:.3f}")
        if self.noise:
            parts.append(f"шум {self.noise} ×{self.noise_opacity:.2f}")
        return ", ".join(parts) or "без изменений"


def build_filter(look: Look) -> tuple[str, bool]:
    """filter_complex и нужен ли второй вход (шум)."""
    w, h = palette.scaled_size(look.zoom)
    chain = [f"[0:v]setpts=PTS/{look.speed}", f"fps={FPS}"]
    if look.look >= 0:
        chain.append(f"scale={w}:{h}:flags=bicubic")
        if look.tilt:
            chain.append(f"rotate={look.tilt}*PI/180:ow=iw:oh=ih:c=black:bilinear=1")
        chain.append(f"crop={W}:{H}:{look.x}:{look.y}")
    chain += ["setsar=1", "format=yuv420p"]
    video = ",".join(chain)
    if look.noise:
        video += ("[m];[1:v]scale={W}:{H},format=yuv420p[n];"
                  # Шум только по яркости. В режиме normal ffmpeg считает
                  # A·op + B·(1−op), поэтому цвет кадра сохраняет opacity 1, а не 0
                  # (0 отдаёт серый цвет слоя шума — ролик становится чёрно-белым).
                  f"[m][n]blend=c0_mode=overlay:c0_opacity={look.noise_opacity:.3f}:"
                  "c1_mode=normal:c1_opacity=1:c2_mode=normal:c2_opacity=1:shortest=1[v]"
                  ).replace("{W}", str(W)).replace("{H}", str(H))
    else:
        video += "[v]"
    audio = f"[0:a]atempo={look.speed},aresample=48000[a]"
    return f"{video};{audio}", bool(look.noise)


def render(src: Path, out: Path, look: Look, noise_dir: Path, cfg: dict) -> None:
    fc, with_noise = build_filter(look)
    inputs = ["-i", src]
    if with_noise:
        inputs += ["-stream_loop", "-1", "-i", noise_dir / look.noise]
    sh(["ffmpeg", "-y", "-v", "error", *inputs,
        "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
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


# ── подготовка: исходники, облики, отбор ──────────────────────────────────────

def collect(folders: list[Path], limit: int | None) -> list[Path]:
    from qa_dedup import videos_in

    srcs = []
    for folder in folders:
        found = videos_in(folder)
        srcs += found[:limit] if limit else found
    return srcs


def limits_from(cfg: dict) -> palette.Limits:
    fr = cfg["framing"]
    return palette.Limits(text_box=tuple(fr["text_box"]), text_margin=fr["text_margin"],
                          max_top=fr["max_top"], max_bottom=fr["max_bottom"])


def find_looks(srcs: list[Path], levers: set[str], cfg: dict, jobs: int) -> list[tuple]:
    if "frame" not in levers:
        return [IDENTITY]
    # Выборка для перебора — до 10 роликов, равномерно по списку: в неё попадают
    # и говорящая голова, и экран, и мемы.
    step = max(1, len(srcs) // 10)
    sample = srcs[::step][:10]
    print(f"Подбираю облики под материал ({'с наклоном' if 'tilt' in levers else 'без наклона'}, "
          f"выборка {len(sample)} роликов)…", flush=True)
    return palette.search(sample, "tilt" in levers, limits_from(cfg), CACHE, jobs=jobs)


def plan(srcs: list[Path], levers: set[str], count: int | None, allow_shared: bool,
         threshold: float, cfg: dict, jobs: int):
    print(f"Сравниваю исходники между собой ({len(srcs)} роликов)…", flush=True)
    weights = over(similarity(srcs, jobs), threshold)
    looks = find_looks(srcs, levers, cfg, jobs)
    strict = select_strict(weights, len(looks), count)
    assign = strict
    if allow_shared and count and len(strict) < count:
        assign = fill_shared(weights, len(looks), strict, min(count, len(srcs)))
    return weights, looks, strict, assign


def make_looks(srcs: list[Path], assign: dict[int, int], looks: list[tuple],
               levers: set[str], cfg: dict) -> dict[int, Look]:
    index = (json.loads((ROOT / cfg["noise"]["pack"] / "_index.json").read_text())
             if "noise" in levers else [])
    sp = cfg["speed"]
    rank: dict[int, int] = {}
    out = {}
    for n, i in enumerate(sorted(assign)):
        c = assign[i]
        z, x, y, t = looks[c]
        # Скорость считается по номеру ролика ВНУТРИ его облика: у роликов с одним
        # обликом (а их рамка не разводит) скорости разные. Сдвиг 3·облик не даёт
        # первым роликам всех обликов собраться в одно значение.
        r = rank[c] = rank.get(c, -1) + 1
        speed = speed_for(r + 3 * c, sp["min"], sp["max"], sp["dead_zone"]) if "speed" in levers else 1.0
        noise, opacity = None, 0.0
        if index:
            nz = index[(n * 7) % len(index)]
            noise = nz["file"]
            opacity = round(min(0.6, cfg["noise"]["strength"] / max(nz["std"], 1e-3)), 3)
        out[i] = Look(look=c if "frame" in levers else -1, zoom=z, x=x, y=y, tilt=t,
                      speed=speed, noise=noise, noise_opacity=opacity, gop=15 + (n * 5) % 11)
    return out


def label(p: Path, multi: bool) -> str:
    return f"{p.parent.name}/{p.stem}" if multi else p.stem


# ── команды ───────────────────────────────────────────────────────────────────

def cmd_capacity(args: argparse.Namespace, cfg: dict) -> int:
    """Сколько роликов можно развести — по порогам и наборам рычагов."""
    srcs = collect(args.src, args.limit)
    if not srcs:
        print("Нет роликов")
        return 1
    print(f"Сравниваю исходники между собой ({len(srcs)} роликов)…", flush=True)
    weights = similarity(srcs, args.jobs)
    pcts = np.array([v for i, w in enumerate(weights) for j, v in w.items() if i < j])
    print(f"\nРоликов: {len(srcs)} · пар: {len(srcs) * (len(srcs) - 1) // 2}")
    print("Сходство исходников (доля общих кадров у пары):")
    for t in (0, 10, 30, 80):
        print(f"  больше {t:>2}%: {(pcts > t).sum():>6} пар")
    sets = [("скорость+шум", {"speed", "noise"}),
            ("рамка+скорость+шум", {"frame", "speed", "noise"}),
            ("+наклон", {"frame", "tilt", "speed", "noise"})]
    sizes = {name: len(find_looks(srcs, lv, cfg, args.jobs)) for name, lv in sets}
    thresholds = args.thresholds
    print("\nСколько роликов можно сделать уникальными при пороге гейта:")
    print("  " + f"{'рычаги':<22}{'обликов':>8}" + "".join(f"{f'≤{t:g}%':>8}" for t in thresholds))
    for name, _ in sets:
        k = sizes[name]
        row = [len(select_strict(over(weights, t), k, None)) for t in thresholds]
        print("  " + f"{name:<22}{k:>8}" + "".join(f"{n:>8}" for n in row))
    print(f"\nПорог Meta для «копии» — {80}% общих кадров; строгий порог проекта — "
          f"{cfg['gate']['fail_pct']:g}%.")
    return 0


def cmd_run(args: argparse.Namespace, cfg: dict) -> int:
    levers = parse_levers(args.levers)
    noise_dir = ROOT / cfg["noise"]["pack"]
    if "noise" in levers and not (noise_dir / "_index.json").exists():
        print("Нет нарезок шума — сначала: python uniquify.py pack")
        return 1
    srcs = collect(args.src, args.limit)
    if not srcs:
        print(f"В {', '.join(map(str, args.src))} нет роликов")
        return 1
    multi = len(args.src) > 1
    threshold = cfg["gate"]["fail_pct"] if args.threshold is None else args.threshold
    weights, looks, strict, assign = plan(srcs, levers, args.count, args.allow_shared,
                                          threshold, cfg, args.jobs)
    want = min(args.count or len(srcs), len(srcs))
    print(f"\nРычаги: {', '.join(sorted(levers))} · обликов: {len(looks)}")
    print(f"Порог: пара не проходит при >{threshold:g}% общих кадров")
    print(f"Можно развести: {len(strict)} из {len(srcs)}"
          + (f" (просили {args.count})" if args.count else ""))
    if len(assign) < want:
        print(f"⚠ Выдам {len(assign)}, а не {want}: остальным не хватает свободного облика. "
              f"Включите наклон (--levers …,наклон) или добавьте --allow-shared.")
    shared = shared_pairs(weights, assign)
    if shared:
        print(f"⚠ --allow-shared: {len(shared)} похожих пар делят облик, их разводит только "
              f"скорость. Самые похожие:")
        for i, j, wt in shared[:5]:
            print(f"    {label(srcs[i], multi)} ↔ {label(srcs[j], multi)}: {wt:.0f}% общих кадров")
    per = make_looks(srcs, assign, looks, levers, cfg)
    if args.dry_run:
        for i in sorted(per):
            print(f"  {label(srcs[i], multi):<28} {per[i].describe()}")
        return 0

    out = next_version(ROOT / "output" / args.name)
    folders = [out / f.name for f in args.src] if multi else [out]
    for f in folders:
        f.mkdir(exist_ok=True)
    (out / "plan.json").write_text(json.dumps({
        "levers": sorted(levers), "count": args.count, "threshold": threshold, "looks": looks,
        "skipped": [label(p, multi) for i, p in enumerate(srcs) if i not in assign],
        "videos": {label(srcs[i], multi): asdict(l) for i, l in sorted(per.items())},
    }, ensure_ascii=False, indent=1))
    print(f"Пишу в {out}", flush=True)

    def target(src: Path) -> Path:
        return (out / src.parent.name if multi else out) / src.name

    done = 0

    def one(i: int):
        nonlocal done
        render(srcs[i], target(srcs[i]), per[i], noise_dir, cfg)
        done += 1
        print(f"  [{done}/{len(per)}] {label(srcs[i], multi)}: {per[i].describe()}", flush=True)

    with ThreadPoolExecutor(cfg["render"]["jobs"]) as ex:
        list(ex.map(one, sorted(per)))

    base = [sys.executable, str(ROOT / "qa_dedup.py"), *map(str, folders),
            "--distance", str(cfg["gate"]["distance"]),
            "--fail-pct", str(threshold)]
    print("\n── новые ролики между собой ──", flush=True)
    rc = subprocess.run(base + ["--report", str(out / "dedup_report.json")]).returncode
    if args.against:
        print("\n── новые ролики против " + ", ".join(map(str, args.against)) + " ──", flush=True)
        against = [a for ref in args.against for a in ("--against", str(ref))]
        rc |= subprocess.run(base + against
                             + ["--report", str(out / "dedup_vs_against.json")]).returncode
    print(f"\nГотово: {out}")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pack", help="нарезать шумы из noise/sources в noise/pack")
    p.add_argument("--jobs", type=int, default=4)

    c = sub.add_parser("capacity", help="сколько роликов можно сделать уникальными")
    c.add_argument("src", type=Path, nargs="+")
    c.add_argument("--limit", type=int)
    c.add_argument("--jobs", type=int, default=10)
    c.add_argument("--thresholds", type=float, nargs="+", default=[0, 5, 10, 20, 30])

    r = sub.add_parser("run", help="уникализировать и проверить гейтом")
    r.add_argument("src", type=Path, nargs="+",
                   help="папки с готовыми роликами; несколько — уникализируются вместе")
    r.add_argument("--name", required=True, help="папка в output/ (версии v1, v2…)")
    r.add_argument("--count", type=int, help="сколько роликов выдать (по умолчанию — сколько выйдет)")
    r.add_argument("--levers", default=DEFAULT_LEVERS,
                   help="рамка,наклон,скорость,шум — через запятую (по умолчанию %(default)s)")
    r.add_argument("--threshold", type=float,
                   help="пара не проходит гейт при > N%% общих кадров (по умолчанию gate.fail_pct из config.yaml)")
    r.add_argument("--allow-shared", action="store_true",
                   help="добрать до --count роликами с общим обликом (гейт их, скорее всего, не пропустит)")
    r.add_argument("--against", type=Path, action="append",
                   help="ещё сравнить с роликами из этой папки (например, уже выложенными)")
    r.add_argument("--limit", type=int, help="брать не больше N роликов из каждой папки")
    r.add_argument("--jobs", type=int, default=10, help="процессов для хэширования")
    r.add_argument("--dry-run", action="store_true", help="только показать план")

    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.cmd == "pack":
        return cmd_pack(args, cfg)
    if args.cmd == "capacity":
        return cmd_capacity(args, cfg)
    return cmd_run(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
