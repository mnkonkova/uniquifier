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
              скорость ступени 0.92…1.08 через 4%, высота голоса не меняется —
                      для картинки слабая, для звука сильная: разница 4% между
                      роликами с общим голосом — 91% → 28% совпавших отрезков;
              тон     высота голоса −3% или +3% с сохранением тембра — главный
                      рычаг для звука (против оригинала 36% → 12% совпавших
                      отрезков Chromaprint), на слух не заметен.
            По умолчанию: рамка,скорость,тон.

            Шума нет намеренно: ни на картинке (100% совпавших кадров при шуме
            25%), ни в звуке (91–100% совпавших отрезков даже при шуме на 6 дБ
            тише голоса) он отпечатки не меняет.

Облики (рамка ± наклон) подбираются под материал при каждом запуске: перебором
по кадрам самих исходников (``palette.py``), результат кэшируется.

Команды:
  python uniquify.py capacity SRC [SRC…]                   # сколько можно развести
  python uniquify.py run SRC [SRC…] --name X --count 100 --threshold 10 --levers рамка,скорость,тон
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
GOLDEN = 0.6180339887498949
CACHE = ROOT / "output" / "_cache"

LEVERS = {"рамка": "frame", "наклон": "tilt", "скорость": "speed", "тон": "pitch",
          "frame": "frame", "tilt": "tilt", "speed": "speed", "pitch": "pitch"}
DEFAULT_LEVERS = "рамка,скорость,тон"
IDENTITY = (1.0, 0.0, 0.0, 0.0)


def parse_levers(text: str) -> set[str]:
    out = set()
    for part in text.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part not in LEVERS:
            raise SystemExit(f"Неизвестный рычаг «{part}». Есть: рамка, наклон, скорость, тон.")
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


def sh(cmd: list, what: str) -> str:
    r = subprocess.run([str(c) for c in cmd], text=True, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"{what} не удался:\n{r.stderr[-2000:]}")
    return r.stdout


# ── голос: тон × скорость ───────────────────────────────────────────────────
# Звук двух роликов с общим голосом (хук + совет) разводят две вещи — высота и
# темп. Замер Chromaprint, доля совпавших отрезков у такой пары:
#   тон:      одинаковый 87–91%, разница 1% — 67–76%, 3% — 12–29%, 6% — 1–5%;
#   скорость: одинаковая 91%, разница 3% — 35–38%, 4% — 28%, 6% — 20%, 10% — 12%.
# Неслышных значений мало, поэтому и то и другое — ступенями (config.yaml), а
# пара ступеней раздаётся так, чтобы ролики с общим голосом попадали в разные.
_PITCH_CURVE = ((0.0, 1.0), (0.01, 0.8), (0.03, 0.25), (0.06, 0.03))
_SPEED_CURVE = ((0.0, 1.0), (0.03, 0.4), (0.04, 0.3), (0.06, 0.22), (0.10, 0.13))


def _curve(table: tuple, x: float) -> float:
    x = abs(x)
    for (x0, y0), (x1, y1) in zip(table, table[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return table[-1][1]


def voice_overlap(pa: float, sa: float, pb: float, sb: float) -> float:
    """Какая доля общего голоса останется похожей при таких тонах и скоростях."""
    return _curve(_PITCH_CURVE, pa - pb) * _curve(_SPEED_CURVE, sa - sb)


# Какую долю звука ролика занимает каждый кусок имени h_t_c (хук ~¼, совет ~½,
# финал ~¼) — общий кусок значит общий голос, даже если кадры уже разведены.
BLOCK_AUDIO_SHARE = (25.0, 50.0, 25.0)


def voice_weights(raw: Weights, stems: list[str]) -> Weights:
    """Сколько звука делят ролики: общие куски по имени ∪ общие кадры (мемы)."""
    out: Weights = [dict(w) for w in raw]
    parts = [s.lower().split("_") if blocks_of(s) else None for s in stems]
    for i in range(len(stems)):
        for j in range(i + 1, len(stems)):
            a, b = parts[i], parts[j]
            if a is None or b is None or len(a) != len(b):
                continue
            share = sum(BLOCK_AUDIO_SHARE[k] if k < 3 else 0.0
                        for k, (x, y) in enumerate(zip(a, b)) if x == y)
            if share > out[i].get(j, 0.0):
                out[i][j] = out[j][i] = share
    return out


ORIGINAL_WEIGHT = 100.0
# Сочетания, при которых голос слишком похож на свой оригинал, не выдаются:
# тон 3% без смены скорости — 0.25 (на тесте 31–32% совпавших отрезков).
MAX_OWN_OVERLAP = 0.2


def voice_options(pitches: list[float], speeds: list[float]) -> list[tuple[float, float]]:
    opts = [(p, s) for p in pitches for s in speeds]
    far = [o for o in opts if voice_overlap(o[0], o[1], 1.0, 1.0) <= MAX_OWN_OVERLAP]
    return far or opts


def voice_cost(o: tuple[float, float], u: int, weights: Weights,
               picked: dict[int, tuple[float, float]], own_weight: float = ORIGINAL_WEIGHT) -> float:
    # Свой оригинал — тоже сосед, с которым ролик делит весь голос.
    own = own_weight * voice_overlap(o[0], o[1], 1.0, 1.0)
    return own + sum(w * voice_overlap(o[0], o[1], *picked[j])
                     for j, w in weights[u].items() if j in picked)


def voice_menu(cfg: dict, levers: set[str]) -> tuple[list[tuple[float, float]], dict]:
    """Варианты голоса (тон, скорость) и способ обработки каждого.

    ``config.yaml → voices`` — явный список ``{pitch, speed, mode}``:
      r3    — rubberband-r3 --fine, тон с сохранением тембра (без «робота»);
      vinyl — «пластинка»: темп и тон вместе, без вокодера (pitch = speed).
    Без списка — все сочетания ступеней ``pitch.levels × speed.levels`` движком r3.
    """
    voices = cfg.get("voices")
    if voices and levers & {"pitch", "speed"}:
        opts = [(float(v["pitch"]), float(v["speed"])) for v in voices]
        modes = {(float(v["pitch"]), float(v["speed"])): v.get("mode", "r3") for v in voices}
        return opts, modes
    pitches = cfg["pitch"]["levels"] if "pitch" in levers else [1.0]
    speeds = cfg["speed"]["levels"] if "speed" in levers else [1.0]
    opts = voice_options(pitches, speeds)
    return opts, {o: "r3" for o in opts}


def assign_voice(weights: Weights, chosen: list[int],
                 options: list[tuple[float, float]]) -> dict[int, tuple[float, float]]:
    """(тон, скорость) каждому ролику: у похожих по звуку — как можно дальше.

    Жадно, от самых «нагруженных» роликов: пара ступеней, при которой сумма
    оставшейся похожести с уже раздавшими соседями минимальна.
    """
    order = sorted(chosen, key=lambda i: -sum(weights[i].values()))
    picked: dict[int, tuple[float, float]] = {}
    used = {o: 0 for o in options}
    for u in order:
        # Близость к оригиналу здесь — только запрет (voice_options отсекает
        # слишком близкие), а не вес: с весом все ролики без соседей получали
        # один и тот же самый далёкий голос (на примерах — три «пластинки»).
        best = min(options, key=lambda o: (round(voice_cost(o, u, weights, picked, 0.0), 6),
                                           used[o], options.index(o)))
        picked[u] = best
        used[best] += 1
    return picked


# ── рендер ────────────────────────────────────────────────────────────────────

@dataclass
class Look:
    look: int           # номер облика в наборе; −1 — без рамки
    zoom: float
    ox: float           # сдвиг окна — в долях кадра (см. palette.py)
    oy: float
    tilt: float
    speed: float
    gop: int
    pitch: float = 1.0
    # Как менять голос: r3 (rubberband-r3 --fine), vinyl («пластинка»),
    # r2 — прежний фильтр ffmpeg (старые планы; давал «робо-эхо»).
    voice: str = "r2"

    @classmethod
    def from_plan(cls, d: dict) -> "Look":
        """Из plan.json; старые планы хранили сдвиг в пикселях кадра 1080×1920."""
        d = dict(d)
        if "x" in d:
            d["ox"], d["oy"] = d.pop("x") / 1080, d.pop("y") / 1920
        d.pop("noise", None)
        d.pop("noise_opacity", None)
        return cls(**d)

    def describe(self) -> str:
        parts = []
        if self.look >= 0:
            tilt = f", наклон {self.tilt:+.1f}°" if self.tilt else ""
            parts.append(f"облик #{self.look:02d} (×{self.zoom:.2f}, сдвиг "
                         f"{100 * self.ox:.1f}%/{100 * self.oy:.1f}%{tilt})")
        if self.speed != 1.0:
            parts.append(f"скорость {self.speed:.3f}")
        if self.pitch != 1.0:
            parts.append(f"тон {100 * (self.pitch - 1):+.0f}%")
        return ", ".join(parts) or "без изменений"


@dataclass(frozen=True)
class Media:
    width: int          # как показывает плеер (поворот из метаданных учтён)
    height: int
    fps: str            # как у исходника: «30/1», «60000/1001»…
    has_audio: bool


def probe(src: Path) -> Media:
    out = sh(["ffprobe", "-v", "error", "-show_entries",
              "stream=codec_type,width,height,avg_frame_rate", "-of", "json", src],
             f"ffprobe {src.name}")
    streams = json.loads(out).get("streams", [])
    video = next((st for st in streams if st.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError(f"{src.name}: нет видеодорожки")
    w, h = palette.video_dims(src)
    fps = video.get("avg_frame_rate") or "30/1"
    if fps in ("0/0", "0/1"):
        fps = "30/1"
    return Media(w, h, fps, any(st.get("codec_type") == "audio" for st in streams))


def out_fps(src_fps: str, speed: float) -> str:
    """Частота кадров выхода = исходная × скорость (дробью, без округления)."""
    from fractions import Fraction

    f = Fraction(src_fps) * Fraction(str(speed))
    return f"{f.numerator}/{f.denominator}"


def audio_filter(look: Look) -> str:
    """Фильтр звука для ffmpeg (r2 и vinyl; r3 обрабатывается до ffmpeg — ``r3_audio``).

    vinyl — «пластинка»: частота дискретизации ×pitch, без вокодера — ни эха,
    ни «робота»; темп и тон меняются вместе (в списке голосов pitch = speed).

    r2 — темп и тон ОДНИМ проходом rubberband из ffmpeg.

    Раньше темп менял atempo, а тон — rubberband отдельно: два растяжения
    подряд, и на крайних значениях (тон ±6%, скорость ±8%) голос давал
    металлическое «робо-эхо» (отзыв 2026-10-12). formant=preserved — двигается
    высота, тембр остаётся.
    """
    if look.pitch == 1.0 and look.speed == 1.0:
        return "aresample=48000"
    if look.voice == "vinyl":
        return f"aresample=48000,asetrate=48000*{look.pitch},aresample=48000"
    return (f"rubberband=tempo={look.speed}:pitch={look.pitch}:formant=preserved:"
            f"pitchq=quality,aresample=48000")


def r3_audio(src: Path, look: Look, work: Path) -> Path | None:
    """Голос движком R3 (``rubberband-r3 --fine --formant``) — отдельным шагом:
    в ffmpeg есть только движок R2, а он с сохранением тембра звучит «роботом»
    (сравнение на слух, 2026-10-12). Возвращает wav или None, если не нужно."""
    if look.voice != "r3" or (look.pitch == 1.0 and look.speed == 1.0):
        return None
    work.mkdir(parents=True, exist_ok=True)
    raw = work / f"{src.stem}.{look.pitch}_{look.speed}.in.wav"
    out = work / f"{src.stem}.{look.pitch}_{look.speed}.r3.wav"
    sh(["ffmpeg", "-y", "-v", "error", "-i", src, "-vn", "-ac", "2", "-ar", "48000",
        "-c:a", "pcm_s16le", raw], f"звук {src.name}")
    sh(["rubberband-r3", "--fine", "--formant", "-T", look.speed, "-f", look.pitch, raw, out],
       f"rubberband-r3 {src.name}")
    raw.unlink(missing_ok=True)
    return out


_ASS_DIA = re.compile(r"^(Dialogue:\s*\d+,)(\d+:\d\d:\d\d\.\d\d),(\d+:\d\d:\d\d\.\d\d)(,.*)$")


def _ass_sec(ts: str) -> float:
    h, m, sec = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def _ass_ts(t: float) -> str:
    h, rem = divmod(max(0.0, t), 3600)
    m, sec = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{sec:05.2f}"


def retime_ass(src: Path, out: Path, speed: float) -> None:
    """Титры под новую скорость: все времена ÷ speed (ролик ×1.04 короче на 4%)."""
    lines = []
    for ln in src.read_text(encoding="utf-8").splitlines():
        m = _ASS_DIA.match(ln)
        if m:
            ln = (m.group(1) + _ass_ts(_ass_sec(m.group(2)) / speed) + ","
                  + _ass_ts(_ass_sec(m.group(3)) / speed) + m.group(4))
        lines.append(ln)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def clean_pair(src: Path) -> tuple[Path, Path] | None:
    """Версия без титров и титры, которые сборщик кладёт в ``_clean/`` рядом."""
    mp4, ass = src.parent / "_clean" / f"{src.stem}.mp4", src.parent / "_clean" / f"{src.stem}.ass"
    return (mp4, ass) if mp4.exists() and ass.exists() else None


def build_filter(look: Look, media: Media, external_audio: bool = False,
                 captions: tuple[Path, Path] | None = None) -> str:
    """Рамка в долях кадра: выход того же размера и пропорций, что исходник.

    Увеличение — по обеим осям одним множителем, поэтому картинка не
    сплющивается; окно вырезается размером с исходный кадр (чётным — так
    требует yuv420p).
    """
    w0, h0 = palette.even(media.width), palette.even(media.height)
    # Кадры не выбрасываются и не дублируются: частота кадров меняется вместе
    # со скоростью (30 → 31.2 при ×1.04). С жёсткими 30 к/с ×0.92 давал повтор
    # каждого 12-го кадра — это читалось как подёргивание.
    chain = [f"[0:v]setpts=PTS/{look.speed}"]
    if look.look >= 0:
        w, h = palette.even(w0 * look.zoom), palette.even(h0 * look.zoom)
        chain.append(f"scale={w}:{h}:flags=bicubic")
        if look.tilt:
            chain.append(f"rotate={look.tilt}*PI/180:ow=iw:oh=ih:c=black:bilinear=1")
        x = min(round(look.ox * w0), w - w0)
        y = min(round(look.oy * h0), h - h0)
        chain.append(f"crop={w0}:{h0}:{x}:{y}")
    else:
        chain.append(f"scale={w0}:{h0}")
    chain += ["setsar=1", "format=yuv420p"]
    if captions is not None:
        # Титры — ПОСЛЕ рамки: кадр приближен и сдвинут, а текст стоит на месте.
        ass, fonts = captions
        chain.append(f"ass=filename={ass}:fontsdir={fonts}")
    graph = ",".join(chain) + "[v]"
    if media.has_audio:
        graph += (";[1:a]aresample=48000[a]" if external_audio
                  else f";[0:a]{audio_filter(look)}[a]")
    return graph


def render_cmd(src: Path, out: Path, look: Look, media: Media, cfg: dict,
               audio: Path | None = None, captions: tuple[Path, Path] | None = None) -> list:
    inputs = ["-i", src] + (["-i", audio] if audio else [])
    cmd = ["ffmpeg", "-y", "-v", "error", *inputs,
           "-filter_complex", build_filter(look, media, audio is not None, captions),
           "-map", "[v]"]
    # У сырых файлов звука часто нет — тогда и дорожки в выходе нет.
    if media.has_audio:
        cmd += ["-map", "[a]", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]
    cmd += ["-c:v", "libx264", "-preset", cfg["render"]["preset"], "-crf", cfg["render"]["crf"],
            "-r", out_fps(media.fps, look.speed), "-g", look.gop, "-pix_fmt", "yuv420p",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
            "-map_metadata", "-1", "-map_chapters", "-1", "-movflags", "+faststart", out]
    return cmd


def render(src: Path, out: Path, look: Look, cfg: dict) -> None:
    """Собрать ролик. С ``--captions-on-top`` берётся версия без титров из
    ``_clean/``, а титры накладываются поверх после рамки и под новую скорость."""
    captions = None
    fonts = cfg.get("_captions_fonts")
    pair = clean_pair(src) if fonts else None
    if pair is not None:
        src, ass = pair
        (CACHE / "ass").mkdir(parents=True, exist_ok=True)
        retimed = CACHE / "ass" / f"{out.stem}.{look.speed}.ass"
        retime_ass(ass, retimed, look.speed)
        captions = (retimed, Path(fonts))
    media = probe(src)
    audio = r3_audio(src, look, CACHE / "r3") if media.has_audio else None
    try:
        sh(render_cmd(src, out, look, media, cfg, audio, captions), f"рендер {src.name}")
    finally:
        if audio is not None:
            audio.unlink(missing_ok=True)


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


def limits_from(cfg: dict, profile: str | None) -> palette.Limits:
    """Профиль рамки из config.yaml → framing.profiles (доли кадра)."""
    fr = cfg["framing"]
    name = profile or fr["profile"]
    if name not in fr["profiles"]:
        raise SystemExit(f"Нет профиля рамки «{name}». Есть: {', '.join(fr['profiles'])}.")
    p = fr["profiles"][name]
    box = p.get("text_box")
    return palette.Limits(text_box=tuple(box) if box else None,
                          text_margin=p.get("text_margin", 0.0),
                          max_top=p.get("max_top", 1.0), max_bottom=p.get("max_bottom", 1.0),
                          max_side=p.get("max_side", 1.0),
                          max_zoom=p.get("max_zoom", 9.0))


def find_looks(srcs: list[Path], levers: set[str], cfg: dict, jobs: int,
               profile: str | None = None) -> list[tuple]:
    if "frame" not in levers:
        return [IDENTITY]
    if cfg.get("_captions_fonts"):
        # Титры кладутся поверх после рамки — облики подбираются по кадрам БЕЗ
        # титров, и беречь зону титров не нужно.
        srcs = [clean_pair(p)[0] if clean_pair(p) else p for p in srcs]
    # Облик должен быть безопасен для каждого соотношения сторон в партии.
    aspects = sorted({round(w / h, 4) for w, h in map(palette.video_dims, srcs) if w and h})
    # Выборка для перебора — до 10 роликов, равномерно по списку: в неё попадают
    # и говорящая голова, и экран, и мемы.
    step = max(1, len(srcs) // 10)
    sample = srcs[::step][:10]
    print(f"Подбираю облики под материал ({'с наклоном' if 'tilt' in levers else 'без наклона'}, "
          f"выборка {len(sample)} роликов)…", flush=True)
    lim = limits_from(cfg, profile)
    if cfg.get("_captions_fonts"):
        import dataclasses
        lim = dataclasses.replace(lim, text_box=None)
    looks = palette.search(sample, "tilt" in levers, lim, aspects,
                           CACHE, jobs=jobs)
    if not looks:
        raise SystemExit("В полях профиля рамки нет облика, который ушёл бы от оригинала "
                         "(рамка в таких полях слишком слабая). Расширьте поля профиля в "
                         "config.yaml или уберите рычаг «рамка».")
    return looks


def plan(srcs: list[Path], levers: set[str], count: int | None, allow_shared: bool,
         threshold: float, cfg: dict, jobs: int, profile: str | None = None):
    print(f"Сравниваю исходники между собой ({len(srcs)} роликов)…", flush=True)
    raw = similarity(srcs, jobs)
    weights = over(raw, threshold)
    looks = find_looks(srcs, levers, cfg, jobs, profile)
    strict = select_strict(weights, len(looks), count)
    assign = strict
    if allow_shared and count and len(strict) < count:
        assign = fill_shared(weights, len(looks), strict, min(count, len(srcs)))
    return raw, weights, looks, strict, assign


def make_looks(raw: Weights, assign: dict[int, int], looks: list[tuple],
               levers: set[str], cfg: dict, stems: list[str]) -> dict[int, Look]:
    # Голос раздаётся по ПОЛНОМУ графу сходства, а не по графу «выше порога»:
    # общий хук звучит одинаково и у пар, которые картинкой уже разведены.
    options, modes = voice_menu(cfg, levers)
    voice = assign_voice(voice_weights(raw, stems), list(assign), options)
    out = {}
    for n, i in enumerate(sorted(assign)):
        c = assign[i]
        z, ox, oy, t = looks[c]
        pitch, speed = voice[i]
        out[i] = Look(look=c if "frame" in levers else -1, zoom=z, ox=ox, oy=oy, tilt=t,
                      speed=speed, gop=15 + (n * 5) % 11, pitch=pitch,
                      voice=modes.get((pitch, speed), "r3"))
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
    sets = [("скорость", {"speed"}),
            ("рамка+скорость", {"frame", "speed"}),
            ("+наклон", {"frame", "tilt", "speed"})]
    sizes = {name: len(find_looks(srcs, lv, cfg, args.jobs, args.profile)) for name, lv in sets}
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
    srcs = collect(args.src, args.limit)
    if not srcs:
        print(f"В {', '.join(map(str, args.src))} нет роликов")
        return 1
    multi = len(args.src) > 1
    threshold = cfg["gate"]["fail_pct"] if args.threshold is None else args.threshold
    raw, weights, looks, strict, assign = plan(srcs, levers, args.count, args.allow_shared,
                                               threshold, cfg, args.jobs, args.profile)
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
    per = make_looks(raw, assign, looks, levers, cfg, [p.stem for p in srcs])
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
        render(srcs[i], target(srcs[i]), per[i], cfg)
        done += 1
        print(f"  [{done}/{len(per)}] {label(srcs[i], multi)}: {per[i].describe()}", flush=True)

    with ThreadPoolExecutor(cfg["render"]["jobs"]) as ex:
        list(ex.map(one, sorted(per)))

    rc = gate_and_repair(out, folders, srcs, per, looks, threshold, levers, cfg, args, multi)
    print(f"\nГотово: {out}")
    return rc


def run_gate(folders: list[Path], threshold: float, cfg: dict, report: Path,
             against: list[Path] | None = None) -> int:
    cmd = [sys.executable, str(ROOT / "qa_dedup.py"), *map(str, folders),
           "--distance", str(cfg["gate"]["distance"]), "--fail-pct", str(threshold),
           "--report", str(report)]
    for ref in against or []:
        cmd += ["--against", str(ref)]
    return subprocess.run(cmd).returncode


REPAIR_ROUNDS = 3


def gate_and_repair(out: Path, folders: list[Path], srcs: list[Path], per: dict[int, Look],
                    looks: list[tuple], threshold: float, levers: set[str], cfg: dict,
                    args: argparse.Namespace, multi: bool) -> int:
    """Гейты звука и картинки; пары за порогом чинятся, гейт повторяется.

    Сначала звук: его чинит смена тона и скорости, а скорость трогает и
    картинку — поэтому картинка проверяется после. Смена облика картинки звук
    уже не меняет.

    Отбор разводит картинкой только пары, которые были за порогом УЖЕ в
    исходниках. Пара чуть ниже порога с общим обликом может перешагнуть его
    после рендера: смена скорости сдвигает секундную выборку (замер на 450
    роликах: 7 пар, 29–44%). Такой паре один ролик пересобирается с обликом, где
    у него меньше всего общего с соседями, и гейт повторяется.
    """
    by_label = {label(p, multi): i for i, p in enumerate(srcs)}
    target = lambda i: (out / srcs[i].parent.name if multi else out) / srcs[i].name  # noqa: E731
    raw = None

    def similar() -> Weights:
        nonlocal raw
        if raw is None:
            print("Чиню пары за порогом: сравниваю исходники…", flush=True)
            raw = similarity(srcs, args.jobs)
        return raw

    rc_audio = 0
    if levers & {"pitch", "speed"}:
        audio_report = out / "audio_report.json"
        for rnd in range(REPAIR_ROUNDS + 1):
            print("\n── звук" + (f" (повтор {rnd})" if rnd else "") + " ──", flush=True)
            rc_audio = run_audio_gate(folders, args.against, audio_report)
            limit = cfg["gate"]["audio_fail_pct"]
            failed = [r for r in json.loads(audio_report.read_text()) if r["pct"] > limit]
            if not failed or rnd == REPAIR_ROUNDS:
                break
            vw = voice_weights(similar(), [p.stem for p in srcs])
            options, modes = voice_menu(cfg, levers)
            rows = json.loads(audio_report.read_text())
            changed = optimize_voice(per, rows, by_label, multi, vw, options, modes, limit)
            if not changed:
                print("Перебор голосов не нашёл улучшения — оставшиеся пары не разводятся.",
                      flush=True)
                break
            print(f"Перебор голосов: меняю голос у {len(changed)} роликов", flush=True)
            rerender(sorted(changed), srcs, per, target, out, multi, cfg)

    report = out / "dedup_report.json"
    rc = 1
    for rnd in range(REPAIR_ROUNDS + 1):
        print("\n── новые ролики между собой" + (f" (повтор {rnd})" if rnd else "") + " ──", flush=True)
        rc = run_gate(folders, threshold, cfg, report)
        failed = [p for p in json.loads(report.read_text())
                  if max(p["pct_a"], p["pct_b"]) > threshold]
        if not failed or "frame" not in levers or rnd == REPAIR_ROUNDS:
            break
        fix = repair_targets(failed, by_label, multi)
        pairs = [(_stem(f["a"], multi), _stem(f["b"], multi)) for f in failed]
        for i in fix:
            me = label(srcs[i], multi)
            partners = {by_label[b if a == me else a] for a, b in pairs if me in (a, b)}
            per[i] = relook(i, per, looks, similar(), partners)
        rerender(fix, srcs, per, target, out, multi, cfg)
    if args.against:
        print("\n── новые ролики против " + ", ".join(map(str, args.against)) + " ──", flush=True)
        rc |= run_gate(folders, threshold, cfg, out / "dedup_vs_against.json", args.against)
    return rc | rc_audio


def rerender(indices: list[int], srcs: list[Path], per: dict[int, Look], target,
             out: Path, multi: bool, cfg: dict) -> None:
    """Пересобрать ролики параллельно (как основная сборка) и сохранять план
    после КАЖДОГО — остановка посередине не оставит план и файлы врозь.

    Раньше починка пересобирала по одному: 337 роликов ≈ 11 часов.
    """
    import threading

    lock = threading.Lock()
    done = 0

    def one(i: int) -> None:
        nonlocal done
        render(srcs[i], target(i), per[i], cfg)
        with lock:
            done += 1
            save_plan(out, per, srcs, multi)
            print(f"  [{done}/{len(indices)}] пересобран {label(srcs[i], multi)}: "
                  f"{per[i].describe()}", flush=True)

    with ThreadPoolExecutor(cfg["render"]["jobs"]) as ex:
        list(ex.map(one, indices))


def broken(path: Path) -> bool:
    """Файл не читается (например, сборку прервали посреди записи)."""
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return r.returncode != 0 or not r.stdout.strip()


def run_audio_gate(folders: list[Path], originals: list[Path] | None, report: Path) -> int:
    cmd = [sys.executable, str(ROOT / "qa_audio.py"), *map(str, folders), "--report", str(report)]
    for ref in originals or []:
        cmd += ["--originals", str(ref)]
    return subprocess.run(cmd).returncode


def optimize_voice(per: dict[int, Look], rows: list[dict], by_label: dict[str, int],
                   multi: bool, fallback: Weights, options: list[tuple[float, float]],
                   modes: dict, limit: float, passes: int = 30) -> set[int]:
    """Перебор голосов по ЗАМЕРЕННОМУ сходству — без пересборки, в расчёте.

    Отчёт гейта даёт сходство каждой пары с общим голосом при её нынешних
    голосах. Делим на ``voice_overlap`` этих голосов — получаем, сколько голоса
    пара делит вообще (если голоса и так далеки, деление ненадёжно — тогда
    берётся оценка по именам). Дальше локальный поиск: каждому ролику из
    конфликтов — вариант, при котором предсказанных пар за порогом (с запасом)
    меньше всего; круги, пока что-то меняется. Пересобираются только
    изменившиеся ролики — один раз за круг гейта, а не по ролику на пару.

    Прежняя починка (по одному ролику на пару, оценка по именам) не сходилась:
    629 пар → 602 после круга — перекрашенный ролик садился на соседей.
    """
    target = limit * 0.8
    cur = {i: (l.pitch, l.speed) for i, l in per.items()}
    base: dict[int, dict[int, float]] = {}
    hot: set[int] = set()
    for r in rows:
        a, b = by_label.get(_stem(r["a"], multi)), by_label.get(_stem(r["b"], multi))
        if r["kind"] == "оригинал":
            if a is not None and r["pct"] > limit:
                hot.add(a)
            continue
        if a is None or b is None or a not in cur or b not in cur:
            continue
        ov = voice_overlap(*cur[a], *cur[b])
        shared = r["pct"] / ov if ov >= 0.1 else max(fallback[a].get(b, 0.0), r["pct"])
        shared = min(100.0, shared)
        base.setdefault(a, {})[b] = base.setdefault(b, {})[a] = shared
        if r["pct"] > limit:
            hot |= {a, b}

    def local(i: int, o: tuple[float, float]) -> tuple[int, float]:
        bad, total = 0, ORIGINAL_WEIGHT * voice_overlap(o[0], o[1], 1.0, 1.0)
        for j, sh in base.get(i, {}).items():
            pred = sh * voice_overlap(o[0], o[1], *cur[j])
            bad += pred > target
            total += pred
        return bad, total

    changed: set[int] = set()
    # Пересматриваются ВСЕ ролики с общим голосом, а не только конфликтные:
    # расчёт на 450 роликах — по конфликтным 497 → 494 пар, по всем — до ~200
    # (при скорости ±8%). Ролик, которому замена не помогает, остаётся как есть.
    for i, o in cur.items():
        if o not in options:          # голос из старых ступеней — пересмотреть
            hot.add(i)
    todo = sorted(set(base) | hot, key=lambda k: -len(base.get(k, {})))
    for _ in range(passes):
        moved = False
        for i in todo:
            best = min(options, key=lambda o: (local(i, o), o != cur[i]))
            if best != cur[i] and (local(i, best) < local(i, cur[i]) or cur[i] not in options):
                cur[i] = best
                changed.add(i)
                moved = True
        if not moved:
            break
    for i in changed:
        old = per[i]
        per[i] = Look(look=old.look, zoom=old.zoom, ox=old.ox, oy=old.oy, tilt=old.tilt,
                      speed=cur[i][1], gop=old.gop, pitch=cur[i][0],
                      voice=modes.get(cur[i], "r3"))
    return changed


def _stem(report_label: str, multi: bool) -> str:
    """«папка/имя.mp4» из отчёта гейта → метка ролика, как в плане."""
    folder, name = report_label.rsplit("/", 1)
    stem = name.rsplit(".", 1)[0]
    return f"{folder}/{stem}" if multi else stem


def repair_targets(failed: list[dict], by_label: dict[str, int], multi: bool) -> list[int]:
    """Кого пересобирать: по одному ролику из каждой пары, по возможности общему
    для нескольких пар (так одной пересборкой чинится сразу несколько)."""
    count: dict[str, int] = {}
    for f in failed:
        for side in (f["a"], f["b"]):
            count[_stem(side, multi)] = count.get(_stem(side, multi), 0) + 1
    chosen: list[str] = []
    for f in failed:
        a, b = _stem(f["a"], multi), _stem(f["b"], multi)
        if a in chosen or b in chosen:
            continue
        chosen.append(a if count[a] >= count[b] else b)
    return [by_label[c] for c in chosen]


def relook(i: int, per: dict[int, Look], looks: list[tuple], weights: Weights,
           partners: set[int]) -> Look:
    """Облик, где у ролика меньше всего общих кадров с соседями этого облика.

    Облики пар, с которыми ролик только что не прошёл гейт, исключаются.
    """
    cost = [0.0] * len(looks)
    for j, pct in weights[i].items():
        if j in per and per[j].look >= 0:
            cost[per[j].look] += pct
    banned = {per[j].look for j in partners if j in per} | {per[i].look}
    options = [c for c in range(len(looks)) if c not in banned] or list(range(len(looks)))
    c = min(options, key=lambda k: (cost[k], k))
    z, ox, oy, t = looks[c]
    return Look(look=c, zoom=z, ox=ox, oy=oy, tilt=t, speed=per[i].speed, gop=per[i].gop,
                pitch=per[i].pitch, voice=per[i].voice)


def save_plan(out: Path, per: dict[int, Look], srcs: list[Path], multi: bool) -> None:
    plan_file = out / "plan.json"
    data = json.loads(plan_file.read_text())
    data["videos"] = {label(srcs[i], multi): asdict(l) for i, l in sorted(per.items())}
    plan_file.write_text(json.dumps(data, ensure_ascii=False, indent=1))


def cmd_repair(args: argparse.Namespace, cfg: dict) -> int:
    """Починить уже собранную версию: гейт + пересборка роликов за порогом."""
    out = args.version.resolve()
    data = json.loads((out / "plan.json").read_text())
    srcs = collect(args.src, None)
    multi = len(args.src) > 1
    by_label = {label(p, multi): i for i, p in enumerate(srcs)}
    per = {by_label[k]: Look.from_plan(v) for k, v in data["videos"].items() if k in by_label}
    looks = [tuple(x) for x in data["looks"]]
    if any(x[1] > 1 or x[2] > 1 for x in looks):       # старый план: пиксели 1080×1920
        looks = [(z, x / 1080, y / 1920, t) for z, x, y, t in looks]
    levers = set(data["levers"])
    threshold = data.get("threshold", cfg["gate"]["fail_pct"]) if args.threshold is None else args.threshold
    folders = [out / f.name for f in args.src] if multi else [out]
    target = lambda i: (out / srcs[i].parent.name if multi else out) / srcs[i].name  # noqa: E731
    bad = [i for i in per if not target(i).exists() or broken(target(i))]
    if bad:
        print(f"Недописанных или пропавших файлов: {len(bad)} — пересобираю по плану", flush=True)
        rerender(bad, srcs, per, target, out, multi, cfg)
    return gate_and_repair(out, folders, srcs, per, looks, threshold, levers, cfg, args, multi)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capacity", help="сколько роликов можно сделать уникальными")
    c.add_argument("src", type=Path, nargs="+")
    c.add_argument("--limit", type=int)
    c.add_argument("--profile", help="профиль рамки из config.yaml (reels, raw…)")
    c.add_argument("--jobs", type=int, default=10)
    c.add_argument("--thresholds", type=float, nargs="+", default=[0, 5, 10, 20, 30])

    r = sub.add_parser("run", help="уникализировать и проверить гейтом")
    r.add_argument("src", type=Path, nargs="+",
                   help="папки с готовыми роликами; несколько — уникализируются вместе")
    r.add_argument("--name", required=True, help="папка в output/ (версии v1, v2…)")
    r.add_argument("--count", type=int, help="сколько роликов выдать (по умолчанию — сколько выйдет)")
    r.add_argument("--levers", default=DEFAULT_LEVERS,
                   help="рамка,наклон,скорость,тон — через запятую (по умолчанию %(default)s)")
    r.add_argument("--threshold", type=float,
                   help="пара не проходит гейт при > N%% общих кадров (по умолчанию gate.fail_pct из config.yaml)")
    r.add_argument("--allow-shared", action="store_true",
                   help="добрать до --count роликами с общим обликом (гейт их, скорее всего, не пропустит)")
    r.add_argument("--against", type=Path, action="append",
                   help="ещё сравнить с роликами из этой папки (например, уже выложенными)")
    r.add_argument("--limit", type=int, help="брать не больше N роликов из каждой папки")
    r.add_argument("--profile", help="профиль рамки из config.yaml: reels — готовые рилсы "
                                     "(можно срезать зоны интерфейса), raw — сырое видео")
    r.add_argument("--jobs", type=int, default=10, help="процессов для хэширования")
    r.add_argument("--dry-run", action="store_true", help="только показать план")
    r.add_argument("--captions-on-top", type=Path, metavar="FONTS_DIR",
                   help="титры поверх ПОСЛЕ рамки: брать версию без титров из <папка>/_clean/ "
                        "и .ass оттуда же; FONTS_DIR — шрифты титров (montazh/assets/fonts)")

    f = sub.add_parser("repair", help="починить собранную версию: пары за порогом пересобрать")
    f.add_argument("version", type=Path, help="папка версии, например output/bot_уник/v1")
    f.add_argument("src", type=Path, nargs="+", help="те же папки исходников, что при сборке")
    f.add_argument("--threshold", type=float)
    f.add_argument("--against", type=Path, action="append")
    f.add_argument("--jobs", type=int, default=10)
    f.add_argument("--captions-on-top", type=Path, metavar="FONTS_DIR")

    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if getattr(args, "captions_on_top", None):
        cfg["_captions_fonts"] = str(args.captions_on_top.resolve())
    if args.cmd == "repair":
        return cmd_repair(args, cfg)
    if args.cmd == "capacity":
        return cmd_capacity(args, cfg)
    return cmd_run(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
