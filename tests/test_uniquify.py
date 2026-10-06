"""Свойства уникализатора, которые ломаются молча: файлы собираются, а гейт
через полчаса рендера показывает пары за порогом или срезанные титры."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import palette  # noqa: E402
from qa_dedup import distances  # noqa: E402
from uniquify import (  # noqa: E402
    blocks_of,
    conflicts,
    fill_shared,
    parse_levers,
    select_strict,
    shared_pairs,
)

REELS = palette.Limits(text_box=(0.088, 0.680, 0.920, 0.843), text_margin=0.0074,
                       max_top=0.078, max_bottom=0.120)
RAW = palette.Limits(max_top=0.03, max_bottom=0.03, max_side=0.03)
VERTICAL, HORIZONTAL = 1080 / 1920, 1920 / 1080


def test_облики_рилсов_не_режут_титры_и_не_открывают_углы():
    for look in palette.candidates(tilt=True, lim=REELS, aspects=[VERTICAL]):
        assert palette.no_black_corners(look, VERTICAL), look
        l, tp, r, b = REELS.text_box
        for u in (l, r):
            for v in (tp, b):
                x, _ = palette.to_out(u, v, look, VERTICAL)
                assert REELS.text_margin <= x <= 1 - REELS.text_margin, (look, u, v, x)


def test_облики_сырого_видео_режут_не_больше_3_процентов():
    looks = palette.candidates(tilt=False, lim=RAW, aspects=[VERTICAL, HORIZONTAL])
    assert looks
    for look in looks:
        for a in (VERTICAL, HORIZONTAL):
            assert -palette.to_out(0.5, 0, look, a)[1] <= 0.03 + 1e-9
            assert palette.to_out(1, 0.5, look, a)[0] - 1 <= 0.03 + 1e-9


def test_наклон_даёт_больше_кандидатов():
    assert (len(palette.candidates(True, REELS, [VERTICAL]))
            > len(palette.candidates(False, REELS, [VERTICAL])))


def test_без_рамки_оригинал_на_месте():
    x, y = palette.to_out(0.1, 0.2, (1.0, 0.0, 0.0, 0.0), HORIZONTAL)
    assert abs(x - 0.1) < 1e-9 and abs(y - 0.2) < 1e-9


def test_горизонтальное_видео_не_сплющивается():
    from uniquify import Look, Media, build_filter

    look = Look(look=0, zoom=1.1, ox=0.05, oy=0.05, tilt=0.0, speed=1.0, gop=15)
    graph = build_filter(look, Media(1920, 1080, "25/1", True))
    assert "scale=2112:1188" in graph          # оба измерения ×1.1
    assert "crop=1920:1080:96:54" in graph     # выход того же размера, что исходник
    assert "fps=25/1" in graph


def test_видео_без_звука_не_требует_дорожку():
    from uniquify import Look, Media, render_cmd

    look = Look(look=-1, zoom=1.0, ox=0.0, oy=0.0, tilt=0.0, speed=1.0, gop=15)
    cfg = {"render": {"preset": "fast", "crf": 20}}
    cmd = [str(c) for c in render_cmd(Path("a.mp4"), Path("b.mp4"), look,
                                      Media(1920, 1080, "30/1", False), cfg)]
    assert "[a]" not in cmd and "[0:a]" not in " ".join(cmd)


def test_старый_план_в_пикселях_читается():
    from uniquify import Look

    look = Look.from_plan({"look": 1, "zoom": 1.1, "x": 108, "y": 192, "tilt": 0.0,
                           "speed": 1.0, "gop": 15, "noise": None, "noise_opacity": 0.0})
    assert (look.ox, look.oy) == (0.1, 0.1)


def test_имя_разбирается_на_куски():
    assert blocks_of("h3_t7_c4") == {"h3", "t7", "c4"}
    assert blocks_of("IMG_0123") is None


def test_общий_кусок_делает_соседями():
    adj = conflicts(["h1_t1_c1", "h1_t2_c3", "h2_t3_c4"])
    assert adj[0] == {1}
    assert adj[2] == set()


def test_рычаги_по_русски_и_наклон_тянет_рамку():
    assert parse_levers("наклон,скорость") == {"tilt", "frame", "speed"}


def _clique(n):
    return [{j: 1.0 for j in range(n) if j != i} for i in range(n)]


def test_строгий_отбор_не_даёт_похожим_один_облик():
    w = _clique(5)
    chosen = select_strict(w, 3, None)
    assert len(chosen) == 3 and not shared_pairs(w, chosen)


def test_отбор_останавливается_на_count():
    w = [dict() for _ in range(10)]
    assert len(select_strict(w, 1, 4)) == 4


def test_добор_отдаёт_общий_облик_самой_слабой_паре():
    # Треугольник, два облика: общий должна получить пара с весом 1, а не 10.
    w = [{1: 10.0, 2: 10.0}, {0: 10.0, 2: 1.0}, {0: 10.0, 1: 1.0}]
    chosen = select_strict(w, 2, None)
    full = fill_shared(w, 2, chosen, 3)
    assert [(i, j) for i, j, _ in shared_pairs(w, full)] == [(1, 2)]


def test_расстояние_хэмминга():
    a = np.array([[0, 1, 1, 0]], dtype=np.uint8)
    b = np.array([[1, 1, 0, 0], [0, 1, 1, 0]], dtype=np.uint8)
    assert distances(a, b).tolist() == [[2, 0]]


def test_починка_берёт_ролик_общий_для_нескольких_пар():
    from uniquify import repair_targets

    failed = [{"a": "p/x.mp4", "b": "p/y.mp4"}, {"a": "p/y.mp4", "b": "p/z.mp4"}]
    idx = {"p/x": 0, "p/y": 1, "p/z": 2}
    assert repair_targets(failed, idx, multi=True) == [1]


def test_голос_разводит_самую_похожую_пару_дальше_всего():
    from uniquify import assign_voice, voice_overlap

    w = [{1: 90.0, 2: 5.0}, {0: 90.0}, {0: 5.0}]
    v = assign_voice(w, [0, 1, 2], [0.97, 1.03], [0.96, 1.0, 1.04])
    assert voice_overlap(*v[0], *v[1]) < 0.05


def test_шесть_роликов_с_общим_голосом_получают_разные_варианты():
    from uniquify import assign_voice

    w = [{j: 75.0 for j in range(6) if j != i} for i in range(6)]
    v = assign_voice(w, list(range(6)), [0.94, 0.97, 1.03, 1.06], [0.96, 1.0, 1.04])
    assert len(set(v.values())) == 6


def test_голос_не_выдаётся_слишком_близким_к_оригиналу():
    from uniquify import MAX_OWN_OVERLAP, voice_options, voice_overlap

    opts = voice_options([0.94, 0.97, 1.03, 1.06], [0.96, 1.0, 1.04])
    assert (0.97, 1.0) not in opts and (1.03, 1.0) not in opts
    assert all(voice_overlap(p, s, 1.0, 1.0) <= MAX_OWN_OVERLAP for p, s in opts)


def test_общий_кусок_по_имени_даёт_вес_голосу():
    from uniquify import voice_weights

    w = voice_weights([dict(), dict(), dict()], ["h1_t1_c1", "h1_t1_c2", "h2_t3_c4"])
    assert w[0][1] == 75.0 and 2 not in w[0]


def test_рычаг_тон_по_русски():
    assert parse_levers("рамка,тон") == {"frame", "pitch"}
