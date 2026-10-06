"""Свойства уникализатора, которые ломаются молча: файлы собираются, а гейт
через полчаса рендера показывает пары за порогом или срезанные титры."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from qa_dedup import distances  # noqa: E402
from uniquify import (  # noqa: E402
    FRAMINGS,
    assign_framings,
    blocks_of,
    conflicts,
    framing_is_safe,
    shared_framing_pairs,
    speed_for,
)


def test_рамки_не_режут_титры_бот_работ():
    """Все рамки набора безопасны для титров 95–994 px с полем 8 px."""
    unsafe = [f for f in FRAMINGS if not framing_is_safe(f, 95, 994, 8)]
    assert not unsafe, f"рамки срезают титры: {unsafe}"


def test_рамки_не_повторяются():
    assert len(set(FRAMINGS)) == len(FRAMINGS)


def test_имя_разбирается_на_куски():
    assert blocks_of("h3_t7_c4") == {"h3", "t7", "c4"}
    assert blocks_of("IMG_0123") is None


def test_общий_кусок_делает_соседями():
    adj = conflicts(["h1_t1_c1", "h1_t2_c3", "h2_t3_c4"])
    assert adj[0] == {1}
    assert adj[2] == set()


def _grid_weights():
    """Сетка h×t×c, как у «бот работ»: 200 роликов, у пары с общим куском вес 1."""
    stems = [f"h{h}_t{t}_c{c}" for h in range(1, 11) for t in range(1, 11)
             for c in ((h + t - 2) % 10 + 1, (h + t + 3) % 10 + 1)]
    adj = conflicts(stems)
    return [{j: 1.0 for j in a} for a in adj]


def test_раскраска_сетки_без_общих_рамок():
    """На сетке по именам 33 рамок хватает с запасом — ни одной общей пары."""
    w = _grid_weights()
    colors = assign_framings(w, len(FRAMINGS))
    assert not shared_framing_pairs(w, colors)


def test_нехватка_рамок_отдаёт_общую_самой_слабой_паре():
    # Треугольник, две рамки: общую рамку должна получить пара с весом 1, а не 10.
    w = [{1: 10.0, 2: 10.0}, {0: 10.0, 2: 1.0}, {0: 10.0, 1: 1.0}]
    colors = assign_framings(w, 2)
    shared = shared_framing_pairs(w, colors)
    assert [(i, j) for i, j, _ in shared] == [(1, 2)]


def test_скорость_в_коридоре_и_мимо_мёртвой_зоны():
    for i in range(300):
        v = speed_for(i, 0.97, 1.03, 0.01)
        assert 0.97 <= v <= 1.03
        assert abs(v - 1.0) >= 0.01 - 1e-9


def test_расстояние_хэмминга():
    a = np.array([[0, 1, 1, 0]], dtype=np.uint8)
    b = np.array([[1, 1, 0, 0], [0, 1, 1, 0]], dtype=np.uint8)
    assert distances(a, b).tolist() == [[2, 0]]
