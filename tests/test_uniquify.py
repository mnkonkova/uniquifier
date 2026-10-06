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
    speed_for,
)

LIM = palette.Limits()


def test_кандидаты_не_режут_титры_и_не_открывают_углы():
    for look in palette.candidates(tilt=True, lim=LIM):
        assert palette.no_black_corners(look), look
        l, tp, r, b = LIM.text_box
        for px in (l, r):
            for py in (tp, b):
                u, _ = palette.to_out(px, py, look)
                assert LIM.text_margin <= u <= 1080 - LIM.text_margin, (look, px, py, u)


def test_наклон_даёт_больше_кандидатов():
    assert len(palette.candidates(True, LIM)) > len(palette.candidates(False, LIM))


def test_без_рамки_оригинал_на_месте():
    assert palette.to_out(100, 200, (1.0, 0, 0, 0.0)) == (100, 200)


def test_имя_разбирается_на_куски():
    assert blocks_of("h3_t7_c4") == {"h3", "t7", "c4"}
    assert blocks_of("IMG_0123") is None


def test_общий_кусок_делает_соседями():
    adj = conflicts(["h1_t1_c1", "h1_t2_c3", "h2_t3_c4"])
    assert adj[0] == {1}
    assert adj[2] == set()


def test_рычаги_по_русски_и_наклон_тянет_рамку():
    assert parse_levers("наклон,шум") == {"tilt", "frame", "noise"}


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


def test_скорость_в_коридоре_и_мимо_мёртвой_зоны():
    for i in range(300):
        v = speed_for(i, 0.97, 1.03, 0.01)
        assert 0.97 <= v <= 1.03
        assert abs(v - 1.0) >= 0.01 - 1e-9


def test_расстояние_хэмминга():
    a = np.array([[0, 1, 1, 0]], dtype=np.uint8)
    b = np.array([[1, 1, 0, 0], [0, 1, 1, 0]], dtype=np.uint8)
    assert distances(a, b).tolist() == [[2, 0]]
