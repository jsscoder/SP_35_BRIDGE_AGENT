import pytest

from vision.tiling import tile_boxes


def covered(w, h, boxes):
    grid = [[False] * w for _ in range(h)]
    for x0, y0, x1, y1 in boxes:
        for y in range(y0, y1):
            for x in range(x0, x1):
                grid[y][x] = True
    return all(all(r) for r in grid)


@pytest.mark.parametrize("w,h", [(30, 40), (10, 10), (25, 7)])
def test_tiles_cover_the_image(w, h):
    boxes = tile_boxes(w, h, size=10, overlap=0.2)
    assert covered(w, h, boxes)
    assert all(x1 - x0 <= 10 and y1 - y0 <= 10 for x0, y0, x1, y1 in boxes)


def test_12mp_photo_tiling():
    boxes = tile_boxes(3024, 4032, size=1008, overlap=0.2)
    assert len(boxes) == 20 and boxes[-1] == (2016, 3024, 3024, 4032)


def test_small_image_is_one_tile():
    assert tile_boxes(800, 600) == [(0, 0, 800, 600)]
