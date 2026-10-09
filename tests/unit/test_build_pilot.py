from pathlib import Path

from PIL import Image, ImageDraw

from ingest.build_pilot import Item, classify, dhash, hamming


def _img(seed: int) -> Image.Image:
    im = Image.new("L", (160, 120), 128)
    d = ImageDraw.Draw(im)
    for i in range(6):
        x = (seed * 37 + i * 23) % 140
        d.rectangle([x, i * 18, x + 20, i * 18 + 12], fill=(seed * 50 + i * 30) % 255)
    return im


def test_dhash_detects_resaves_not_different_photos():
    a = _img(1)
    resaved = a.resize((320, 240)).resize((160, 120))
    assert hamming(dhash(a), dhash(resaved)) <= 20
    assert hamming(dhash(a), dhash(_img(2))) > 20


def _item(name, sha="x", make="Apple", xmp="", h=0, taken=""):
    return Item(path=Path(name), sha256=sha, width=10, height=10, make=make, model="", taken=taken,
                gps=False, xmp_source=xmp, dhash=h)


def test_classification_rules():
    items = [
        _item("IMG_1.jpg", sha="a", h=0),
        _item("IMG_1.HEIC.jpg", sha="b", h=0),                       # twin of IMG_1.jpg
        _item("IMG_2.jpg", sha="c", h=(1 << 200) - 1),
        _item("IMG_2_1.jpg", sha="c", h=(1 << 200) - 1),             # exact duplicate
        _item("IMG_3.jpg", sha="d", h=1, taken="2026:03:15 10:00:01"),  # near-duplicate of IMG_1
        _item("abc.JPG", sha="e", make="", xmp="InfraredCamera", h=5),
        _item("Screenshot_2026-03-18_100509.png", sha="f", make="", h=7),
        _item("Screenshot_2026-02-27_201207.png", sha="g", make="", h=9),
    ]
    classify(items)
    by = {i.path.name: i for i in items}
    assert by["IMG_1.jpg"].keep and by["IMG_2.jpg"].keep
    assert by["IMG_1.HEIC.jpg"].reason == "HEIC-conversion twin"
    assert by["IMG_2_1.jpg"].reason == "exact duplicate"
    assert by["IMG_3.jpg"].reason.startswith("near-duplicate") and by["IMG_3.jpg"].duplicate_of == "IMG_1.jpg"
    assert by["abc.JPG"].category == "thermal" and not by["abc.JPG"].keep
    assert by["Screenshot_2026-03-18_100509.png"].category == "streetview"
    assert by["Screenshot_2026-02-27_201207.png"].category == "unrelated"
