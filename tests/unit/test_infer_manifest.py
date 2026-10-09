import pytest

from vision.infer import photos_from_manifest


def _manifest(tmp_path, rows):
    m = tmp_path / "manifest.csv"
    m.write_text("file,category,keep,reason\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return m


def test_only_kept_photos_are_returned(tmp_path):
    src = tmp_path / "photos"
    src.mkdir()
    for name in ("IMG_1.jpg", "IMG_2.jpg", "IMG_1.HEIC.jpg"):
        (src / name).write_bytes(b"x")
    m = _manifest(tmp_path, ["IMG_1.jpg,iphone,yes,", 'IMG_1.HEIC.jpg,iphone,no,"HEIC-conversion twin"',
                             "IMG_2.jpg,iphone,yes,"])
    assert [p.name for p in photos_from_manifest(m, src)] == ["IMG_1.jpg", "IMG_2.jpg"]


def test_missing_photo_stops_the_run(tmp_path):
    src = tmp_path / "photos"
    src.mkdir()
    m = _manifest(tmp_path, ["IMG_9.jpg,iphone,yes,"])
    with pytest.raises(SystemExit, match="missing"):
        photos_from_manifest(m, src)
