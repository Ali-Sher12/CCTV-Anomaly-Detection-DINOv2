import pytest
from capture_registry import normalize_source, CaptureRegistry

def test_source_normalization():
    assert normalize_source(0) == 0
    assert normalize_source("0") == 0
    assert normalize_source(" 1 ") == 1
    assert normalize_source("rtsp://example.com/live") == "rtsp://example.com/live"
    assert normalize_source("http://192.168.1.5:8080") == "http://192.168.1.5:8080/video"
    assert normalize_source("http://192.168.1.5:8080/") == "http://192.168.1.5:8080/video"
    assert normalize_source("video.mp4") == "video.mp4"

    with pytest.raises(ValueError):
        normalize_source("")

    with pytest.raises(ValueError):
        normalize_source(None)

    with pytest.raises(ValueError):
        normalize_source("   ")
