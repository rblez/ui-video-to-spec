"""Paso 5: ajuste de easing y medición de animaciones del video ejemplo."""

from ui_video_to_spec.extract_frames import extract_frames
from ui_video_to_spec.measure_motion import bezier_y, fit_easing, measure_motion
from ui_video_to_spec.measure_static import measure_static
from ui_video_to_spec.types import Config


def test_bezier_endpoints():
    p = (0.22, 1.0, 0.36, 1.0)
    assert bezier_y(0.0, p) == 0.0
    assert bezier_y(1.0, p) == 1.0
    assert 0.5 < bezier_y(0.25, p) < 1.0  # ease-out fuerte: avanza rápido


def test_fit_easing_recovers_curve():
    true = (0.22, 1.0, 0.36, 1.0)
    ts = [i / 20 for i in range(21)]
    ps = [bezier_y(t, true) for t in ts]
    params, name, err = fit_easing(ts, ps)
    assert err < 0.02
    for got, want in zip(params, true):
        assert abs(got - want) <= 0.18, (params, true)


def test_measure_motion_on_example(example_video, tmp_path, expected):
    cfg = Config(fps_static=2, fps_motion=24)
    meta, frames, _ = extract_frames(example_video, tmp_path / "w", cfg)
    m = measure_static(frames, meta, cfg)
    motion = [f for f in frames if f.kind == "motion"]
    animations = measure_motion(
        motion, m.elements, m.palette[0].hex, cfg, meta.width)
    assert animations, "el botón animado debe producir animaciones"
    props = {a.property for a in animations}
    assert "opacity" in props or "translateY" in props
    best = min(animations, key=lambda a: abs(a.duration_ms - 600))
    assert abs(best.duration_ms - expected["animation"]["duration_ms"]) <= 350
    assert abs(best.delay_ms - expected["animation"]["delay_ms"]) <= 400
