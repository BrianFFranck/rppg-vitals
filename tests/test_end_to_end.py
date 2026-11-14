"""Full pipeline on the synthetic face video: MediaPipe tracking -> ROI ->
POS -> HR. Slow-ish (~15 s); skip with `pytest -m "not slow"`."""

import numpy as np
import pytest

from rppg.estimator import EstimatorConfig, VitalsEstimator
from rppg.synthetic import DEFAULT_IMAGE, SyntheticSpec, synthetic_frames

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def tracker():
    if not DEFAULT_IMAGE.exists():
        pytest.skip("assets/astronaut.png missing")
    from rppg.face import FaceROITracker

    with FaceROITracker() as tr:
        yield tr


def test_pipeline_tracks_known_heart_rate_and_rejects_shake(tracker):
    spec = SyntheticSpec(duration=20, hr_variation_bpm=0, shake=(14, 17))
    est = VitalsEstimator(EstimatorConfig(method="POS"))
    statuses = {}
    for i, (t, frame) in enumerate(synthetic_frames(spec)):
        obs = tracker.process(frame, t)
        assert obs is not None, f"face lost at t={t:.2f}"
        est.push_observation(t, obs)
        if i % 30 == 29:
            statuses[round(t)] = est.update()

    s12 = statuses[12]  # clean 10 s window before the shake
    assert s12.hr.status == "ok"
    assert abs(s12.hr.value - spec.hr_bpm) < 1.5
    assert s12.hr.snr_db > 3

    s17 = statuses[17]  # window containing the shake
    assert s17.hr.status == "motion"
    assert s17.hr_bpm is None
