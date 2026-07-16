#!/usr/bin/env python3
"""Smoke test for ArtifactTracker (run directly, no test framework): python3 test_tracker.py"""
import sys
sys.path.insert(0, 'src')

from automata_detection.tracker import ArtifactTracker


def detection(x, y, height=None):
    return {"centroid": (x, y), "contour": None, "segment": None, "height_mm": height}


def test_stable_id_under_jitter():
    tracker = ArtifactTracker(match_distance_px=20, miss_limit=3)
    added, removed, _ = tracker.update([detection(100, 100)])
    assert added == [1] and removed == []
    track_id = added[0]

    # Small jitter each cycle: the same artifact must keep the same id.
    for offset in (2, -3, 1, 4):
        added, removed, _ = tracker.update([detection(100 + offset, 100)])
        assert added == [] and removed == []
        assert list(tracker.tracks.keys()) == [track_id]

    print("OK: stable id under jitter")


def test_new_id_for_far_detection():
    tracker = ArtifactTracker(match_distance_px=20, miss_limit=3)
    tracker.update([detection(100, 100)])
    added, removed, _ = tracker.update([detection(100, 100), detection(500, 500)])
    assert added == [2]

    print("OK: new id for a far detection")


def test_removed_only_after_miss_limit():
    tracker = ArtifactTracker(match_distance_px=20, miss_limit=3)
    added, _, _ = tracker.update([detection(100, 100)])
    track_id = added[0]

    for _ in range(3):
        _, removed, _ = tracker.update([])
        assert removed == []

    _, removed, _ = tracker.update([])
    assert removed == [track_id]

    print("OK: removed only after miss_limit consecutive misses")


def test_reappearance_gets_new_id_without_reacquire():
    # reacquire_window_s defaults to 0: reacquisition is disabled, so the old
    # strict rule applies (any reappearance gets a brand new id).
    tracker = ArtifactTracker(match_distance_px=20, miss_limit=1)
    added, _, _ = tracker.update([detection(100, 100)])
    old_id = added[0]

    tracker.update([])  # miss 1
    tracker.update([])  # miss 2 -> removed (miss_limit=1)
    added, _, _ = tracker.update([detection(100, 100)])
    assert added != [old_id]

    print("OK: reappearance gets a new id when reacquisition is disabled")


def test_reacquire_same_spot_keeps_id_and_pose():
    tracker = ArtifactTracker(match_distance_px=20, miss_limit=1, reacquire_window_s=10.0)
    added, _, _ = tracker.update([detection(100, 100)])
    old_id = added[0]
    # Give the track a frozen pose, like inference.py does.
    tracker.tracks[old_id].pose = {"x_1": 1.0}
    tracker.tracks[old_id].pose_valid = True

    tracker.update([])  # miss 1
    _, removed, _ = tracker.update([])  # miss 2 -> removed
    assert removed == [old_id]

    # The detection flickers back at (almost) the same spot: same id, same pose.
    added, _, ids = tracker.update([detection(103, 100)])
    assert added == [old_id]
    assert ids == [old_id]
    assert tracker.tracks[old_id].pose == {"x_1": 1.0}
    assert tracker.tracks[old_id].pose_valid

    print("OK: reacquisition at the same spot keeps the old id and pose")


def test_reacquire_far_away_gets_new_id():
    tracker = ArtifactTracker(match_distance_px=20, miss_limit=1, reacquire_window_s=10.0)
    added, _, _ = tracker.update([detection(100, 100)])
    old_id = added[0]

    tracker.update([])  # miss 1
    tracker.update([])  # miss 2 -> removed

    # Reappears far from the old spot: it was physically moved, new id.
    added, _, _ = tracker.update([detection(500, 500)])
    assert added != [old_id]

    print("OK: reappearance far away still gets a new id")


def test_reacquire_window_expires():
    import time

    tracker = ArtifactTracker(match_distance_px=20, miss_limit=1, reacquire_window_s=0.05)
    added, _, _ = tracker.update([detection(100, 100)])
    old_id = added[0]

    tracker.update([])  # miss 1
    tracker.update([])  # miss 2 -> removed
    time.sleep(0.1)  # wait until the reacquisition window is over

    added, _, _ = tracker.update([detection(100, 100)])
    assert added != [old_id]

    print("OK: reacquisition window expires")


def test_freeze_misses_keeps_covered_tracks_alive():
    tracker = ArtifactTracker(match_distance_px=20, miss_limit=1)
    added, _, _ = tracker.update([detection(100, 100)])
    track_id = added[0]

    # The scene is occluded (a hand over the table) for many cycles: the covered
    # artifact must not count misses and must never expire.
    for _ in range(10):
        _, removed, _ = tracker.update([], freeze_misses=True)
        assert removed == []
    assert tracker.tracks[track_id].miss_count == 0

    # Once the scene is clear again, misses count normally.
    tracker.update([])  # miss 1
    _, removed, _ = tracker.update([])  # miss 2 -> removed
    assert removed == [track_id]

    print("OK: freeze_misses keeps covered tracks alive")


def test_phantom_flicker_never_creates_id():
    tracker = ArtifactTracker(match_distance_px=20, miss_limit=3, confirm_hits=3)
    # A phantom detected for 2 cycles, then gone: no id must ever be created.
    added, _, ids = tracker.update([detection(100, 100)])
    assert added == [] and ids == [None]
    added, _, ids = tracker.update([detection(101, 100)])
    assert added == [] and ids == [None]
    tracker.update([])  # gone -> the candidate is discarded
    assert tracker.tracks == {} and tracker.pending == []

    print("OK: a 2-cycle phantom never creates an id")


def test_confirmed_after_enough_hits():
    tracker = ArtifactTracker(match_distance_px=20, miss_limit=3, confirm_hits=3)
    tracker.update([detection(100, 100)])
    tracker.update([detection(100, 100)])
    added, _, ids = tracker.update([detection(100, 100)])  # 3rd consecutive hit
    assert added == [1] and ids == [1]
    assert list(tracker.tracks.keys()) == [1]

    print("OK: a real artifact is confirmed after confirm_hits cycles")


def test_gap_resets_confirmation():
    tracker = ArtifactTracker(match_distance_px=20, miss_limit=3, confirm_hits=3)
    tracker.update([detection(100, 100)])
    tracker.update([detection(100, 100)])
    tracker.update([])  # one-cycle gap: candidate discarded
    tracker.update([detection(100, 100)])
    added, _, _ = tracker.update([detection(100, 100)])
    assert added == []  # only 2 hits since the gap, not confirmed yet
    added, _, _ = tracker.update([detection(100, 100)])
    assert added == [1]

    print("OK: a gap during confirmation restarts the count")


def test_reacquire_happens_at_confirmation():
    tracker = ArtifactTracker(match_distance_px=20, miss_limit=1, reacquire_window_s=10.0, confirm_hits=2)
    tracker.update([detection(100, 100)])
    added, _, _ = tracker.update([detection(100, 100)])  # confirmed
    old_id = added[0]
    tracker.tracks[old_id].pose = {"x_1": 1.0}
    tracker.tracks[old_id].pose_valid = True

    tracker.update([])  # miss 1
    _, removed, _ = tracker.update([])  # miss 2 -> removed
    assert removed == [old_id]

    # Reappears at the same spot: the old id comes back only at confirmation.
    added, _, _ = tracker.update([detection(100, 100)])
    assert added == []
    added, _, ids = tracker.update([detection(100, 100)])
    assert added == [old_id] and ids == [old_id]
    assert tracker.tracks[old_id].pose == {"x_1": 1.0}

    print("OK: reacquisition happens at confirmation, pose preserved")


def test_flat_phantom_rejected_by_height():
    # Flat blob (1 mm above surroundings): never confirmed.
    tracker = ArtifactTracker(match_distance_px=20, miss_limit=3, confirm_hits=2, min_height_mm=5.0)
    tracker.update([detection(100, 100, height=1.0)])
    added, _, _ = tracker.update([detection(100, 100, height=1.0)])
    assert added == [] and tracker.tracks == {}

    # Raised blob (10 mm): confirmed.
    tracker = ArtifactTracker(match_distance_px=20, miss_limit=3, confirm_hits=2, min_height_mm=5.0)
    tracker.update([detection(100, 100, height=10.0)])
    added, _, _ = tracker.update([detection(100, 100, height=10.0)])
    assert added == [1]

    # No depth data (height None): cannot judge, must be kept.
    tracker = ArtifactTracker(match_distance_px=20, miss_limit=3, confirm_hits=2, min_height_mm=5.0)
    tracker.update([detection(100, 100)])
    added, _, _ = tracker.update([detection(100, 100)])
    assert added == [1]

    print("OK: flat phantoms rejected by height, unjudgeable blobs kept")


def test_freeze_keeps_pending_candidates():
    tracker = ArtifactTracker(match_distance_px=20, miss_limit=3, confirm_hits=3)
    tracker.update([detection(100, 100)])
    tracker.update([detection(100, 100)])
    # Scene occluded for a few cycles with no detections: the candidate must
    # survive untouched (no drop, no progress).
    for _ in range(3):
        added, _, _ = tracker.update([], freeze_misses=True)
        assert added == []
    assert len(tracker.pending) == 1

    # Scene clear again: the third hit confirms it.
    added, _, _ = tracker.update([detection(100, 100)])
    assert added == [1]

    print("OK: freeze keeps pending candidates untouched")


def test_split_blob_is_not_a_new_track():
    tracker = ArtifactTracker(match_distance_px=20, miss_limit=3)
    tracker.update([detection(100, 100)])
    # Two close detections in the same cycle: the second is a split of the same blob.
    added, _, ids = tracker.update([detection(102, 100), detection(108, 100)])
    assert added == []
    assert len(tracker.tracks) == 1
    assert ids[0] is not None and ids[1] is None

    print("OK: split blob does not spawn a duplicate track")


if __name__ == "__main__":
    try:
        test_stable_id_under_jitter()
        test_new_id_for_far_detection()
        test_removed_only_after_miss_limit()
        test_reappearance_gets_new_id_without_reacquire()
        test_reacquire_same_spot_keeps_id_and_pose()
        test_reacquire_far_away_gets_new_id()
        test_reacquire_window_expires()
        test_freeze_misses_keeps_covered_tracks_alive()
        test_phantom_flicker_never_creates_id()
        test_confirmed_after_enough_hits()
        test_gap_resets_confirmation()
        test_reacquire_happens_at_confirmation()
        test_flat_phantom_rejected_by_height()
        test_freeze_keeps_pending_candidates()
        test_split_blob_is_not_a_new_track()
        print("\nSUCCESS: all tracker tests passed!")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
