import time
from typing import Dict, List, Optional, Tuple


class Track:
    """One tracked artifact: its identity, its last known position, and its pose."""

    def __init__(self, track_id: int, centroid: Tuple[int, int], contour, segment: Dict[str, Tuple[int, int]]) -> None:
        self.id = track_id
        self.centroid = centroid
        self.contour = contour
        self.segment = segment  # full-frame grasp segment endpoints, kept to (re)estimate the pose
        self.pose: Optional[Dict[str, float]] = None
        self.pose_valid = False
        self.miss_count = 0


def _distance(point_a: Tuple[float, float], point_b: Tuple[float, float]) -> float:
    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    return (dx * dx + dy * dy) ** 0.5


class ArtifactTracker:
    """Matches new detections to previously tracked artifacts, by centroid distance.

    A detection is a dict: {"centroid": (x, y), "contour": ..., "segment": {"1": (x, y), "2": (x, y)},
    "height_mm": float or None}. All coordinates are full-frame pixels, so tracking
    is not affected by where the crop currently is. The tracker only decides
    identity and position; it never touches the pose of an artifact that was
    already there - that is left to the caller (see inference.py), which is what
    keeps a placed artifact's pose fixed once estimated.

    A brand new detection does NOT get an id right away: it becomes a "pending"
    candidate and must be detected confirm_hits cycles in a row first. BiRefNet
    sometimes hallucinates objects out of the empty mat's texture for a cycle or
    two - those phantoms never survive the confirmation, so they never reach
    MQTT or the grid. A candidate that skips even one cycle is discarded.

    At confirmation, two more things happen:
    - if min_height_mm > 0 and the candidate's average height over its local
      surroundings (accumulated from detection["height_mm"]) is below it, the
      candidate is silently dropped: it is flat "paint" on the mat, not an object;
    - otherwise, a recently removed track near the same spot is reacquired (same
      id and pose - a fragment whose segmentation flickered off and on), or a
      brand new id is assigned. A fragment that reappears somewhere else still
      gets a new id.
    """

    def __init__(self, match_distance_px: float, miss_limit: int, reacquire_window_s: float = 0.0, confirm_hits: int = 1, min_height_mm: float = 0.0) -> None:
        self.match_distance_px = match_distance_px
        self.miss_limit = miss_limit
        self.reacquire_window_s = reacquire_window_s
        self.confirm_hits = confirm_hits
        self.min_height_mm = min_height_mm
        self.tracks: Dict[int, Track] = {}
        # Recently removed tracks that can still be reacquired: id -> (track, removed_at).
        self.lost: Dict[int, Tuple[Track, float]] = {}
        # Candidates not yet confirmed (no id assigned): list of dicts with
        # centroid/contour/segment, consecutive "hits" and accumulated height.
        self.pending: List[dict] = []
        self._next_id = 1

    def update(self, detections: List[dict], freeze_misses: bool = False) -> Tuple[List[int], List[int], List[Optional[int]]]:
        """Update the tracked artifacts with this cycle's detections.

        Args:
            detections: this cycle's detections (full-frame coordinates).
            freeze_misses: True when the scene is occluded (a hand or a robot arm
                over the table): tracks not seen this cycle do not count a miss,
                and pending candidates are left untouched, so nothing expires or
                gets confirmed while the view is unreliable.

        Returns:
            added_ids: ids of artifacts confirmed (or reacquired) this cycle
            removed_ids: ids of artifacts dropped this cycle (missing too long)
            detection_track_ids: one entry per detection, the id it was assigned to,
                or None if it is still pending confirmation or was discarded as a
                duplicate of an already matched track
        """
        now = time.time()

        # Forget lost tracks that are too old to be reacquired. With a window of
        # 0 seconds nothing is ever kept, i.e. reacquisition is disabled.
        for track_id in list(self.lost.keys()):
            _, removed_at = self.lost[track_id]
            if now - removed_at >= self.reacquire_window_s:
                del self.lost[track_id]

        # Every track/detection pair close enough to be the same artifact, closest first.
        pairs = []
        for track_id, track in self.tracks.items():
            for det_index, detection in enumerate(detections):
                distance = _distance(track.centroid, detection["centroid"])
                if distance <= self.match_distance_px:
                    pairs.append((distance, track_id, det_index))
        pairs.sort(key=lambda pair: pair[0])

        # Greedily assign the closest pairs first, each track and each detection used once.
        matched_tracks = set()
        detection_track_ids: List[Optional[int]] = [None] * len(detections)
        for distance, track_id, det_index in pairs:
            if track_id in matched_tracks or detection_track_ids[det_index] is not None:
                continue
            matched_tracks.add(track_id)
            detection_track_ids[det_index] = track_id

        # Refresh matched tracks: only position updates, the pose is left untouched.
        for det_index, track_id in enumerate(detection_track_ids):
            if track_id is None:
                continue
            detection = detections[det_index]
            track = self.tracks[track_id]
            track.centroid = detection["centroid"]
            track.contour = detection["contour"]
            track.segment = detection["segment"]
            track.miss_count = 0

        # Leftover detections: candidates for new artifacts, unless they sit on
        # top of a track matched this same cycle (a split blob of the same fragment).
        leftover = []
        for det_index, detection in enumerate(detections):
            if detection_track_ids[det_index] is not None:
                continue
            is_duplicate = False
            for other_track_id in detection_track_ids:
                if other_track_id is None:
                    continue
                if _distance(self.tracks[other_track_id].centroid, detection["centroid"]) <= self.match_distance_px:
                    is_duplicate = True
                    break
            if not is_duplicate:
                leftover.append(det_index)

        added_ids = []
        if not freeze_misses:
            added_ids = self._update_pending(detections, leftover, detection_track_ids)

        # Tracks with no detection this cycle were not seen: count the miss, and
        # remove only after miss_limit consecutive misses in a row (so a hand or a
        # robot arm passing over an artifact for a moment does not delete it).
        # When the scene is occluded (freeze_misses) nothing is counted at all.
        removed_ids = []
        seen_track_ids = set(track_id for track_id in detection_track_ids if track_id is not None)
        for track_id in list(self.tracks.keys()):
            if track_id in seen_track_ids:
                continue
            if freeze_misses:
                continue
            track = self.tracks[track_id]
            track.miss_count += 1
            if track.miss_count > self.miss_limit:
                # Keep the removed track around for a while: if the fragment is
                # detected again at the same spot it gets this id back.
                self.lost[track_id] = (track, now)
                del self.tracks[track_id]
                removed_ids.append(track_id)

        return added_ids, removed_ids, detection_track_ids

    def _update_pending(self, detections: List[dict], leftover: List[int], detection_track_ids: List[Optional[int]]) -> List[int]:
        """Advance the pending candidates with this cycle's leftover detections.

        Returns the ids of the candidates that got confirmed this cycle (their
        entry in detection_track_ids is filled in too).
        """
        # Match leftover detections to pending candidates, closest first.
        pending_pairs = []
        for p_index, candidate in enumerate(self.pending):
            for det_index in leftover:
                distance = _distance(candidate["centroid"], detections[det_index]["centroid"])
                if distance <= self.match_distance_px:
                    pending_pairs.append((distance, p_index, det_index))
        pending_pairs.sort(key=lambda pair: pair[0])

        pending_used = set()
        det_to_pending = {}
        for distance, p_index, det_index in pending_pairs:
            if p_index in pending_used or det_index in det_to_pending:
                continue
            pending_used.add(p_index)
            det_to_pending[det_index] = p_index

        # Refresh the matched candidates; candidates with no detection this cycle
        # are dropped (a real object is detected every cycle - a one-cycle phantom
        # is not). "this_cycle" remembers which detection touched each candidate.
        next_pending = []
        for det_index, p_index in det_to_pending.items():
            candidate = self.pending[p_index]
            detection = detections[det_index]
            candidate["hits"] += 1
            candidate["centroid"] = detection["centroid"]
            candidate["contour"] = detection["contour"]
            candidate["segment"] = detection["segment"]
            height = detection.get("height_mm")
            if height is not None:
                candidate["height_sum"] += height
                candidate["height_count"] += 1
            candidate["det_index"] = det_index
            next_pending.append(candidate)

        # Brand new candidates - skipping a detection that sits on top of another
        # candidate updated/created this same cycle (a split blob).
        for det_index in leftover:
            if det_index in det_to_pending:
                continue
            detection = detections[det_index]
            too_close = False
            for candidate in next_pending:
                if _distance(candidate["centroid"], detection["centroid"]) <= self.match_distance_px:
                    too_close = True
                    break
            if too_close:
                continue
            candidate = {
                "centroid": detection["centroid"],
                "contour": detection["contour"],
                "segment": detection["segment"],
                "hits": 1,
                "height_sum": 0.0,
                "height_count": 0,
                "det_index": det_index,
            }
            height = detection.get("height_mm")
            if height is not None:
                candidate["height_sum"] += height
                candidate["height_count"] += 1
            next_pending.append(candidate)

        # Confirm the candidates that were seen enough cycles in a row.
        added_ids = []
        self.pending = []
        for candidate in next_pending:
            if candidate["hits"] < self.confirm_hits:
                self.pending.append(candidate)
                continue

            # Flat-phantom check: a "blob" that does not rise above its local
            # surroundings is the mat itself (texture/shadow), not an object.
            if self.min_height_mm > 0 and candidate["height_count"] > 0:
                average_height = candidate["height_sum"] / candidate["height_count"]
                if average_height < self.min_height_mm:
                    continue

            det_index = candidate["det_index"]

            # Reacquisition: a recently lost track close to this candidate gets
            # its old id (and frozen pose) back - the fragment did not move, its
            # detection just flickered off longer than the debounce.
            reacquired_id = None
            best_distance = self.match_distance_px
            for lost_id, (lost_track, _) in self.lost.items():
                distance = _distance(lost_track.centroid, candidate["centroid"])
                if distance <= best_distance:
                    best_distance = distance
                    reacquired_id = lost_id

            if reacquired_id is not None:
                track, _ = self.lost.pop(reacquired_id)
                track.centroid = candidate["centroid"]
                track.contour = candidate["contour"]
                track.segment = candidate["segment"]
                track.miss_count = 0
                self.tracks[reacquired_id] = track
                detection_track_ids[det_index] = reacquired_id
                added_ids.append(reacquired_id)
            else:
                track_id = self._next_id
                self._next_id += 1
                self.tracks[track_id] = Track(track_id, candidate["centroid"], candidate["contour"], candidate["segment"])
                detection_track_ids[det_index] = track_id
                added_ids.append(track_id)

        return added_ids
