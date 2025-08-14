import numpy as np
from scipy.spatial.distance import cdist

class TrackedObject:
    """Represents a single object being tracked across frames."""

    def __init__(self, obj_id, prediction, smoothing_factor):
        self.id = obj_id
        self.liveness_counter = 1
        self.smoothing_factor = smoothing_factor
        self.bbox = np.array(prediction['bbox'], dtype=np.float32)
        self.class_id = prediction['class']
        self.conf = prediction['conf']
        self.avg_mask = prediction.get('mask')
        self.middle = prediction['middle']

    def update(self, prediction):
        """Updates the object's properties with a new prediction."""
        self.bbox = prediction['bbox']

        if 'mask' in prediction and prediction['mask'] is not None:
            new_mask = prediction['mask'].astype(np.float32)

            # This line smoothly blends the old mask with the new one.
            # The 'smoothing_factor' from the UI controls the blend ratio.
            self.avg_mask = (self.smoothing_factor * new_mask) + ((1 - self.smoothing_factor) * self.avg_mask)

        self.conf = prediction['conf']
        self.middle = prediction['middle']


class Tracker:
    """Manages all tracked objects and their state."""

    def __init__(self, distance_threshold,max_liveliness, smoothing_factor, liveness_increment=1, liveness_decrement=1,
                 deletion_threshold=-5):
        self.tracked_objects = {}
        self.next_id = 0
        self.distance_threshold = distance_threshold
        self.liveness_increment = liveness_increment
        self.liveness_decrement = liveness_decrement
        self.deletion_threshold = deletion_threshold
        self.smoothing_factor = smoothing_factor
        self.max_liveliness = max_liveliness

    def update(self, new_predictions):
        """
        Updates the tracker with a new list of predictions from the current frame.
        Returns a list of 'good' objects to be displayed.
        """

        if not new_predictions:
            for track_id in self.tracked_objects:
                self.tracked_objects[track_id].liveness_counter -= self.liveness_decrement
            self._prune_objects()
            return self.get_good_objects()

        if not self.tracked_objects:
            for pred in new_predictions:
                self._register_new_object(pred)
            return self.get_good_objects()

        tracked_ids = list(self.tracked_objects.keys())
        tracked_middles = np.array([self.tracked_objects[tid].middle for tid in tracked_ids])
        new_middles = np.array([p['middle'] for p in new_predictions])

        if tracked_middles.size == 0 or new_middles.size == 0:
            # Handle cases where there are no existing tracks or new predictions to avoid cdist error
            for i, pred in enumerate(new_predictions):
                self._register_new_object(pred)
            self._prune_objects()
            return self.get_good_objects()

        distance_matrix = cdist(tracked_middles, new_middles)
        potential_matches_indices = np.where(distance_matrix <= self.distance_threshold)

        matched_track_ids = set()
        matched_new_pred_indices = set()

        for track_idx, pred_idx in zip(*potential_matches_indices):
            track_id = tracked_ids[track_idx]

            if track_id in matched_track_ids or pred_idx in matched_new_pred_indices:
                continue

            if self.tracked_objects[track_id].class_id == new_predictions[pred_idx]['class']:
                if self.tracked_objects[track_id].liveness_counter <= self.max_liveliness:
                    self.tracked_objects[track_id].liveness_counter += self.liveness_increment
                self.tracked_objects[track_id].update(new_predictions[pred_idx])

                matched_track_ids.add(track_id)
                matched_new_pred_indices.add(pred_idx)

        for track_id in self.tracked_objects:
            if track_id not in matched_track_ids:
                self.tracked_objects[track_id].liveness_counter -= self.liveness_decrement

        for i, pred in enumerate(new_predictions):
            if i not in matched_new_pred_indices:
                self._register_new_object(pred)

        self._prune_objects()
        return self.get_good_objects()

    def _register_new_object(self, prediction):
        """Creates a new TrackedObject and adds it to the tracker."""
        new_object = TrackedObject(self.next_id, prediction, smoothing_factor=self.smoothing_factor)
        self.tracked_objects[self.next_id] = new_object
        self.next_id += 1

    def _prune_objects(self):
        """Removes objects whose liveness counter has fallen below the deletion threshold."""
        dead_object_ids = [
            tid for tid, obj in self.tracked_objects.items()
            if obj.liveness_counter <= self.deletion_threshold
        ]
        for tid in dead_object_ids:
            del self.tracked_objects[tid]

    def get_good_objects(self):
        """Returns a list of objects that should be displayed (counter > 0)."""
        good_objects = [
            obj for obj in self.tracked_objects.values()
            if obj.liveness_counter > 0
        ]
        return good_objects

    def update_params(self, distance_threshold=None, smoothing_factor=None,max_liveliness=None):
        if distance_threshold is not None:
            self.distance_threshold = distance_threshold
        if smoothing_factor is not None:
            self.smoothing_factor = smoothing_factor
        if max_liveliness is not None:
            self.max_liveliness = max_liveliness
