import numpy as np
from scipy.spatial.distance import cdist

class TrackedObject:
    """Represents a single object being tracked across frames."""
    def __init__(self, obj_id, prediction,smoothing_factor):
        self.id = obj_id
        self.liveness_counter = 1  # Start with a positive counter
        self.smoothing_factor = smoothing_factor
        # --- Store all properties from the prediction ---
        self.bbox = np.array(prediction['bbox'], dtype=np.float32)
        self.class_id = prediction['class']
        self.conf = prediction['conf']
        self.avg_mask = prediction.get('mask') # Use .get() for optional keys
        self.middle = prediction['middle']

    def update(self, prediction):
        """Updates the object's properties with a new prediction."""
        self.bbox = prediction['bbox']

        # --- Smooth the mask using EMA (if it exists) ---
        pred_mask = prediction['mask']
        b = np.where(pred_mask == 1, pred_mask, -1)
        self.avg_mask = self.avg_mask+b
        self.avg_mask = np.where(self.avg_mask >-5, self.avg_mask,-5)


        # Update confidence to the latest one
        self.conf = prediction['conf']
        self.middle = prediction['middle']

class Tracker:
    """Manages all tracked objects and their state."""
    def __init__(self, distance_threshold,smoothing_factor, liveness_increment=1, liveness_decrement=1, deletion_threshold=-5):
        self.tracked_objects = {}  # A dictionary to store TrackedObject instances {id: object}
        self.next_id = 0
        self.distance_threshold = distance_threshold
        self.liveness_increment = liveness_increment
        self.liveness_decrement = liveness_decrement
        self.deletion_threshold = deletion_threshold
        self.smoothing_factor = smoothing_factor

    def update(self, new_predictions):
        """
        Updates the tracker with a new list of predictions from the current frame.
        Returns a list of 'good' objects to be displayed.
        """

        if not new_predictions:
            # If there are no new detections, decrement the counter for all existing tracks
            for track_id in self.tracked_objects:
                self.tracked_objects[track_id].liveness_counter -= self.liveness_decrement

            # Prune any objects that have been lost for too long
            self._prune_objects()
            # Return the current list of good objects
            return self.get_good_objects()
        # --- Case 1: No objects are being tracked yet ---
        if not self.tracked_objects:
            for pred in new_predictions:
                self._register_new_object(pred)
            return self.get_good_objects()

        # --- Step 1: Prepare for matching ---
        tracked_ids = list(self.tracked_objects.keys())
        tracked_middles = np.array([self.tracked_objects[tid].middle for tid in tracked_ids])
        new_middles = np.array([p['middle'] for p in new_predictions])

        # --- Step 2: Perform distance matching ---
        distance_matrix = cdist(tracked_middles, new_middles)
        potential_matches_indices = np.where(distance_matrix <= self.distance_threshold)



        # --- Step 3: Process potential matches to find confirmed matches ---
        matched_track_ids = set()
        matched_new_pred_indices = set()

        for track_idx, pred_idx in zip(*potential_matches_indices):
            track_id = tracked_ids[track_idx]

            # Ensure neither the track nor the prediction is already matched
            if track_id in matched_track_ids or pred_idx in matched_new_pred_indices:
                continue

            # Check if classes match
            if self.tracked_objects[track_id].class_id == new_predictions[pred_idx]['class']:
                # --- Successful Match Found ---
                # Increment counter up to 25
                if self.tracked_objects[track_id].liveness_counter <= 25:
                    self.tracked_objects[track_id].liveness_counter += self.liveness_increment
                # Update object properties (bbox, conf, etc.)
                self.tracked_objects[track_id].update(new_predictions[pred_idx])

                # Register the match
                matched_track_ids.add(track_id)
                matched_new_pred_indices.add(pred_idx)


        # --- Step 4: Handle unmatched objects ---

        # Decrement counter for tracked objects that were NOT found in this frame
        for track_id in self.tracked_objects:
            if track_id not in matched_track_ids:
                self.tracked_objects[track_id].liveness_counter -= self.liveness_decrement

        # Register new objects for predictions that were NOT matched to any existing track
        for i, pred in enumerate(new_predictions):
            if i not in matched_new_pred_indices:
                self._register_new_object(pred)

        # --- Step 5: Prune dead objects and return the 'good' set ---
        self._prune_objects()
        return self.get_good_objects()



    def _register_new_object(self, prediction):
        """Creates a new TrackedObject and adds it to the tracker."""
        new_object = TrackedObject(self.next_id, prediction,smoothing_factor=self.smoothing_factor)
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