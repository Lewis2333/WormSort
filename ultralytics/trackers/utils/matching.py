import numpy as np
import scipy
from scipy.spatial.distance import cdist

from ultralytics.utils.metrics import batch_probiou, bbox_ioa

try:
    import lap

    assert lap.__version__
except (ImportError, AssertionError, AttributeError):
    from ultralytics.utils.checks import check_requirements

    check_requirements("lapx>=0.5.2")
    import lap


def linear_assignment(cost_matrix: np.ndarray, thresh: float, use_lap: bool = True) -> tuple:

    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    if use_lap:


        _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        matches = [[ix, mx] for ix, mx in enumerate(x) if mx >= 0]
        unmatched_a = np.where(x < 0)[0]
        unmatched_b = np.where(y < 0)[0]
    else:


        x, y = scipy.optimize.linear_sum_assignment(cost_matrix)
        matches = np.asarray([[x[i], y[i]] for i in range(len(x)) if cost_matrix[x[i], y[i]] <= thresh])
        if len(matches) == 0:
            unmatched_a = list(np.arange(cost_matrix.shape[0]))
            unmatched_b = list(np.arange(cost_matrix.shape[1]))
        else:
            unmatched_a = list(set(np.arange(cost_matrix.shape[0])) - set(matches[:, 0]))
            unmatched_b = list(set(np.arange(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def iou_distance(atracks: list, btracks: list) -> np.ndarray:

    if atracks and isinstance(atracks[0], np.ndarray) or btracks and isinstance(btracks[0], np.ndarray):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.xywha if track.angle is not None else track.xyxy for track in atracks]
        btlbrs = [track.xywha if track.angle is not None else track.xyxy for track in btracks]

    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if len(atlbrs) and len(btlbrs):
        if len(atlbrs[0]) == 5 and len(btlbrs[0]) == 5:
            ious = batch_probiou(
                np.ascontiguousarray(atlbrs, dtype=np.float32),
                np.ascontiguousarray(btlbrs, dtype=np.float32),
            ).numpy()
        else:
            ious = bbox_ioa(
                np.ascontiguousarray(atlbrs, dtype=np.float32),
                np.ascontiguousarray(btlbrs, dtype=np.float32),
                iou=True,
            )
    return 1 - ious


def embedding_distance(tracks: list, detections: list, metric: str = "cosine") -> np.ndarray:

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float32)


    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float32)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))
    return cost_matrix


def fuse_score(cost_matrix: np.ndarray, detections: list) -> np.ndarray:

    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    return 1 - fuse_sim


def worm_midpoint_distance(tracks: list, detections: list) -> np.ndarray:

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix


    track_points = []
    for t in tracks:
        if hasattr(t, 'midpoint') and t.midpoint is not None:
            track_points.append(t.midpoint)
        else:

            track_points.append(t.mean[:2])

    track_points = np.array(track_points, dtype=np.float32)


    det_points = []
    det_diagonals = []

    for det in detections:
        w, h = det.tlwh[2], det.tlwh[3]
        diagonal = np.sqrt(w ** 2 + h ** 2) + 1e-6
        det_diagonals.append(diagonal)

        if hasattr(det, 'midpoint') and det.midpoint is not None:
            det_points.append(det.midpoint)
        else:
            det_points.append([det.tlwh[0] + w / 2, det.tlwh[1] + h / 2])

    det_points = np.array(det_points, dtype=np.float32)
    det_diagonals = np.array(det_diagonals, dtype=np.float32).reshape(1, -1)


    dists = cdist(track_points, det_points, metric='euclidean')


    normalized_dists = dists / det_diagonals


    normalized_dists[normalized_dists > 1.0] = 1.0

    return normalized_dists


def worm_curvature_distance(tracks: list, detections: list, n_samples: int = 10) -> np.ndarray:

    cost_matrix = np.full((len(tracks), len(detections)), 0.5, dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix

    for i, t in enumerate(tracks):
        t_curv = getattr(t, 'curvature', None)
        if t_curv is None or np.all(t_curv == 0):
            continue

        for j, d in enumerate(detections):
            d_curv = getattr(d, 'curvature', None)
            if d_curv is None or np.all(d_curv == 0):
                continue


            if len(t_curv) != len(d_curv):
                continue


            t_norm = np.linalg.norm(t_curv)
            d_norm = np.linalg.norm(d_curv)
            if t_norm < 1e-8 or d_norm < 1e-8:
                continue

            cos_sim = np.dot(t_curv, d_curv) / (t_norm * d_norm)
            cos_sim = np.clip(cos_sim, -1, 1)


            cost_matrix[i, j] = (1.0 - cos_sim) / 2.0

    return cost_matrix


def worm_length_distance(tracks: list, detections: list) -> np.ndarray:

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix

    for i, t in enumerate(tracks):
        t_len = getattr(t, 'skeleton_length', 0.0)
        if t_len <= 0:
            continue

        for j, d in enumerate(detections):
            d_len = getattr(d, 'skeleton_length', 0.0)
            if d_len <= 0:
                continue


            ratio = min(t_len, d_len) / max(t_len, d_len)
            cost_matrix[i, j] = 1.0 - ratio

    return cost_matrix


def fuse_worm_score(iou_cost: np.ndarray, mid_cost: np.ndarray,
                    iou_weight=0.5, mid_weight=0.5) -> np.ndarray:

    if iou_cost.shape != mid_cost.shape:
        print(f"[WormSort Warning] Shape mismatch: IoU {iou_cost.shape} vs Mid {mid_cost.shape}")
        return iou_cost

    return (iou_weight * iou_cost) + (mid_weight * mid_cost)


def fuse_worm_multi_modal(iou_cost: np.ndarray,
                           mid_cost: np.ndarray,
                           curv_cost: np.ndarray,
                           len_cost: np.ndarray,
                           w_iou: float = 0.4,
                           w_mid: float = 0.3,
                           w_curv: float = 0.15,
                           w_len: float = 0.15) -> np.ndarray:

    target_shape = iou_cost.shape
    if target_shape[0] == 0 or target_shape[1] == 0:
        return iou_cost


    costs = [mid_cost, curv_cost, len_cost]
    weights = [w_mid, w_curv, w_len]
    valid_costs = []
    valid_weights = [w_iou]

    fused = w_iou * iou_cost

    for cost, weight in zip(costs, weights):
        if cost.shape == target_shape:
            fused += weight * cost
            valid_weights.append(weight)
        else:

            pass


    total_w = sum(valid_weights)
    if abs(total_w - 1.0) > 0.01:
        fused /= total_w

    return fused
