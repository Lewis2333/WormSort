import cv2
import numpy as np
from collections import deque
from skimage import morphology
from scipy import ndimage
import networkx as nx
from networkx.algorithms.shortest_paths.generic import has_path
from itertools import combinations

from .bot_sort import BOTSORT
from .basetrack import TrackState
from .byte_tracker import STrack
from .utils import matching


class SkeletonExtractor:


    @staticmethod
    def extract_from_mask(mask, min_points=5):
        if mask is None:
            return []
        binary = (mask > 0.5).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = ndimage.binary_fill_holes(binary).astype(np.uint8)
        try:
            skeleton = morphology.skeletonizes(binary > 0).astype(np.uint8)
        except Exception:
            return []
        points = np.column_stack(np.where(skeleton > 0))
        if len(points) < min_points:
            return []
        return SkeletonExtractor._order_points(skeleton, points)

    @staticmethod
    def _order_points(skeleton_img, points):
        if len(points) == 0:
            return []
        G = nx.Graph()
        point_tuples = [(int(p[0]), int(p[1])) for p in points]
        G.add_nodes_from(point_tuples)
        for y, x in point_tuples:
            for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                           (0, 1), (1, -1), (1, 0), (1, 1)]:
                n_y, n_x = y + dy, x + dx
                if 0 <= n_y < skeleton_img.shape[0] and 0 <= n_x < skeleton_img.shape[1]:
                    if skeleton_img[n_y, n_x] > 0:
                        G.add_edge((y, x), (n_y, n_x))
        if len(G.nodes) == 0:
            return []
        endpoints = [node for node in G if G.degree(node) == 1]
        try:
            if len(endpoints) >= 2:
                max_len, max_path = 0, []
                search_nodes = endpoints[:6] if len(endpoints) > 6 else endpoints
                for s, t in combinations(search_nodes, 2):
                    if has_path(G, s, t):
                        path = nx.shortest_path(G, s, t)
                        if len(path) > max_len:
                            max_len, max_path = len(path), path
                path = max_path
            else:
                if nx.is_connected(G):
                    nodes = list(G.nodes)
                    path = nx.shortest_path(G, nodes[0], nodes[-1])
                else:
                    path = list(G.nodes)
        except Exception:
            path = list(G.nodes)
        return [(x, y) for y, x in path] if path else []

    @staticmethod
    def get_head_tail(centerline):
        if not centerline or len(centerline) < 2:
            return None, None
        head = np.array([centerline[0][0], centerline[0][1]], dtype=np.float32)
        tail = np.array([centerline[-1][0], centerline[-1][1]], dtype=np.float32)
        return head, tail

    @staticmethod
    def compute_length(centerline):
        if not centerline or len(centerline) < 2:
            return 0.0
        pts = np.array(centerline, dtype=np.float32)
        diffs = np.diff(pts, axis=0)
        return float(np.sum(np.sqrt(np.sum(diffs ** 2, axis=1))))

    @staticmethod
    def compute_curvature(centerline, n_samples=10):
        if not centerline or len(centerline) < 5:
            return np.zeros(n_samples, dtype=np.float32)
        pts = np.array(centerline, dtype=np.float32)
        n_pts = len(pts)
        indices = np.linspace(1, n_pts - 2, n_samples).astype(int)
        step = max(1, n_pts // 20)
        curvatures = []
        for idx in indices:
            idx_prev = max(0, idx - step)
            idx_next = min(n_pts - 1, idx + step)
            if idx_prev == idx or idx_next == idx:
                curvatures.append(0.0)
                continue
            v1 = pts[idx] - pts[idx_prev]
            v2 = pts[idx_next] - pts[idx]
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            curvatures.append(np.arctan2(cross, dot))
        return np.array(curvatures, dtype=np.float32)


class WORMSORT(BOTSORT):


    def __init__(self, args, frame_rate=30):
        super().__init__(args, frame_rate)
        self.is_wormsort = True


        self.w_iou = float(getattr(args, 'w_iou', 0.6))
        self.w_mid = float(getattr(args, 'w_mid', 0.1))
        self.w_curv = float(getattr(args, 'w_curv', 0.2))
        self.w_len = float(getattr(args, 'w_len', 0.1))


        self.alpha_update = float(getattr(args, 'alpha_update', 0.7))
        self.alpha_reactivate = float(getattr(args, 'alpha_reactivate', 0.9))


        ukf_q_pos = float(getattr(args, 'ukf_q_pos', 4.0))
        ukf_q_vel = float(getattr(args, 'ukf_q_vel', 2.0))
        ukf_q_theta = float(getattr(args, 'ukf_q_theta', 0.3))
        ukf_q_omega = float(getattr(args, 'ukf_q_omega', 0.1))
        ukf_r_pos = float(getattr(args, 'ukf_r_pos', 2.0))

        from .utils.kalman_filter import WormMidpointUKF
        STrack.shared_ukf = WormMidpointUKF(dt=1.0)
        STrack.shared_ukf.Q = np.diag([ukf_q_pos, ukf_q_pos, ukf_q_vel,
                                        ukf_q_theta, ukf_q_omega])
        STrack.shared_ukf.R = np.diag([ukf_r_pos, ukf_r_pos])


        self.curvature_samples = int(getattr(args, 'curvature_samples', 10))
        self.min_skeleton_points = int(getattr(args, 'min_skeleton_points', 5))


        STrack._wormsort_alpha_update = self.alpha_update
        STrack._wormsort_alpha_reactivate = self.alpha_reactivate


        total = self.w_iou + self.w_mid + self.w_curv + self.w_len
        if abs(total - 1.0) > 0.01:
            print(f"⚠️  WormSort weights sum={total:.2f}, normalizing...")
            self.w_iou /= total
            self.w_mid /= total
            self.w_curv /= total
            self.w_len /= total

        print(f"✅ WormSort V4 | w_iou={self.w_iou:.2f} w_mid={self.w_mid:.2f} "
              f"w_curv={self.w_curv:.2f} w_len={self.w_len:.2f} | "
              f"alpha={self.alpha_update}/{self.alpha_reactivate} | "
              f"UKF Q=[{ukf_q_pos},{ukf_q_vel},{ukf_q_theta},{ukf_q_omega}] R={ukf_r_pos}")


    def get_worm_dists(self, tracks, detections):

        iou_cost = matching.iou_distance(tracks, detections)
        mid_cost = matching.worm_midpoint_distance(tracks, detections)
        curv_cost = matching.worm_curvature_distance(tracks, detections,
                                                      n_samples=self.curvature_samples)
        len_cost = matching.worm_length_distance(tracks, detections)
        return matching.fuse_worm_multi_modal(
            iou_cost, mid_cost, curv_cost, len_cost,
            w_iou=self.w_iou, w_mid=self.w_mid,
            w_curv=self.w_curv, w_len=self.w_len
        )


    def update(self, results, img=None):

        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []


        det = resultss.boxes.data.cpu().numpy()
        scores = det[:, 4]
        bboxes = det[:, :4]
        classes = det[:, 5]


        masks = results.masks
        current_heads = [None] * len(bboxes)
        current_tails = [None] * len(bboxes)
        current_lengths = [0.0] * len(bboxes)
        current_curvatures = [None] * len(bboxes)

        if masks is not None:
            mask_data = masks.data
            orig_h, orig_w = img.shape[:2]
            for i, mask_tensor in enumerate(mask_data):
                if i >= len(bboxes):
                    break
                mask_np = mask_tensor.cpu().numpy()
                if mask_np.shape[0] != orig_h or mask_np.shape[1] != orig_w:
                    mask_np = cv2.resize(mask_np, (orig_w, orig_h))
                centerline = SkeletonExtractor.extract_from_mask(
                    mask_np, min_points=self.min_skeleton_points
                )
                if centerline and len(centerline) >= self.min_skeleton_points:
                    h_pt, t_pt = SkeletonExtractor.get_head_tail(centerline)
                    current_heads[i] = h_pt
                    current_tails[i] = t_pt
                    current_lengths[i] = SkeletonExtractor.compute_length(centerline)
                    current_curvatures[i] = SkeletonExtractor.compute_curvature(
                        centerline, n_samples=self.curvature_samples
                    )


        detections = []
        for i, tlbr in enumerate(bboxes):
            w = tlbr[2] - tlbr[0]
            h = tlbr[3] - tlbr[1]
            cx = tlbr[0] + w / 2
            cy = tlbr[1] + h / 2
            xywh_with_idx = np.array([cx, cy, w, h, i], dtype=np.float32)
            track = STrack(
                xywh_with_idx, scores[i], cls=int(classes[i]),
                head=current_heads[i], tail=current_tails[i],
                skeleton_length=current_lengths[i],
                curvature=current_curvatures[i]
            )
            detections.append(track)


        remain_inds = scores >= self.args.track_high_thresh
        inds_low = scores > self.args.track_low_thresh
        inds_high = scores < self.args.track_high_thresh
        inds_second = inds_low & inds_high
        dets_second = [detections[i] for i in range(len(detections)) if inds_second[i]]
        dets = [detections[i] for i in range(len(detections)) if remain_inds[i]]

        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)


        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        self.multi_predict(strack_pool)
        STrack.multi_predict_ukf(strack_pool)

        dists = self.get_worm_dists(strack_pool, dets)
        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=self.args.match_thresh
        )

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det_item = dets[idet]
            if track.state == TrackState.Tracked:
                track.update(det_item, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det_item, self.frame_id, new_id=False)
                refind_stracks.append(track)


        r_tracked_stracks = [strack_pool[i] for i in u_track
                             if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, dets_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det_item = dets_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det_item, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det_item, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)


        dets_unconfirmed = [dets[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, dets_unconfirmed)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            unconfirmed[itracked].update(dets_unconfirmed[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)


        for inew in u_detection:
            track = dets_unconfirmed[inew]
            if track.score < self.args.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)


        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )

        return np.asarray(
            [x.result for x in self.tracked_stracks if x.is_activated],
            dtype=np.float32
        )
