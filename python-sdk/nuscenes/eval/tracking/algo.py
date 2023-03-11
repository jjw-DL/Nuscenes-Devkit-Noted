"""
nuScenes dev-kit.
Code written by Holger Caesar, Caglayan Dicle and Oscar Beijbom, 2019.

This code is based on two repositories:

Xinshuo Weng's AB3DMOT code at:
https://github.com/xinshuoweng/AB3DMOT/blob/master/evaluation/evaluate_kitti3dmot.py

py-motmetrics at:
https://github.com/cheind/py-motmetrics
"""
import os
from typing import List, Dict, Callable, Tuple
import unittest

import numpy as np
import sklearn
import tqdm

try:
    import pandas
except ModuleNotFoundError:
    raise unittest.SkipTest('Skipping test as pandas was not found!')

from nuscenes.eval.tracking.constants import MOT_METRIC_MAP, TRACKING_METRICS
from nuscenes.eval.tracking.data_classes import TrackingBox, TrackingMetricData
from nuscenes.eval.tracking.mot import MOTAccumulatorCustom
from nuscenes.eval.tracking.render import TrackingRenderer
from nuscenes.eval.tracking.utils import print_threshold_metrics, create_motmetrics


class TrackingEvaluation(object):
    def __init__(self,
                 tracks_gt: Dict[str, Dict[int, List[TrackingBox]]],
                 tracks_pred: Dict[str, Dict[int, List[TrackingBox]]],
                 class_name: str,
                 dist_fcn: Callable,
                 dist_th_tp: float,
                 min_recall: float,
                 num_thresholds: int,
                 metric_worst: Dict[str, float],
                 verbose: bool = True,
                 output_dir: str = None,
                 render_classes: List[str] = None):
        """
        Create a TrackingEvaluation object which computes all metrics for a given class.
        :param tracks_gt: The ground-truth tracks.
        :param tracks_pred: The predicted tracks.
        :param class_name: The current class we are evaluating on.
        :param dist_fcn: The distance function used for evaluation.
        :param dist_th_tp: The distance threshold used to determine matches.
        :param min_recall: The minimum recall value below which we drop thresholds due to too much noise.
        :param num_thresholds: The number of recall thresholds from 0 to 1. Note that some of these may be dropped.
        :param metric_worst: Mapping from metric name to the fallback value assigned if a recall threshold
            is not achieved.
        :param verbose: Whether to print to stdout.
        :param output_dir: Output directory to save renders.
        :param render_classes: Classes to render to disk or None.

        Computes the metrics defined in:
        - Stiefelhagen 2008: Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics.
          MOTA, MOTP
        - Nevatia 2008: Global Data Association for Multi-Object Tracking Using Network Flows.
          MT/PT/ML
        - Weng 2019: "A Baseline for 3D Multi-Object Tracking".
          AMOTA/AMOTP
        """
        self.tracks_gt = tracks_gt # tracks的gt
        self.tracks_pred = tracks_pred # tracks的pred，按照scene组织的dict
        self.class_name = class_name # 类别名称
        self.dist_fcn = dist_fcn # 距离函数
        self.dist_th_tp = dist_th_tp # 确定匹配中的距离阈值
        self.min_recall = min_recall # 最小的recall值
        self.num_thresholds = num_thresholds # recall阈值数量
        self.metric_worst = metric_worst # 如果没达到recall阈值，分配的最差的评估值
        self.verbose = verbose
        self.output_dir = output_dir
        self.render_classes = [] if render_classes is None else render_classes

        self.n_scenes = len(self.tracks_gt) # 场景数量

        # Specify threshold naming pattern. Note that no two thresholds may have the same name.
        def name_gen(_threshold):
            return 'thr_%.4f' % _threshold
        self.name_gen = name_gen

        # Check that metric definitions are consistent.
        for metric_name in MOT_METRIC_MAP.values():
            assert metric_name == '' or metric_name in TRACKING_METRICS

    def accumulate(self) -> TrackingMetricData:
        """
        Compute metrics for all recall thresholds of the current class.
        :return: TrackingMetricData instance which holds the metrics for each threshold.
        """
        # 1.Init.
        if self.verbose:
            print('Computing metrics for class %s...\n' % self.class_name)
        accumulators = []
        thresh_metrics = []
        md = TrackingMetricData() # MetricData简记md

        # 2.Skip missing classes.
        gt_box_count = 0
        gt_track_ids = set()
        # 逐seq处理, 一个seq20s，每0.5s采样一帧，一共40帧
        for scene_tracks_gt in self.tracks_gt.values():
            # 逐帧处理
            for frame_gt in scene_tracks_gt.values():
                # 帧内逐个bbox处理
                for box in frame_gt:
                    # 如果是该bbox是要计算的类
                    if box.tracking_name == self.class_name:
                        gt_box_count += 1
                        gt_track_ids.add(box.tracking_id) # 记录该seq的track id
        if gt_box_count == 0:
            # Do not add any metric. The average metrics will then be nan.
            return md

        # 3.Register mot metrics.
        mh = create_motmetrics() # 创建MetricsHost对象，并且注册自定义metrics

        # 4.Get thresholds.
        # Note: The recall values are the hypothetical recall (10%, 20%, ..).
        # The actual recall may vary as there is no way to compute it without trying all thresholds.
        thresholds, recalls = self.compute_thresholds(gt_box_count) # 计算阈值和预定义阈值
        md.confidence = thresholds
        md.recall_hypo = recalls
        if self.verbose:
            print('Computed thresholds\n')

        # 5.逐个阈值计算
        for t, threshold in enumerate(thresholds):
            # If recall threshold is not achieved, we assign the worst possible value in AMOTA and AMOTP.
            if np.isnan(threshold):
                continue # 跳过无效阈值

            # Do not compute the same threshold twice.
            # This becomes relevant when a user submits many boxes with the exact same score.
            if threshold in thresholds[:t]:
                continue

            # -------------------------
            # Accumulate track data.
            # -------------------------
            acc, _ = self.accumulate_threshold(threshold) # 针对该阈值的合并accumulator
            accumulators.append(acc) # 添加进accumulators

            # Compute metrics for current threshold.
            # 针对当前阈值计算metrics
            thresh_name = self.name_gen(threshold) # eg：‘thr_0.1589’
            # -------------------------
            # 针对该阈值计算各指标
            # -------------------------
            thresh_summary = mh.compute(acc, metrics=MOT_METRIC_MAP.keys(), name=thresh_name)
            thresh_metrics.append(thresh_summary) # 将计算指标加入thresh_summary

            # Print metrics to stdout.
            if self.verbose:
                print_threshold_metrics(thresh_summary.to_dict()) # 打印阈值

        # Concatenate all metrics. We only do this for more convenient access.
        # 连接所有指标，我们这样做只是为了更方便的访问
        if len(thresh_metrics) == 0:
            summary = []
        else:
            summary = pandas.concat(thresh_metrics)

        # Get the number of thresholds which were not achieved (i.e. nan).
        unachieved_thresholds = np.array([t for t in thresholds if np.isnan(t)]) # nan的threshold
        num_unachieved_thresholds = len(unachieved_thresholds) # eg:8

        # Get the number of thresholds which were achieved (i.e. not nan).
        # 获取达到的阈值数量（即不是 nan）
        valid_thresholds = [t for t in thresholds if not np.isnan(t)]
        assert valid_thresholds == sorted(valid_thresholds)
        # 计算重复阈值的数量
        num_duplicate_thresholds = len(valid_thresholds) - len(np.unique(valid_thresholds)) # eg:0

        # Sanity check. 完整性检查
        assert num_unachieved_thresholds + num_duplicate_thresholds + len(thresh_metrics) == self.num_thresholds

        # Figure out how many times each threshold should be repeated. 大概率都是1
        rep_counts = [np.sum(thresholds == t) for t in np.unique(valid_thresholds)]

        # Store all traditional metrics. 存储所有传统指标
        for (mot_name, metric_name) in MOT_METRIC_MAP.items():
            # Skip metrics which we don't output. 跳过我们不输出的指标
            if metric_name == '':
                continue

            # Retrieve and store values for current metric. 检索和存储当前指标的值
            if len(thresh_metrics) == 0:
                # Set all the worst possible value if no recall threshold is achieved.
                worst = self.metric_worst[metric_name]
                if worst == -1:
                    if metric_name == 'ml':
                        worst = len(gt_track_ids)
                    elif metric_name in ['gt', 'fn']:
                        worst = gt_box_count
                    elif metric_name in ['fp', 'ids', 'frag']:
                        worst = np.nan  # We can't know how these error types are distributed.
                    else:
                        raise NotImplementedError

                all_values = [worst] * TrackingMetricData.nelem
            else:
                values = summary.get(mot_name).values # 将自定义的指标与传统指标对应
                assert np.all(values[np.logical_not(np.isnan(values))] >= 0)

                # If a threshold occurred more than once, duplicate the metric values.
                assert len(rep_counts) == len(values)
                values = np.concatenate([([v] * r) for (v, r) in zip(values, rep_counts)])

                # Pad values with nans for unachieved recall thresholds.
                all_values = [np.nan] * num_unachieved_thresholds
                all_values.extend(values)

            assert len(all_values) == TrackingMetricData.nelem
            md.set_metric(metric_name, all_values)

        return md

    def accumulate_threshold(self, threshold: float = None) -> Tuple[pandas.DataFrame, List[float]]:
        """
        Accumulate metrics for a particular recall threshold of the current class.
        The scores are only computed if threshold is set to None. This is used to infer the recall thresholds.
        :param threshold: score threshold used to determine positives and negatives.
        :return: (The MOTAccumulator that stores all the hits/misses/etc, Scores for each TP).
        """
        accs = []
        # TP的分数， 用于最初确定召回阈值
        scores = []  # The scores of the TPs. These are used to determine the recall thresholds initially.

        # Go through all frames and associate ground truth and tracker results.
        # Groundtruth and tracker contain lists for every single frame containing lists detections.
        # 逐个seq(scene)处理
        for scene_id in tqdm.tqdm(self.tracks_gt.keys(), disable=not self.verbose, leave=False):

            # 1.Initialize accumulator and frame_id for this scene 针对当前的scene初始化accumulator和帧id
            acc = MOTAccumulatorCustom()
            frame_id = 0  # Frame ids must be unique across all scenes

            # 2.Retrieve GT and preds.
            scene_tracks_gt = self.tracks_gt[scene_id]
            scene_tracks_pred = self.tracks_pred[scene_id]

            # Visualize the boxes in this frame.
            if self.class_name in self.render_classes and threshold is None:
                save_path = os.path.join(self.output_dir, 'render', str(scene_id), self.class_name)
                os.makedirs(save_path, exist_ok=True)
                renderer = TrackingRenderer(save_path)
            else:
                renderer = None

            # 3.在seq内逐个时间戳处理
            for timestamp in scene_tracks_gt.keys():
                # 3.1 Select only the current class. 仅针对当前类别处理
                frame_gt = scene_tracks_gt[timestamp] # 获取该帧的gt
                frame_pred = scene_tracks_pred[timestamp] # 获取该帧的pred
                frame_gt = [f for f in frame_gt if f.tracking_name == self.class_name] # 提取该帧内是该类的gt
                frame_pred = [f for f in frame_pred if f.tracking_name == self.class_name] # 提取该帧内是该类的pred

                # 3.2 Threshold boxes by score. Note that the scores were previously averaged over the whole track.
                # 通过分数过滤box，分数是之前整个轨迹的平均
                if threshold is not None:
                    # 如果阈值不空，针对该帧逐个box判断分数是否大于阈值
                    frame_pred = [f for f in frame_pred if f.tracking_score >= threshold]

                # 3.3 Abort if there are neither GT nor pred boxes.
                gt_ids = [gg.tracking_id for gg in frame_gt] # 记录该帧该类gt的id
                pred_ids = [tt.tracking_id for tt in frame_pred] # 记录该帧该类pred的id
                if len(gt_ids) == 0 and len(pred_ids) == 0:
                    continue

                # 3.4 Calculate distances.
                # Note that the distance function is hard-coded to achieve significant speedups via vectorization.
                assert self.dist_fcn.__name__ == 'center_distance'
                if len(frame_gt) == 0 or len(frame_pred) == 0:
                    distances = np.ones((0, 0))
                else:
                    gt_boxes = np.array([b.translation[:2] for b in frame_gt]) # 计算该帧gt box的中心点
                    pred_boxes = np.array([b.translation[:2] for b in frame_pred]) # 计算该帧pred box的中心点
                    distances = sklearn.metrics.pairwise.euclidean_distances(gt_boxes, pred_boxes) # 计算gt和pred的bbox的中心距离

                # 3.5 Distances that are larger than the threshold won't be associated.
                # 大于阈值的距离将不会关联
                assert len(distances) == 0 or not np.all(np.isnan(distances))
                distances[distances >= self.dist_th_tp] = np.nan

                # ------------------------------
                # 3.6 Accumulate results.
                # Note that we cannot use timestamp as frameid as motmetrics assumes it's an integer.
                # ------------------------------
                acc.update(gt_ids, pred_ids, distances, frameid=frame_id) # 调用Accumulator的update函数

                # 3.7 Store scores of matches, which are used to determine recall thresholds.
                if threshold is None:
                    events = acc.events.loc[frame_id] # 获取该帧对应的event
                    matches = events[events.Type == 'MATCH'] # 获取类型为MATCH的事件
                    match_ids = matches.HId.values # 获取匹配中hypothesis的id
                    # 遍历预测帧中所有框，并记录匹配id的跟踪分数
                    match_scores = [tt.tracking_score for tt in frame_pred if tt.tracking_id in match_ids]
                    scores.extend(match_scores) # 记录匹配分数
                else:
                    events = None

                # 3.8 Render the boxes in this frame.
                if self.class_name in self.render_classes and threshold is None:
                    renderer.render(events, timestamp, frame_gt, frame_pred)

                # 3.9 Increment the frame_id, unless there are no boxes (equivalent to what motmetrics does).
                frame_id += 1

            accs.append(acc) # 记录累加器150个，对应150段数据

        # 4. Merge accumulators
        acc_merged = MOTAccumulatorCustom.merge_event_dataframes(accs)

        return acc_merged, scores

    def compute_thresholds(self, gt_box_count: int) -> Tuple[List[float], List[float]]:
        """
        Compute the score thresholds for predefined recall values.
        AMOTA/AMOTP average over all thresholds, whereas MOTA/MOTP/.. pick the threshold with the highest MOTA.
        :param gt_box_count: The number of GT boxes for this class.
        :return: The lists of thresholds and their recall values.
        """
        # 1.Run accumulate to get the scores of TPs.
        _, scores = self.accumulate_threshold(threshold=None) # 返回值是累积后的accumulate(共6019帧)和scores(eg:1670)

        # 2.Abort if no predictions exist.
        if len(scores) == 0:
            return [np.nan] * self.num_thresholds, [np.nan] * self.num_thresholds

        # 3.Sort scores.
        scores = np.array(scores) 
        scores.sort() # 对分数排序
        scores = scores[::-1] # 逆序,从大到小排序

        # 4.Compute recall levels.
        tps = np.array(range(1, len(scores) + 1)) # [1, 2, ..., 1670]
        rec = tps / gt_box_count # gt_box_count:1993 --> (1670,)
        assert len(scores) / gt_box_count <= 1

        # 5.Determine thresholds.
        max_recall_achieved = np.max(rec) # eg:0.8379
        # self.min_recall:0.1, self.num_thresholds:40
        rec_interp = np.linspace(self.min_recall, 1, self.num_thresholds).round(12)
        # rec, scores相当于x和y坐标，rec_interp是插值点，操作插值点最大值的分数设置为0
        thresholds = np.interp(rec_interp, rec, scores, right=0)

        # 6.Set thresholds for unachieved recall values to nan to penalize AMOTA/AMOTP later.
        # 超过最大阈值分数的值设置为nan
        thresholds[rec_interp > max_recall_achieved] = np.nan

        # 7.Cast to list. 转换为list
        thresholds = list(thresholds.tolist())
        rec_interp = list(rec_interp.tolist())

        # 8.Reverse order for more convenient presentation.
        thresholds.reverse() # （40，）从小到大排序
        rec_interp.reverse() # 原始recall阈值

        # 9.Check that we return the correct number of thresholds.
        assert len(thresholds) == len(rec_interp) == self.num_thresholds

        return thresholds, rec_interp
