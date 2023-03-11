# nuScenes dev-kit.
# Code written by Holger Caesar, Caglayan Dicle and Oscar Beijbom, 2019.

import argparse
import json
import os
import time
from typing import Tuple, List, Dict, Any

import numpy as np

from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.tracking.algo import TrackingEvaluation
from nuscenes.eval.tracking.constants import AVG_METRIC_MAP, MOT_METRIC_MAP, LEGACY_METRICS
from nuscenes.eval.tracking.data_classes import TrackingMetrics, TrackingMetricDataList, TrackingConfig, TrackingBox, \
    TrackingMetricData
from nuscenes.eval.tracking.loaders import create_tracks
from nuscenes.eval.tracking.render import recall_metric_curve, summary_plot
from nuscenes.eval.tracking.utils import print_final_metrics


class TrackingEval:
    """
    This is the official nuScenes tracking evaluation code.
    Results are written to the provided output_dir.

    Here is an overview of the functions in this method:
    - init: Loads GT annotations and predictions stored in JSON format and filters the boxes.
    - run: Performs evaluation and dumps the metric data to disk.
    - render: Renders various plots and dumps to disk.

    We assume that:
    - Every sample_token is given in the results, although there may be not predictions for that sample.

    Please see https://www.nuscenes.org/tracking for more details.
    """
    def __init__(self,
                 config: TrackingConfig,
                 result_path: str,
                 eval_set: str,
                 output_dir: str,
                 nusc_version: str,
                 nusc_dataroot: str,
                 verbose: bool = True,
                 render_classes: List[str] = None):
        """
        Initialize a TrackingEval object.
        :param config: A TrackingConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param nusc_version: The version of the NuScenes dataset.
        :param nusc_dataroot: Path of the nuScenes dataset on disk.
        :param verbose: Whether to print to stdout.
        :param render_classes: Classes to render to disk or None.
        """
        self.cfg = config # TrackingConfig对象
        self.result_path = result_path # 结果文件路径
        self.eval_set = eval_set # 'val'
        self.output_dir = output_dir # 输出文件夹
        self.verbose = verbose # True
        self.render_classes = render_classes # ’None'

        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots') # 拼接plots文件夹
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir) # 创建输出文件夹
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir) # 创建plots文件夹

        # Initialize NuScenes object. 初始化Nuscenes对象
        # We do not store it in self to let garbage collection take care of it and save memory.
        nusc = NuScenes(version=nusc_version, verbose=verbose, dataroot=nusc_dataroot)

        # Load data.
        if verbose:
            print('Initializing nuScenes tracking evaluation')
        # EvalBox是TrackingBox
        pred_boxes, self.meta = load_prediction(self.result_path, self.cfg.max_boxes_per_sample, TrackingBox,
                                                verbose=verbose) # 根据json文件加载并构建pred
        gt_boxes = load_gt(nusc, self.eval_set, TrackingBox, verbose=verbose) # 加载gt, 前后两帧根据annotations的instance_token关联

        assert set(pred_boxes.sample_tokens) == set(gt_boxes.sample_tokens), \
            "Samples in split don't match samples in predicted tracks."

        # Add center distances. 为box添加与自车的距离字段
        pred_boxes = add_center_dist(nusc, pred_boxes)
        gt_boxes = add_center_dist(nusc, gt_boxes)

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print('Filtering tracks')
        pred_boxes = filter_eval_boxes(nusc, pred_boxes, self.cfg.class_range, verbose=verbose)
        if verbose:
            print('Filtering ground truth tracks')
        gt_boxes = filter_eval_boxes(nusc, gt_boxes, self.cfg.class_range, verbose=verbose)

        self.sample_tokens = gt_boxes.sample_tokens # (6019,)

        # Convert boxes to tracks format. 将bbox转换为跟踪格式
        # 一个场景中的track按照时间顺序排序, 记录所有场景
        self.tracks_gt = create_tracks(gt_boxes, nusc, self.eval_set, gt=True)
        self.tracks_pred = create_tracks(pred_boxes, nusc, self.eval_set, gt=False)

    def evaluate(self) -> Tuple[TrackingMetrics, TrackingMetricDataList]:
        """
        Performs the actual evaluation.
        :return: A tuple of high-level and the raw metric data.
        """
        start_time = time.time()
        metrics = TrackingMetrics(self.cfg) # 初始化TrackingMetrics，记录最大mota对应的metrics

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        if self.verbose:
            print('Accumulating metric data...')
        metric_data_list = TrackingMetricDataList()

        def accumulate_class(curr_class_name):
            curr_ev = TrackingEvaluation(self.tracks_gt, self.tracks_pred, curr_class_name, self.cfg.dist_fcn_callable,
                                         self.cfg.dist_th_tp, self.cfg.min_recall,
                                         num_thresholds=TrackingMetricData.nelem,
                                         metric_worst=self.cfg.metric_worst,
                                         verbose=self.verbose,
                                         output_dir=self.output_dir,
                                         render_classes=self.render_classes)
            curr_md = curr_ev.accumulate() # 返回的是一个TrackingMetricData对象
            metric_data_list.set(curr_class_name, curr_md) # 记录该类的metric_data

        # 逐类计算
        for class_name in self.cfg.class_names:
            accumulate_class(class_name)

        # -----------------------------------
        # Step 2: Aggregate metrics from the metric data.
        # -----------------------------------
        if self.verbose:
            print('Calculating metrics...')
        # 逐类别计算
        for class_name in self.cfg.class_names:
            # Find best MOTA to determine threshold to pick for traditional metrics.
            # If multiple thresholds have the same value, pick the one with the highest recall.
            # 找到最佳MOTA以确定选择传统指标的阈值
            # 如果多个阈值具有相同的值，则选择召回率最高的一个
            md = metric_data_list[class_name] # 提取对应的TrackingMetricData对象
            if np.all(np.isnan(md.mota)): # 如果当前的mota都是nan
                best_thresh_idx = None # 将best_thresh_idx设置为None
            else:
                best_thresh_idx = np.nanargmax(md.mota) # 找到非nan的最大mota的索引

            # Pick best value for traditional metrics.
            if best_thresh_idx is not None:
                # 逐个指标计算
                for metric_name in MOT_METRIC_MAP.values():
                    if metric_name == '':
                        continue
                    value = md.get_metric(metric_name)[best_thresh_idx] # 获取最大mota对应的指标值, 标量
                    metrics.add_label_metric(metric_name, class_name, value) # 在metrics中添加对应指标

            # Compute AMOTA / AMOTP. 计算AMOTA和AMOTP
            for metric_name in AVG_METRIC_MAP.keys():
                values = np.array(md.get_metric(AVG_METRIC_MAP[metric_name])) # motar和motp-->(40,)
                assert len(values) == TrackingMetricData.nelem # 40

                if np.all(np.isnan(values)):
                    # If no GT exists, set to nan.
                    value = np.nan
                else:
                    # Overwrite any nan value with the worst possible value.
                    # 将nan的值变为false
                    np.all(values[np.logical_not(np.isnan(values))] >= 0) # 取保存在有效值
                    values[np.isnan(values)] = self.cfg.metric_worst[metric_name] # 用最差的值覆盖nan值
                    value = float(np.nanmean(values)) # 求平均值计算AMOTA和AMOTP
                metrics.add_label_metric(metric_name, class_name, value)

        # Compute evaluation time.
        metrics.add_runtime(time.time() - start_time)

        return metrics, metric_data_list

    def render(self, md_list: TrackingMetricDataList) -> None:
        """
        Renders a plot for each class and each metric.
        :param md_list: TrackingMetricDataList instance.
        """
        if self.verbose:
            print('Rendering curves')

        def savepath(name):
            return os.path.join(self.plot_dir, name + '.pdf')

        # Plot a summary.
        summary_plot(self.cfg, md_list, savepath=savepath('summary'))

        # For each metric, plot all the classes in one diagram.
        for metric_name in LEGACY_METRICS:
            recall_metric_curve(self.cfg, md_list, metric_name, savepath=savepath('%s' % metric_name))

    def main(self, render_curves: bool = True) -> Dict[str, Any]:
        """
        Main function that loads the evaluation code, visualizes samples, runs the evaluation and renders stat plots.
        :param render_curves: Whether to render PR and TP curves to disk.
        :return: The serialized TrackingMetrics computed during evaluation.
        """
        # Run evaluation.
        metrics, metric_data_list = self.evaluate()

        # Dump the metric data, meta and metrics to disk.
        if self.verbose:
            print('Saving metrics to: %s' % self.output_dir)
        metrics_summary = metrics.serialize() # 在序列化的同时在计算类的平均值
        metrics_summary['meta'] = self.meta.copy()
        with open(os.path.join(self.output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=2) # 将summary写入文件
        with open(os.path.join(self.output_dir, 'metrics_details.json'), 'w') as f:
            json.dump(metric_data_list.serialize(), f, indent=2) # 将细节写入文件

        # Print metrics to stdout.
        if self.verbose:
            print_final_metrics(metrics) # 打印metrics到标准输出

        # Render curves.
        if render_curves:
            self.render(metric_data_list)

        return metrics_summary


if __name__ == "__main__":

    # Settings.
    parser = argparse.ArgumentParser(description='Evaluate nuScenes tracking results.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('result_path', type=str, help='The submission as a JSON file.')
    parser.add_argument('--output_dir', type=str, default='~/nuscenes-metrics',
                        help='Folder to store result metrics, graphs and example visualizations.')
    parser.add_argument('--eval_set', type=str, default='val',
                        help='Which dataset split to evaluate on, train, val or test.')
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuscenes',
                        help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='Which version of the nuScenes dataset to evaluate on, e.g. v1.0-trainval.')
    parser.add_argument('--config_path', type=str, default='',
                        help='Path to the configuration file.'
                             'If no path given, the NIPS 2019 configuration will be used.')
    parser.add_argument('--render_curves', type=int, default=1,
                        help='Whether to render statistic curves to disk.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to print to stdout.')
    parser.add_argument('--render_classes', type=str, default='', nargs='+',
                        help='For which classes we render tracking results to disk.')
    args = parser.parse_args()

    result_path_ = os.path.expanduser(args.result_path)
    output_dir_ = os.path.expanduser(args.output_dir)
    eval_set_ = args.eval_set
    dataroot_ = args.dataroot
    version_ = args.version
    config_path = args.config_path
    render_curves_ = bool(args.render_curves)
    verbose_ = bool(args.verbose)
    render_classes_ = args.render_classes

    if config_path == '':
        cfg_ = config_factory('tracking_nips_2019')
    else:
        with open(config_path, 'r') as _f:
            cfg_ = TrackingConfig.deserialize(json.load(_f))

    nusc_eval = TrackingEval(config=cfg_, result_path=result_path_, eval_set=eval_set_, output_dir=output_dir_,
                             nusc_version=version_, nusc_dataroot=dataroot_, verbose=verbose_,
                             render_classes=render_classes_)
    nusc_eval.main(render_curves=render_curves_)
