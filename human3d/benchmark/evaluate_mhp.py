# Evaluates semantic instance task
# Adapted from the CityScapes evaluation: https://github.com/mcordts/cityscapesScripts/tree/master/cityscapesscripts/evaluation

import os
from tqdm import tqdm
from copy import deepcopy
from uuid import uuid4
import numpy as np
import benchmark.util as util
import benchmark.util_3d_mhp as util_3d

# ---------- Evaluation params ---------- #
# overlaps for evaluation
opt = {}
opt["overlaps"] = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
# minimum region size for evaluation [verts]
opt["min_region_sizes"] = np.array([100])  # 100 for s3dis, scannet
# distance thresholds [m]
opt["distance_threshes"] = np.array([float("inf")])
# distance confidences
opt["distance_confs"] = np.array([-float("inf")])


def evaluate_matches(matches):
    overlaps = opt["overlaps"]
    min_region_sizes = [opt["min_region_sizes"][0]]
    dist_threshes = [opt["distance_threshes"][0]]
    dist_confs = [opt["distance_confs"][0]]

    # results: class x overlap
    ap = np.zeros((len(dist_threshes), len(overlaps)), float)
    pcp = np.zeros((len(dist_threshes), len(overlaps)), float)
    for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
        zip(min_region_sizes, dist_threshes, dist_confs)
    ):
        for oi, overlap_th in enumerate(overlaps):
            pred_visited = {}
            for m in matches:
                for p in matches[m]["pred"]:
                    pred_visited[p["uuid"]] = False

            y_true = np.empty(0)
            y_score = np.empty(0)

            cur_pcps = []

            # y_pcp = np.empty(0)
            hard_false_negatives = 0
            has_gt = False
            has_pred = False
            for m in matches:
                pred_instances = matches[m]["pred"]
                gt_instances = matches[m]["gt"]
                # filter groups in ground truth
                # gt_instances = [ gt for gt in gt_instances if gt['instance_id']>=1000 and gt['vert_count']>=min_region_size and gt['med_dist']<=distance_thresh and gt['dist_conf']>=distance_conf ]
                gt_instances = [
                    gt for gt in gt_instances if gt["vert_count"] >= min_region_size
                ]
                if gt_instances:
                    has_gt = True
                if pred_instances:
                    has_pred = True

                cur_true = np.ones(len(gt_instances))
                cur_score = np.ones(len(gt_instances)) * (-float("inf"))

                cur_pcp = np.zeros(len(gt_instances))

                cur_match = np.zeros(len(gt_instances), dtype=bool)
                # collect matches
                for (gti, gt) in enumerate(gt_instances):
                    found_match = False
                    num_pred = len(gt["matched_pred"])
                    for pred in gt["matched_pred"]:
                        # greedy assignments
                        if pred_visited[pred["uuid"]]:
                            continue
                        # overlap = float(pred['intersection']) / (gt['vert_count']+pred['vert_count']-pred['intersection'])
                        overlap = float(pred["intersection"])

                        if overlap > overlap_th:
                            confidence = pred["confidence"]

                            gt_human_visible_parts = np.unique(gt["body_semseg"])
                            gt_human_visible_parts = gt_human_visible_parts[
                                gt_human_visible_parts != 0
                            ]

                            # count instances which are at least partly detected (> overlap threshold)
                            percentage_correct_parts_denominator = len(
                                gt_human_visible_parts
                            )

                            percentage_correct_parts_nominator = float(
                                np.sum(pred["part_iou"] > overlap_th)
                            )
                            if percentage_correct_parts_denominator > 0:
                                single_pcp = (
                                    percentage_correct_parts_nominator
                                    / percentage_correct_parts_denominator
                                )
                            else:
                                single_pcp = 0.0

                            # if already have a prediction for this gt,
                            # the prediction with the lower score is automatically a false positive
                            if cur_match[gti]:
                                max_score = max(cur_score[gti], confidence)
                                min_score = min(cur_score[gti], confidence)
                                cur_score[gti] = max_score

                                # max_pcp = max(cur_pcp[gti], single_pcp)
                                # min_pcp = min(cur_pcp[gti], single_pcp)
                                cur_pcp[gti] = max(cur_pcp[gti], single_pcp)
                                # cur_pcp = np.append(cur_pcp, 0)

                                # append false positive
                                cur_true = np.append(cur_true, 0)
                                cur_score = np.append(cur_score, min_score)
                                cur_match = np.append(cur_match, True)
                            # otherwise set score
                            else:
                                found_match = True
                                cur_match[gti] = True
                                cur_score[gti] = confidence
                                cur_pcp[gti] = single_pcp
                                pred_visited[pred["uuid"]] = True
                    if not found_match:
                        hard_false_negatives += 1
                # remove non-matched ground truth instances
                cur_true = cur_true[cur_match == True]
                cur_score = cur_score[cur_match == True]
                # cur_pcp = cur_pcp[ cur_match==True ]

                # collect non-matched predictions as false positive
                for pred in pred_instances:
                    found_gt = False
                    for gt in pred["matched_gt"]:
                        # overlap = float(gt['intersection']) / (gt['vert_count']+pred['vert_count']-gt['intersection'])
                        overlap = float(gt["intersection"])
                        if overlap > overlap_th:
                            found_gt = True
                            break
                    if not found_gt:
                        cur_true = np.append(cur_true, 0)
                        confidence = pred["confidence"]
                        cur_score = np.append(cur_score, confidence)
                        # cur_pcp = np.append(cur_pcp, pcp) not needed for false positive
                        """                      
                        num_ignore = pred['void_human_intersection']
                        for gt in pred['matched_gt']:
                            # group?
                            #if gt['instance_id'] < 1000:
                            #    num_ignore += gt['intersection']
                            # small ground truth instances
                            if gt['vert_count'] < min_region_size: # or gt['med_dist']>distance_thresh or gt['dist_conf']<distance_conf:
                                num_ignore += gt['intersection']
                        proportion_ignore = float(num_ignore)/pred['vert_count']
                        # if not ignored append false positive
                        if proportion_ignore <= overlap_th:
                            cur_true = np.append(cur_true,0)
                            confidence = pred["confidence"]
                            cur_score = np.append(cur_score,confidence)"""

                # append to overall results
                y_true = np.append(y_true, cur_true)
                y_score = np.append(y_score, cur_score)
                # y_pcp = np.append(y_pcp, cur_pcp)
                cur_pcps.append(cur_pcp)

            # compute average precision
            if has_gt and has_pred:
                # compute precision recall curve first

                # sorting and cumsum
                score_arg_sort = np.argsort(y_score)
                y_score_sorted = y_score[score_arg_sort]
                y_true_sorted = y_true[score_arg_sort]
                y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                # unique thresholds
                (thresholds, unique_indices) = np.unique(
                    y_score_sorted, return_index=True
                )
                num_prec_recall = len(unique_indices) + 1

                # prepare precision recall
                num_examples = len(y_score_sorted)
                # https://github.com/ScanNet/ScanNet/pull/26
                # all predictions are non-matched but also all of them are ignored and not counted as FP
                # y_true_sorted_cumsum is empty
                # num_true_examples = y_true_sorted_cumsum[-1]
                num_true_examples = (
                    y_true_sorted_cumsum[-1] if len(y_true_sorted_cumsum) > 0 else 0
                )
                precision = np.zeros(num_prec_recall)
                recall = np.zeros(num_prec_recall)

                # deal with the first point
                y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                # deal with remaining
                for idx_res, idx_scores in enumerate(unique_indices):
                    cumsum = y_true_sorted_cumsum[idx_scores - 1]
                    tp = num_true_examples - cumsum
                    fp = num_examples - idx_scores - tp
                    fn = cumsum + hard_false_negatives
                    p = float(tp) / (tp + fp)
                    r = float(tp) / (tp + fn)
                    precision[idx_res] = p
                    recall[idx_res] = r

                # first point in curve is artificial
                precision[-1] = 1.0
                recall[-1] = 0.0

                # compute average of precision-recall curve
                recall_for_conv = np.copy(recall)
                recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                recall_for_conv = np.append(recall_for_conv, 0.0)

                stepWidths = np.convolve(recall_for_conv, [-0.5, 0, 0.5], "valid")
                # integrate is now simply a dot product
                ap_current = np.dot(precision, stepWidths)

                pcp_current = np.concatenate(cur_pcps).mean()

            elif has_gt:
                ap_current = 0.0
                pcp_current = 0.0
            else:
                ap_current = float("nan")
                pcp_current = float("nan")
            ap[di, oi] = ap_current
            pcp[di, oi] = pcp_current
    return ap, pcp


def compute_averages(aps):
    d_inf = 0
    o50 = np.where(np.isclose(opt["overlaps"], 0.5))
    o25 = np.where(np.isclose(opt["overlaps"], 0.25))
    oAllBut25 = np.where(np.logical_not(np.isclose(opt["overlaps"], 0.25)))
    avg_dict = {}
    # avg_dict['all_ap']     = np.nanmean(aps[ d_inf,:,:  ])
    avg_dict["all_ap"] = np.nanmean(aps[d_inf, oAllBut25])
    avg_dict["all_ap_50%"] = np.nanmean(aps[d_inf, o50])
    avg_dict["all_ap_25%"] = np.nanmean(aps[d_inf, o25])

    return avg_dict


def make_pred_info(pred: dict):
    # pred = {'pred_scores' = 100, 'pred_classes' = 100 'pred_masks' = Nx100}
    pred_info = {}
    # assert(pred['pred_classes'].shape[0] == pred['pred_scores'].shape[0] == pred['pred_masks'].shape[1])
    for i in range(len(pred["pred_human_scores"])):
        info = {}
        # info["label_id"] = pred['pred_human_classes'][i]
        info["conf"] = pred["pred_human_scores"][i]
        info["human_mask"] = pred["pred_human_masks"][:, i]
        info["body_semseg"] = pred["body_semseg"][i, :]
        pred_info[uuid4()] = info  # we later need to identify these objects
    return pred_info


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


def assign_instances_for_scan(pred: dict, gt_file: str):
    pred_info = make_pred_info(pred)
    try:
        gt_ids = util_3d.load_ids(gt_file)
    except Exception as e:
        util.print_error("unable to load " + gt_file + ": " + str(e))

    # get gt instances
    gt_instances = util_3d.get_instances(gt_ids)
    # associate
    gt2pred = deepcopy(gt_instances)

    for gt in gt2pred:
        gt["matched_pred"] = []

    pred2gt = []
    num_pred_instances = 0
    # mask of void labels in the groundtruth
    # bool_void = gt_ids//1000 != 0
    # go through all prediction masks
    for uuid in pred_info:
        # label_id = int(pred_info[uuid]['label_id'])
        conf = pred_info[uuid]["conf"]
        # if not label_id in ID_TO_LABEL:
        #    continue
        # label_name = ID_TO_LABEL[label_id]
        # read the mask
        pred_human_mask = pred_info[uuid]["human_mask"]
        assert len(pred_human_mask) == len(gt_ids)
        # convert to binary
        # pred_human_mask = np.not_equal(pred_human_mask, 0)
        num = np.count_nonzero(pred_human_mask)
        if num < opt["min_region_sizes"][0]:
            continue  # skip if empty

        pred_instance = {}
        pred_instance["uuid"] = uuid
        pred_instance["pred_id"] = num_pred_instances
        # pred_instance['label_id'] = label_id
        pred_instance["vert_count"] = num
        pred_instance["confidence"] = conf
        # pred_instance['void_human_intersection'] = np.count_nonzero(np.logical_and(bool_void, pred_human_mask))
        pred_instance["body_semseg"] = pred_info[uuid]["body_semseg"]

        # matched gt instances
        matched_gt = []
        # go thru all gt instances with matching label
        for (gt_num, gt_inst) in enumerate(gt2pred):
            gt_part_single_human = gt_ids.copy()
            gt_part_single_human[
                gt_ids % 1000 != gt_inst["instance_id"]
            ] = 0  # other humans to background
            gt_part_single_human = gt_part_single_human // 1000
            hist = fast_hist(
                gt_part_single_human, pred_instance["body_semseg"], 16
            )  # 27)
            hist = hist[1:, 1:]
            num_cor_pix = np.diag(hist)
            num_gt_pix = hist.sum(1)
            part_iou = num_cor_pix / (num_gt_pix + hist.sum(0) - num_cor_pix)
            intersection = np.nanmean(part_iou)

            if np.isnan(intersection):
                intersection = 0.0

            # intersection = np.count_nonzero(np.logical_and(gt_ids % 1000 == gt_inst['instance_id'], pred_human_mask))
            if intersection > 0:
                gt_copy = gt_inst.copy()
                pred_copy = pred_instance.copy()
                gt_copy["intersection"] = intersection
                pred_copy["intersection"] = intersection

                gt_copy["part_iou"] = part_iou
                pred_copy["part_iou"] = part_iou

                matched_gt.append(gt_copy)
                gt2pred[gt_num]["matched_pred"].append(pred_copy)

        pred_instance["matched_gt"] = matched_gt
        num_pred_instances += 1
        pred2gt.append(pred_instance)

    return gt2pred, pred2gt


def evaluate(preds: dict, gt_path: str, output_file: str, dataset: str = "scannet"):
    matches = {}
    for (k, v) in tqdm(preds.items(), desc="evaluate human instances"):
        # gt_file = os.path.join(f"data/processed/human_segmentation/gt_part/train/", k + ".txt")
        gt_file = os.path.join(gt_path, k + ".txt")
        if not os.path.isfile(gt_file):
            util.print_error(
                "Scan {} does not match any gt file".format(k), user_fault=True
            )

        matches_key = os.path.abspath(gt_file)
        # assign gt to predictions
        gt2pred, pred2gt = assign_instances_for_scan(v, gt_file)
        matches[matches_key] = {}
        matches[matches_key]["gt"] = gt2pred
        matches[matches_key]["pred"] = pred2gt

    ap_scores, pcp_scores = evaluate_matches(matches)
    avgs = compute_averages(ap_scores)
    pcp_scores = compute_averages(pcp_scores)

    return avgs, pcp_scores
