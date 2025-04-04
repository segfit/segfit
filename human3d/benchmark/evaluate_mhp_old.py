import numpy as np


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


def load_gt(file_path):
    ids = open(file_path).read().splitlines()
    ids = np.array(ids, dtype=np.int64)
    gt_human_mask = ids % 1000
    gt_part_mask = ids // 1000
    return gt_human_mask, gt_part_mask


def evaluate(preds: dict, gt_path: str, output_file: str):
    scenes_for_eval = list(preds.keys())
    base_path = f"data/processed/human_segmentation/gt_part/train/"

    overlap_threshold_parts = 0.5
    true_positive_parts = np.zeros(len(scenes_for_eval))
    false_positive_parts = np.zeros(len(scenes_for_eval))
    pcp_list= []

    number_all_humans = 0

    for scene_id, scene_name in enumerate(scenes_for_eval):
        gt_file_path = f"{base_path}/{scene_name}.txt"
        gt_human_mask, gt_part_mask = load_gt(gt_file_path)

        max_overlap = -np.inf

        unique_human_idx = np.unique(gt_human_mask)[1:]
        number_all_humans += len(unique_human_idx)

        matched_gt_human = {k: False for k in unique_human_idx}

        for gt_human_id in unique_human_idx:
            gt_part_single_human = gt_part_mask.copy()
            gt_part_single_human[gt_human_mask != gt_human_id] = 0

            pred_humans = preds[scene_name]['pred_human_scores']
            pred_body_parts = preds[scene_name]['body_semseg']

            for pred_human_id in pred_humans.sort(descending=True).indices:
                hist = fast_hist(gt_part_single_human, pred_body_parts[pred_human_id, :], 27)
                num_cor_pix = np.diag(hist)
                num_gt_pix = hist.sum(1)
                part_iou = num_cor_pix / (num_gt_pix + hist.sum(0) - num_cor_pix)
                mean_part_iou = np.nanmean(part_iou)

                if mean_part_iou > max_overlap:
                    max_overlap = mean_part_iou
                    matched_part_iou = part_iou
                    matched_gt_human_id = gt_human_id
                    gt_human_visible_parts = np.unique(gt_part_single_human)

            if max_overlap > overlap_threshold_parts:
                if not matched_gt_human[matched_gt_human_id]:
                    true_positive_parts[scene_id] += 1.
                    matched_gt_human[matched_gt_human_id] = True  # mark already matched human

                    # count instances which are at least partly detected (> overlap threshold)
                    percentage_correct_parts_denominator = len(gt_human_visible_parts[np.logical_and(gt_human_visible_parts > 0,
                                                                                                     gt_human_visible_parts < 27)])

                    # matched_part_iou[1:] in order to ignore background
                    percentage_correct_parts_nominator = float(
                        np.sum(matched_part_iou[1:] > overlap_threshold_parts))
                    if percentage_correct_parts_denominator > 0:
                        pcp_list.append(percentage_correct_parts_nominator / percentage_correct_parts_denominator)
                    else:
                        pcp_list.append(0.0)
                else:
                    false_positive_parts[scene_id] += 1.
            else:
                false_positive_parts[scene_id] += 1.

    # compute precision recall
    false_positive_parts = np.cumsum(false_positive_parts)
    true_positive_parts = np.cumsum(true_positive_parts)
    rec_seg = true_positive_parts / float(number_all_humans)
    prec_seg = true_positive_parts / (true_positive_parts + false_positive_parts)

    ap_seg = voc_ap(rec_seg, prec_seg)

    assert (np.max(true_positive_parts) == len(pcp_list)), "%d vs %d" % (np.max(true_positive_parts), len(pcp_list))
    pcp_list.extend([0.0] * (number_all_humans - len(pcp_list)))
    pcp = np.mean(pcp_list)

    print('AP_seg, PCP:', ap_seg, pcp)
    return ap_seg, pcp