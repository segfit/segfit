import sys, os
import numpy as np
import cv2
import pickle, gzip
from tqdm import tqdm, trange
from voc_eval import voc_ap
import scipy.sparse
from PIL import Image

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def cal_one_mean_iou(image_array, label_array, NUM_CLASSES):
    hist = fast_hist(label_array, image_array, NUM_CLASSES).astype(np.float)
    num_cor_pix = np.diag(hist)
    num_gt_pix = hist.sum(1)
    union = num_gt_pix + hist.sum(0) - num_cor_pix
    iu = num_cor_pix / (num_gt_pix + hist.sum(0) - num_cor_pix)
    return iu

def get_gt(list_dat, task_id=None):
    if task_id is not None:
        if os.path.isfile('cache/gt_record_{}.pkl'.format(task_id)):
            cached = pickle.load(open('cache/gt_record_{}.pkl'.format(task_id)))
            return cached['class_recs'], cached['npos']
         
    class_recs = {}
    npos = 0

    for dat in tqdm(list_dat, desc='Loading gt..'):
        imagename = dat['filepath'].split('/')[-1].replace('.jpg','')
        if len(dat['bboxes']) == 0:
            gt_box=np.array([])
            det = []
            anno_adds = []
        else:
            gt_box = []
            anno_adds = []
            for bbox in dat['bboxes']:
                mask_gt = np.array(Image.open(bbox['ann_path']))
                if len(mask_gt.shape)==3: mask_gt = mask_gt[:,:,0] # Make sure ann is a two dimensional np array. 
                if np.sum(mask_gt>0)==0: continue
                anno_adds.append(bbox['ann_path'])
                gt_box.append((bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']))
                npos = npos + 1 

            det = [False] * len(anno_adds)
        class_recs[imagename] = {'gt_box': np.array(gt_box),
                                 'anno_adds': anno_adds, 
                                 'det': det}
    return class_recs, npos

    
def eval_seg_ap(results_all, dat_list, nb_class=59, ovthresh_seg=0.5, Sparse=False, From_pkl=False, task_id=None):
    '''
    From_pkl: load results from pickle files 
    Sparse: Indicate that the masks in the results are sparse matrices
    '''
    confidence = []
    image_ids  = []
    BB = []
    Local_segs_ptr = []

    for imagename in tqdm(results_all.keys(), desc='Loading results ..'):
        if From_pkl:
            results = pickle.load(gzip.open(results_all[imagename]))
        else:
            results = results_all[imagename]

        det_rects = results['DETS']
        for idx, rect in enumerate(det_rects):
            image_ids.append(imagename)
            confidence.append(rect[-1])
            BB.append(rect[:4])
            Local_segs_ptr.append(idx)

    confidence = np.array(confidence)
    BB = np.array(BB)
    Local_segs_ptr = np.array(Local_segs_ptr)

    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    Local_segs_ptr = Local_segs_ptr[sorted_ind]
    image_ids =  [image_ids[x]  for x in sorted_ind]


    class_recs, npos = get_gt(dat_list, task_id=task_id)
    nd = len(image_ids)
    tp_seg = np.zeros(nd)
    fp_seg = np.zeros(nd)
    pcp_list= []

    print(BB)

    for d in trange(nd, desc='Finding AP^P at thres %f..'%ovthresh_seg):
        R = class_recs[image_ids[d]]
        print(R)
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        jmax = -1
        if From_pkl:
            results = pickle.load(gzip.open(results_all[image_ids[d]]))
        else:
            results = results_all[image_ids[d]]

        print("results")
        print(results)

        mask0 = results['MASKS'][Local_segs_ptr[d]]
        if Sparse:
            mask_pred = mask0.toarray().astype(np.int) # decode sparse array if it is one
        else:
            mask_pred = mask0.astype(np.int)

        for i in range(len(R['anno_adds'])):
            mask_gt = np.array(Image.open(R['anno_adds'][i]))
            if len(mask_gt.shape)==3: mask_gt = mask_gt[:,:,0] # Make sure ann is a two dimensional np array. 

            seg_iou= cal_one_mean_iou(mask_pred.astype(np.uint8), mask_gt, nb_class)

            mean_seg_iou = np.nanmean(seg_iou)
            if mean_seg_iou > ovmax:
                ovmax =  mean_seg_iou
                seg_iou_max = seg_iou 
                jmax = i
                mask_gt_u = np.unique(mask_gt)

        # mask_gt_u = number of body parts in gt human

        if ovmax > ovthresh_seg:
            if not R['det'][jmax]:  # mark already used detections
                tp_seg[d] = 1.
                R['det'][jmax] = 1
                pcp_d = len(mask_gt_u[np.logical_and(mask_gt_u>0, mask_gt_u<nb_class)])
                pcp_n = float(np.sum(seg_iou_max[1:]>ovthresh_seg)) # count instances which are at least partly detected
                if pcp_d > 0:
                    pcp_list.append(pcp_n/pcp_d)
                else:
                    pcp_list.append(0.0)
            else:
                fp_seg[d] = 1.
        else:
            fp_seg[d] = 1.

    # compute precision recall
    fp_seg = np.cumsum(fp_seg)
    tp_seg = np.cumsum(tp_seg)
    rec_seg = tp_seg / float(npos)
    prec_seg = tp_seg / (tp_seg + fp_seg)

    ap_seg = voc_ap(rec_seg, prec_seg)

    assert(np.max(tp_seg) == len(pcp_list)), "%d vs %d"%(np.max(tp_seg),len(pcp_list))
    pcp_list.extend([0.0]*(npos - len(pcp_list)))
    pcp = np.mean(pcp_list)

    print('AP_seg, PCP:', ap_seg, pcp)
    return ap_seg, pcp


def get_prediction_from_gt(dat_list, NUM_CLASSES, cache_pkl=False, cache_pkl_path='tmp/', Sparse=False):
    '''
    cache_pkl: if the memory can't hold all the results, set cache_pkl to be true to pickle down the results 
    Sparse: Sparsify the masks to save memory
    '''
    results_all = {}
    for dat in tqdm(dat_list, desc='Generating predictions ..'):
        results = {} 

        dets, masks = [], []
        for bbox in dat['bboxes']:
            mask_gt = np.array(Image.open(bbox['ann_path']))
            if len(mask_gt.shape)==3: mask_gt = mask_gt[:,:,0] # Make sure ann is a two dimensional np array. 
            if np.sum(mask_gt)==0: continue
            ys, xs = np.where(mask_gt>0)
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            dets.append((x1, y1, x2, y2, 1.0))
            masks.append(mask_gt)            

        # Reuiqred Field of each result: a list of masks, each is a multi-class masks for one person.
            # It can also be sparsified to [scipy.sparse.csr_matrix(mask) for mask in masks] to save memory cost
        results['MASKS']= masks if not Sparse else [scipy.sparse.csr_matrix(mask) for mask in masks]
        # Reuiqred Field of each result, a list of detections corresponding to results['MASKS']. 
        results['DETS'] = dets    

        key = dat['filepath'].split('/')[-1].replace('.jpg', '')
        if cache_pkl:
            results_cache_add = cache_pkl_path + key + '.pklz'
            pickle.dump(results, gzip.open(results_cache_add, 'w'))
            results_all[key] = results_cache_add
        else:
            results_all[key]=results
    return results_all


def cache_gt_record():
    for set_ in ['val', 'test_all', 'test_inter_top20', 'test_inter_top10']:
        dat_list = pickle.load(open('cache/dat_list_{}.pkl'.format(set_)))
        class_recs, npos = get_gt(dat_list)
        pickle.dump({'class_recs':class_recs, 'npos': npos}, open('cache/gt_record_{}.pkl'.format(set_), 'w'))
        

if __name__ == '__main__':
    import mhp_data
    data_root = '/home/lijianshu/MultiPerson/data/LV-MHP-v2/'
    # set_ in ['train', 'val', 'test_all', 'test_inter_top20', 'test_inter_top10'])
    set_ = 'val'
    dat_list = mhp_data.get_data(data_root, set_)
    #dat_list = pickle.load(open('cache/dat_list_val.pkl'))
    
    NUM_CLASSES = 59
    results_all = get_prediction_from_gt(dat_list, NUM_CLASSES, cache_pkl=False, Sparse=False)
    eval_seg_ap(results_all, dat_list, nb_class=NUM_CLASSES,ovthresh_seg=0.5, From_pkl=False, Sparse=False)
