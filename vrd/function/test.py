import os
import pprint
import shutil

import json
from tqdm import tqdm, trange
import numpy as np
import torch
import torch.nn.functional as F

from common.utils.load import smart_load_model_state_dict
from common.trainer import to_cuda
from common.utils.create_logger import create_logger
from vrd.data.build import make_dataloader
from vrd.modules import *

from collections import defaultdict
import pickle

@torch.no_grad()
def test_net(args, config, ckpt_path=None, save_path=None, save_name=None):
    print('test net...')
    pprint.pprint(args)
    pprint.pprint(config)
    device_ids = [int(d) for d in config.GPUS.split(',')]

    obj_cats = config.OBJECT_CATEGORIES
    pred_cats = config.PREDICATE_CATEGORIES

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if ckpt_path is None:
        _, train_output_path = create_logger(config.OUTPUT_PATH, args.cfg, 
                                             config.DATASET.TRAIN_IMAGE_SET,
                                             split='train')
        model_prefix = os.path.join(train_output_path, config.MODEL_PREFIX)
        ckpt_path = '{}-best.model'.format(model_prefix)
        print('Use best checkpoint {}...'.format(ckpt_path))
    if save_path is None:
        logger, test_output_path = create_logger(config.OUTPUT_PATH, args.cfg, 
                                                 config.DATASET.TEST_IMAGE_SET,
                                                 split='test')
        save_path = test_output_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    shutil.copy2(ckpt_path,
                 os.path.join(save_path, '{}_test_ckpt_{}.model'.format(
                    config.MODEL_PREFIX, config.DATASET.TASK
                    )))

    # get network
    model = eval(config.MODULE)(config)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
    else:
        torch.cuda.set_device(device_ids[0])
        model = model.cuda()
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    smart_load_model_state_dict(model, checkpoint['state_dict'])

    split = args.split
    loader = make_dataloader(config, mode=split, distributed=False)

    nb_of_correct_50 = nb_of_sample = nb_of_correct_top100 = 0
    model.eval()

    save_dir = ''
    if args.visualize_mask: # For mask visualization purpose
        save_dir = 'heatmap/vrd'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    for nbatch, batch in zip(trange(len(loader)), loader):
        batch = to_cuda(batch)
        output = model(*batch)

        n_correct, n_sample, n_correct_top100 = compute_recall(output, obj_cats, pred_cats, remove_bg=config.TRAIN.SAMPLE_RELS != -1, visualize_mask=args.visualize_mask, save_dir=save_dir)
        nb_of_correct_50 += n_correct
        nb_of_correct_top100 += n_correct_top100
        nb_of_sample += n_sample
        
    recall_50 = nb_of_correct_50 / nb_of_sample
    recall_100 = nb_of_correct_top100 / nb_of_sample

    return recall_50, recall_100


def compute_recall(outputs, obj_cats, pred_cats, remove_bg=False, visualize_mask=False, save_dir=False):
    labels = outputs['label']
    labels_so_ids = outputs['labels_so_ids']
    logits = outputs['label_logits']
    if remove_bg: # do NOT consider background prediction: zero out bg prob
        logits[:, 0] = 0
    rels_cand = outputs['rels_cand']

    pred = logits.argmax(dim=1)
    logits = logits[pred != 0]
    rels_cand = rels_cand[pred != 0]
    
    if visualize_mask: # For mask visualization purpose
        img = cv2.imread('data/vrd/images/sg_test_images/' + outputs['img_path'][0], 1)
        img = np.float32(cv2.resize(img, (224, 224))) / 255
        mask = outputs['spo_fused_masks']
        mask = mask[pred != 0]

        subj_obj_classes = outputs['subj_obj_classes']
        subj_obj_classes = subj_obj_classes[pred != 0]
    pred = pred[pred != 0]

    pred_conf = logits[[i for i in range(pred.shape[0])], [pred.cpu().tolist()]].squeeze()
    if len(pred_conf.shape) == 0:
        pred_conf = pred_conf.unsqueeze(0)

    values, indices = pred_conf.sort()
    nb_of_non_nan_values = len(values[values==values])
    pred_conf = indices[:nb_of_non_nan_values].cpu().tolist()
    pred_conf.reverse()
    
    pred_conf_top100 = pred_conf[:]
    pred_top100 = pred[pred_conf_top100].cpu().tolist()
    rels_cand_top100 = rels_cand[pred_conf_top100].cpu().tolist()
    rels_cand_pred_top100 = {tuple(k): v for k, v in zip(rels_cand_top100, pred_top100)}
    
    pred_conf = pred_conf[:]
    pred = pred[pred_conf].cpu().tolist()
    rels_cand = rels_cand[pred_conf].cpu().tolist()
    rels_cand_pred = {tuple(k): v for k, v in zip(rels_cand, pred)}

    if visualize_mask: # For mask visualization purpose
        mask = mask[pred_conf].cpu()
        rels_cand_mask = {tuple(k): v for k, v in zip(rels_cand, mask)}
        
        subj_obj_classes = subj_obj_classes[pred_conf].cpu()
        rels_cand_subj_obj_classes = {tuple(k): v for k, v in zip(rels_cand, subj_obj_classes)}
    
    correct_top100 = 0
    correct = 0
    for idx, rel in enumerate(labels_so_ids.cpu().tolist()):
        if not tuple(rel) in rels_cand_pred_top100:
            continue
        if rels_cand_pred_top100[tuple(rel)] == labels[idx].cpu().item():
            correct_top100 += 1

        if not tuple(rel) in rels_cand_pred:
            continue
        if rels_cand_pred[tuple(rel)] == labels[idx].cpu().item():
            correct += 1
            if visualize_mask: # For mask visualization purpose
                show_cam_on_image(img, rels_cand_mask[tuple(rel)], outputs['img_path'][0], rels_cand_subj_obj_classes[tuple(rel)], obj_cats, pred_cats[labels[idx]], save_dir)

    return correct, labels_so_ids.shape[0], correct_top100

import cv2
def show_cam_on_image(img, mask, img_name, subj_obj_classes, obj_cats, pred_cat, save_dir):
    subj_cat = obj_cats[subj_obj_classes[0]]
    obj_cat = obj_cats[subj_obj_classes[1]]

    for i in range(mask.shape[0]):
        mask2 = F.interpolate(mask[i].unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear').squeeze()
        heatmap = cv2.applyColorMap(np.uint8(255 * mask2), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)

        save_name = f"{img_name}-{subj_cat}-{pred_cat}-{obj_cat}-subject.jpg" if i == 0 else f"{img_name}-{subj_cat}-{pred_cat}-{obj_cat}-object.jpg"
        cv2.imwrite(os.path.join(save_dir, save_name), np.uint8(255 * cam))

        save_heatmap_name = f"{img_name}-{subj_cat}-{pred_cat}-{obj_cat}-subject-heatmap.jpg" if i == 0 else f"{img_name}-{subj_cat}-{pred_cat}-{obj_cat}-object-heatmap.jpg"
        cv2.imwrite(os.path.join(save_dir, save_heatmap_name), np.uint8(255 * heatmap))