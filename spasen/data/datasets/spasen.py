import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import json
import random
import numpy as np
import os.path
import cv2
import math
import sys
import pickle
from PIL import Image, ImageDraw
from .util import read_img, phrase2vec, onehot
from collections import defaultdict
np.random.seed(0)

from common.utils.create_logger import makedirsExist
from external.pytorch_pretrained_bert import BertTokenizer

class SpaSen(Dataset):
    def __init__(self, split, cfg, transform):
        super().__init__()
        self.split = split
        self.cfg = cfg
        self.transform = transform

        self.annotations = []
        n_img = 0
        for img in json.load(open(self.cfg.DATAPATH)):
            split = split + 'id' if split == 'val' else split # 'val' -> 'valid'
            if img['split'] in split.split('_'): # if img['split'] == split:
                n_img += 1
                for annot in img['annotations']:
                    if cfg.TEST.EXCL_LEFT_RIGHT and (annot['predicate'] == 'to the left of' or annot['predicate'] == 'to the right of'):
                        continue

                    annot['url'] = img['url']
                    annot['height'] = img['height']
                    annot['width'] = img['width']
                    annot['subject']['bbox'] = self.fix_bbox(annot['subject']['bbox'], img['height'], img['width'])
                    annot['object']['bbox'] = self.fix_bbox(annot['object']['bbox'], img['height'], img['width'])
                    self.annotations.append(annot)

        print('%d relations in %s' % (len(self.annotations), split))
        print('%d imgs in %s' % (n_img, split))

        self.cache_dir = os.path.join(cfg.DATASET.ROOT_PATH, 'cache')
        if not os.path.exists(self.cache_dir):
            makedirsExist(self.cache_dir)
        if cfg.NETWORK.BERT_MODEL_NAME:
            print('Initializing BERT tokenizer from', cfg.NETWORK.BERT_MODEL_NAME)
        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased' if cfg.NETWORK.BERT_MODEL_NAME is None else cfg.NETWORK.BERT_MODEL_NAME,
            cache_dir=self.cache_dir)
        
    def __len__(self):
        return len(self.annotations) # (Pdb) len(self.annotations) = 11238

    @property
    def data_names(self):
        return ['_id', 'img', 'boxes','spo_ids', 'spo_len', 'label', 'im_info', 'predicate', 'im_path', 'subj_name', 'obj_name', 'pred_name']
    
    def __getitem__(self, idx):
        annot = self.annotations[idx]

        xs0, xs1, ys0, ys1 = annot['subject']['bbox']
        xo0, xo1, yo0, yo1 = annot['object']['bbox']

        ih, iw = annot['height'], annot['width']
        img = read_img(annot['url'], self.cfg.IMAGEPATH)

        spasen_to_vlbert_bbox_idx = [2, 0, 3, 1]
        full_img_bbox = np.asarray([0.0, ih, 0.0, iw], dtype=np.float32)
        full_img_bbox = full_img_bbox[spasen_to_vlbert_bbox_idx].reshape(1, -1)
        subj_bbox = np.asarray(annot['subject']['bbox'], dtype=np.float32) # [x0, x1, y0, y1]
        subj_bbox = subj_bbox[spasen_to_vlbert_bbox_idx].reshape(1, -1) # -> [x0, y0, x1(width), y1(height)]
        union_bbox = np.asarray(self._getUnionBBox(annot['subject']['bbox'], annot['object']['bbox'], ih, iw), dtype=np.float32)
        union_bbox = union_bbox[spasen_to_vlbert_bbox_idx].reshape(1, -1)
        obj_bbox = np.asarray(annot['object']['bbox'], dtype=np.float32)
        obj_bbox = obj_bbox[spasen_to_vlbert_bbox_idx].reshape(1, -1)
        boxes = torch.as_tensor(np.concatenate((full_img_bbox, subj_bbox, union_bbox, obj_bbox)))

        predicate = annot['predicate']
        im_info = torch.tensor([iw, ih, 1.0, 1.0])
        flipped = False
        if self.transform is not None:
            img, boxes, _, predicate, im_info, flipped = self.transform(img, boxes, None, predicate, im_info, flipped)

        subject_token = self.tokenizer.tokenize(annot['subject']['name'])
        subject_id = self.tokenizer.convert_tokens_to_ids(subject_token)
        object_token = self.tokenizer.tokenize(annot['object']['name'])
        object_id = self.tokenizer.convert_tokens_to_ids(object_token)
        predicate_token = self.tokenizer.tokenize(annot['predicate'])
        predicate_id = self.tokenizer.convert_tokens_to_ids(predicate_token)
        
        spo_ids = torch.tensor(subject_id + predicate_id + object_id)
        spo_len = torch.tensor([len(subject_id), len(object_id), len(predicate_id)])
        
        datum = {
            '_id': annot['_id'], # for computing class-wise accuracy
            'img': img,
            'boxes': boxes,
            'predicate': onehot(self.cfg.PREDICATE_CATEGORIES.index(predicate), 9),
            'spo_ids': spo_ids,
            'spo_len': spo_len,
            'label': np.asarray([1 if annot['label'] is True else 0], dtype=np.float32),
            'im_info': im_info,
            'im_path': annot['url'],
            'subj_name': annot['subject']['name'],
            'obj_name': annot['object']['name'],
            'pred_name': annot['predicate'],
        }
    
        return tuple([datum[key] for key in self.data_names])


    def enlarge(self, bbox, factor, ih, iw):
        height = bbox[1] - bbox[0]
        width = bbox[3] - bbox[2]
        assert height > 0 and width > 0
        return [max(0, int(bbox[0] - (factor - 1.) * height / 2.)),
                min(ih, int(bbox[1] + (factor - 1.) * height / 2.)),
                max(0, int(bbox[2] - (factor - 1.) * width / 2.)),
                min(iw, int(bbox[3] + (factor - 1.) * width / 2.))]


    def _getAppr(self, im, bb, out_size=224.):
        subim = im[bb[0] : bb[1], bb[2] : bb[3], :]
        subim = cv2.resize(subim, None, None, out_size / subim.shape[1], out_size / subim.shape[0], interpolation=cv2.INTER_LINEAR)
        subim = (subim / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        return subim.astype(np.float32, copy=False)


    def _getUnionBBox(self, aBB, bBB, ih, iw, margin = 10):
        return [max(0, min(aBB[0], bBB[0]) - margin), \
                min(ih, max(aBB[1], bBB[1]) + margin), \
                max(0, min(aBB[2], bBB[2]) - margin), \
                min(iw, max(aBB[3], bBB[3]) + margin)]


    def _getDualMask(self, ih, iw, bb, heatmap_size=32):
        rh = float(heatmap_size) / ih
        rw = float(heatmap_size) / iw
        x1 = max(0, int(math.floor(bb[0] * rh)))
        x2 = min(heatmap_size, int(math.ceil(bb[1] * rh)))
        y1 = max(0, int(math.floor(bb[2] * rw)))
        y2 = min(heatmap_size, int(math.ceil(bb[3] * rw)))
        mask = np.zeros((heatmap_size, heatmap_size), dtype=np.float32)
        mask[x1 : x2, y1 : y2] = 255
        #assert(mask.sum() == (y2 - y1) * (x2 - x1))
        return mask

    def _getT(self, bbox1, bbox2):
        h1 = bbox1[1] - bbox1[0]
        w1 = bbox1[3] - bbox1[2]
        h2 = bbox2[1] - bbox2[0]
        w2 = bbox2[3] - bbox2[2]
        return [(bbox1[0] - bbox2[0]) / float(h2),
                (bbox1[2] - bbox2[2]) / float(w2),
                math.log(h1 / float(h2)),
                math.log(w1 / float(w2))]


    def fix_bbox(self, bbox, ih, iw):
        if (bbox[1] - bbox[0] < 20):
            if bbox[0] > 10:
                bbox[0] -= 10
            if bbox[1] < ih - 10:
                bbox[1] += 10

        if (bbox[3] - bbox[2] < 20):
            if bbox[2] > 10:
                bbox[2] -= 10
            if bbox[3] < iw - 10:
                bbox[3] += 1
        return bbox

