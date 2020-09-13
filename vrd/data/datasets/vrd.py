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

class VRD(Dataset):
    def __init__(self, split, cfg, transform):
        super().__init__()
        self.split = split
        self.cfg = cfg
        self.transform = transform

        self.all_proposals_test = False
        if cfg.DATASET.ALL_PROPOSALS_TEST:
            self.all_proposals_test = True

        self.annotations = []

        # Load images
        self.path = self.cfg.TEST_PATH if split == 'test' else self.cfg.TRAIN_VAL_PATH
        imgs = json.load(open(self.path))

        skipped_count = 0
        for img in imgs:
            if img['path'].endswith('.png'):
                img['path'] = '.'.join([img['path'].split('.')[0], 'jpg'])
            
            rels_cand = None
            if self.all_proposals_test and split != 'train':
                rels_cand = []
                nb_of_objs = len(img['objects'])
                if nb_of_objs > cfg.DATASET.MAX_NB_OF_OBJ:
                    nb_of_objs = min(cfg.DATASET.MAX_NB_OF_OBJ, nb_of_objs)
                    skipped_count += 1
                for sub_id in range(0, nb_of_objs):
                    for obj_id in range(0, nb_of_objs):
                        if sub_id == obj_id: continue
                        rels_cand.append((sub_id, obj_id))

            annot = {
                'img_path': img['path'],
                'annot': img['relationships'],
                'objects': img['objects'],
                'rels_cand': rels_cand,
            }

            self.annotations.append(annot)

        print(f'number of imgs with skipped objs (skipped_count): {skipped_count}')
        print('%d imgs in %s' % (len(self.annotations), split))

        # categories
        self.num_object_classes = len(self.cfg.OBJECT_CATEGORIES)
        self._object_class_to_ind = dict(zip(self.cfg.OBJECT_CATEGORIES, range(self.num_object_classes)))
        self.num_predicate_classes = len(self.cfg.PREDICATE_CATEGORIES)
        self._predicate_class_to_ind = dict(zip(self.cfg.PREDICATE_CATEGORIES, range(self.num_predicate_classes)))

        self.cache_dir = os.path.join(cfg.DATASET.ROOT_PATH, 'cache')
        if not os.path.exists(self.cache_dir):
            makedirsExist(self.cache_dir)
        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased' if cfg.NETWORK.BERT_MODEL_NAME is None else cfg.NETWORK.BERT_MODEL_NAME,
            cache_dir=self.cache_dir)
        
        self.sample_rels = cfg.TRAIN.SAMPLE_RELS

    def __len__(self):
        return len(self.annotations) # (Pdb) len(self.annotations) = 11238

    @property
    def data_names(self):
        if self.all_proposals_test and self.split != 'train':
            return ['img', 'im_info', 'boxes', 'labels', 'spo_ids', 'spo_lens', 'img_path', 'rels_cand', 'labels_so_ids', 'subj_obj_classes']
        else:
            return ['img', 'im_info', 'boxes', 'labels', 'spo_ids', 'spo_lens', 'img_path']
    
    def __getitem__(self, idx):
        annot = self.annotations[idx]

        img = read_img(self.cfg.IMAGEPATH, annot['img_path'], self.split)
        annot['height'] = img.height
        annot['width'] = img.width

        ih, iw = annot['height'], annot['width']
        
        full_img_bbox = np.asarray([0.0, 0.0, iw, ih], dtype=np.float32).reshape(1, -1)

        nb_of_rels = len(annot['rels_cand']) if self.all_proposals_test and self.split != 'train' else len(annot['annot'])

        # sampling training pairs
        if self.sample_rels != -1 and self.split == 'train':
            nb_of_gt_rels_threshold = self.sample_rels * 3 // 4 # 24 if sample_rels=32
            nb_of_gt_rels = min(nb_of_gt_rels_threshold, nb_of_rels)
            nb_of_bg_rels = self.sample_rels - nb_of_gt_rels

            if nb_of_gt_rels < nb_of_rels:
                annot['annot'] = random.sample(annot['annot'], k=nb_of_gt_rels)
            
            nb_of_objects = len(annot['objects'])
            patient = 0
            while len(annot['annot']) < self.sample_rels:
                if patient > 10 or nb_of_objects < 2:
                    break
                sub_obj = random.sample(range(nb_of_objects), k=2)
                sub_id, obj_id = sub_obj[0], sub_obj[1]
                if any(rel['sub_id'] == sub_id and rel['obj_id'] == obj_id for rel in annot['annot']):
                    patient += 1
                    continue
                annot['annot'].append({'sub_id':sub_id, 'predicate':0, 'obj_id':obj_id})
                patient = 0
            random.shuffle(annot['annot'])

            nb_of_rels = len(annot['annot']) # self.sample_rels

        boxes = torch.zeros((nb_of_rels, 4, 4))
        spo_ids = torch.zeros((nb_of_rels, 20), dtype=torch.long)
        spo_lens = torch.zeros((nb_of_rels, 2), dtype=torch.long)
        im_info = torch.tensor([[iw, ih, 1.0, 1.0] for _ in range(nb_of_rels)])

        if self.all_proposals_test and self.split != 'train' and annot['rels_cand'] is not None: # testing
            labels = torch.zeros((len(annot['annot']), 1), dtype=torch.long)
            labels_so_ids = torch.zeros((len(annot['annot']), 2), dtype=torch.long)
            subj_obj_classes = torch.zeros((len(annot['rels_cand']), 2), dtype=torch.long)

            rels_cand = annot['rels_cand']
            for i, (sub_id, obj_id) in enumerate(rels_cand):
                subj = annot['objects'][sub_id]
                obj = annot['objects'][obj_id]

                subj_bbox = np.asarray(subj['bbox'], dtype=np.float32).reshape(1, -1)
                union_bbox = np.asarray(self._getUnionBBox(np.array(subj['bbox'])[[1,3,0,2]], np.array(obj['bbox'])[[1,3,0,2]], ih, iw), dtype=np.float32) # [x0,y0,w,h] -> [y0, y1(h), x0, x1(w)]
                union_bbox = union_bbox[[2,0,3,1]].reshape(1, -1) # [y0, y1(h), x0, x1(w)] -> [x0,y0,w,h]
                obj_bbox = np.asarray(obj['bbox'], dtype=np.float32).reshape(1, -1)
                boxes[i] = torch.as_tensor(np.concatenate((full_img_bbox, subj_bbox, union_bbox, obj_bbox)))

                subject_token = self.tokenizer.tokenize(self.cfg.OBJECT_CATEGORIES[subj['class']-1])
                subject_id = self.tokenizer.convert_tokens_to_ids(subject_token)
                object_token = self.tokenizer.tokenize(self.cfg.OBJECT_CATEGORIES[obj['class']-1])
                object_id = self.tokenizer.convert_tokens_to_ids(object_token)

                spo_id_len = len(subject_id + object_id)
                spo_ids[i, :spo_id_len] = torch.tensor(subject_id + object_id)
                spo_lens[i] = torch.tensor([len(subject_id), len(object_id)])

                subj_obj_classes[i][0], subj_obj_classes[i][1] = (subj['class']-1, obj['class']-1)
            
            for i, ann in enumerate(annot['annot']):
                labels[i] = ann['predicate'] # if add -1 to back to no '__background__'
                labels_so_ids[i][0], labels_so_ids[i][1] = (ann['sub_id'], ann['obj_id'])

            max_len = max(torch.sum(spo_lens, dim=1)).item()
            spo_ids = spo_ids[:, :max_len]

            flipped = False
            if self.transform is not None:
                img, boxes, _, labels, im_info, flipped = self.transform(img, boxes, None, labels, im_info, flipped)
                
            datum = {
                'img': img,
                'im_info': im_info,
                'boxes': boxes,
                'labels': labels,
                'spo_ids': spo_ids,
                'spo_lens': spo_lens,
                'img_path': annot['img_path'],
                'rels_cand': rels_cand,
                'labels_so_ids': labels_so_ids,
                'subj_obj_classes': subj_obj_classes,
            }
        else: # training
            labels = torch.zeros((len(annot['annot']), 1), dtype=torch.long)

            for i, ann in enumerate(annot['annot']):
                subj = annot['objects'][ann['sub_id']]
                obj = annot['objects'][ann['obj_id']]

                subj_bbox = np.asarray(subj['bbox'], dtype=np.float32).reshape(1, -1)
                union_bbox = np.asarray(self._getUnionBBox(np.array(subj['bbox'])[[1,3,0,2]], np.array(obj['bbox'])[[1,3,0,2]], ih, iw), dtype=np.float32) # [x0,y0,w,h] -> [y0, y1(h), x0, x1(w)]
                union_bbox = union_bbox[[2,0,3,1]].reshape(1, -1) # [y0, y1(h), x0, x1(w)] -> [x0,y0,w,h]
                obj_bbox = np.asarray(obj['bbox'], dtype=np.float32).reshape(1, -1)
                boxes[i] = torch.as_tensor(np.concatenate((full_img_bbox, subj_bbox, union_bbox, obj_bbox)))

                subject_token = self.tokenizer.tokenize(self.cfg.OBJECT_CATEGORIES[subj['class']-1])
                subject_id = self.tokenizer.convert_tokens_to_ids(subject_token)
                object_token = self.tokenizer.tokenize(self.cfg.OBJECT_CATEGORIES[obj['class']-1])
                object_id = self.tokenizer.convert_tokens_to_ids(object_token)

                spo_id_len = len(subject_id + object_id)
                spo_ids[i, :spo_id_len] = torch.tensor(subject_id + object_id)
                spo_lens[i] = torch.tensor([len(subject_id), len(object_id)])

                labels[i] = ann['predicate'] # if add -1 to back to no '__background__'
            
            max_len = max(torch.sum(spo_lens, dim=1)).item()
            spo_ids = spo_ids[:, :max_len]

            flipped = False
            if self.transform is not None:
                img, boxes, _, labels, im_info, flipped = self.transform(img, boxes, None, labels, im_info, flipped)

            datum = {
                'img': img,
                'im_info': im_info,
                'boxes': boxes,
                'labels': labels,
                'spo_ids': spo_ids,
                'spo_lens': spo_lens,
                'img_path': annot['img_path'],
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
