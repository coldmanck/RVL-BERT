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
from spasen.data.build import make_dataloader
from spasen.modules import *

from collections import defaultdict
import pickle

@torch.no_grad()
def test_net(args, config, ckpt_path=None, save_path=None, save_name=None):
    print('test net...')
    pprint.pprint(args)
    pprint.pprint(config)
    device_ids = [int(d) for d in config.GPUS.split(',')]
    # os.environ['CUDA_VISIBLE_DEVICES'] = config.GPUS

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if ckpt_path is None:
        _, train_output_path = create_logger(config.OUTPUT_PATH, args.cfg, config.DATASET.TRAIN_IMAGE_SET,
                                             split='train')
        model_prefix = os.path.join(train_output_path, config.MODEL_PREFIX)
        ckpt_path = '{}-best.model'.format(model_prefix)
        print('Use best checkpoint {}...'.format(ckpt_path))
    if save_path is None:
        logger, test_output_path = create_logger(config.OUTPUT_PATH, args.cfg, config.DATASET.TEST_IMAGE_SET,
                                                 split='test')
        save_path = test_output_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    shutil.copy2(ckpt_path,
                 os.path.join(save_path, '{}_test_ckpt_{}.model'.format(config.MODEL_PREFIX, config.DATASET.TASK)))

    # get network
    model = eval(config.MODULE)(config)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
    else:
        torch.cuda.set_device(device_ids[0])
        model = model.cuda()
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    smart_load_model_state_dict(model, checkpoint['state_dict'])

    # loader
    test_loader = make_dataloader(config, mode='test', distributed=False)

    split = args.split + 'id' if args.split == 'val' else args.split # 'val' -> 'valid'

    # test
    if config.TEST.EXCL_LEFT_RIGHT:
        precompute_test_cache = f'{args.log_dir}/pred_{split}_{ckpt_path[-10:-6]}_excl-left-right.pickle'
    else:
        precompute_test_cache = f'{args.log_dir}/pred_{split}_{ckpt_path[-10:-6]}.pickle'
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)
    pred_file = precompute_test_cache
    if not os.path.exists(precompute_test_cache):
        _ids = []
        losses = []
        predictions = []
        model.eval()
        
        if args.visualize_mask: # For mask visualization purpose
            save_dir = 'heatmap/spasen'
            if not os.path.isdir(save_dir):
                # os.mkdir(save_dir)
                os.makedirs(save_dir)

        for nbatch, batch in zip(trange(len(test_loader)), test_loader):
            _ids.extend(batch[0]) # the first input element is _id

            batch = to_cuda(batch)
            output = model(*batch)
            
            predictions.append(output['prediction'])
            losses.append(output['ans_loss'].item())

            if args.visualize_mask: # For mask visualization purpose
                mask = output['spo_fused_masks'].cpu() # torch.Size([8, 3, 14, 14])
                subj_name = output['subj_name'] # list of 8 strs
                obj_name = output['obj_name'] # list of 8 strs
                pred_name = output['pred_name'] # list of 8 strs
                im_path = output['im_path'] # list of 8 img urls

                for i in range(mask.shape[0]):
                    img, dataset = read_img(im_path[i], config.IMAGEPATH)
                    img_name = dataset + '-' + im_path[i].split('/')[-1]
                    show_cam_on_image(img, mask[i], img_name, subj_name[i], obj_name[i], pred_name[i], save_dir)

        predictions = [v.item() for v in torch.cat(predictions)]
        loss = sum(losses) / len(losses)
        pickle.dump((_ids, predictions, loss), open(pred_file, 'wb'))

    accs, loss = accuracies(pred_file, 'data/spasen/annotations.json', split)

    return accs, loss

def read_img(url, imagepath):
    if url.startswith('http'):  # flickr
        dataset = 'flickr'
    else:  # nyu
        dataset = 'nyu'
    filename = os.path.join(imagepath, dataset, url.split('/')[-1])
    img = cv2.imread(filename, 1)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    return img, dataset

def accuracies(pred_file, gt_file, split):
    gt = {}
    data = json.load(open(gt_file))
    for img in data:
        if img['split'] != split:
            continue
        for annot in img['annotations']:
            annot['url'] = img['url']
            gt[annot['_id']] = annot

    cnts = defaultdict(lambda : {'correct': 0, 'incorrect': 0})
    _ids, predictions, loss = pickle.load(open(pred_file, 'rb'))
    for _id, prediction in zip(_ids, predictions):
        predicate = gt[_id]['predicate']
        if (prediction > 0.) == gt[_id]['label']:
            cnts[predicate]['correct'] += 1
            cnts['overall']['correct'] += 1
        else:
            cnts[predicate]['incorrect'] += 1
            cnts['overall']['incorrect'] += 1

    accs = {}
    for k, v in cnts.items():
        accs[k] = 100. * v['correct'] / (v['correct'] + v['incorrect'])
    return accs, loss

import cv2
def show_cam_on_image(img, mask, img_name, subj_name, obj_name, pred_name, save_dir):
    for i in range(mask.shape[0]):
        mask2 = F.interpolate(mask[i].unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear').squeeze()
        heatmap = cv2.applyColorMap(np.uint8(255 * mask2), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)

        save_heatmap_name = save_name = f'{img_name}-{subj_name}-{pred_name}-{obj_name}-'
        if i == 0:
            save_name += 'subject.jpg'
            save_heatmap_name += 'subject-heatmap.jpg'
        elif i == 1:
            save_name += 'predicate.jpg'
            save_heatmap_name += 'predicate-heatmap.jpg'
        else:
            save_name += 'object.jpg'
            save_heatmap_name += 'object-heatmap.jpg'
        cv2.imwrite(os.path.join(save_dir, save_name), np.uint8(255 * cam))
        cv2.imwrite(os.path.join(save_dir, save_heatmap_name), np.uint8(255 * heatmap))