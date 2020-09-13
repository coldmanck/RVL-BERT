import _init_paths
import os
import argparse
from copy import deepcopy

from spasen.function.config import config, update_config
from spasen.function.test import test_net


def parse_args():
    parser = argparse.ArgumentParser('Get Test Result of VQA Network')
    parser.add_argument('--cfg', type=str, help='path to answer net config yaml')
    parser.add_argument('--ckpt', type=str, help='path to checkpoint of answer net')
    parser.add_argument('--bs', type=int)
    parser.add_argument('--gpus', type=int, nargs='+')
    parser.add_argument('--model-dir', type=str, help='root path to store checkpoint')
    parser.add_argument('--log-dir', type=str, help='root path to store result')
    parser.add_argument('--result-path', type=str, help='path to store test result file.')
    parser.add_argument('--result-name', type=str)
    parser.add_argument('--split', default='test')
    parser.add_argument('--excl-left-right', action='store_true', help='exclude `to the left of` and `to the right of` rels during testing')
    parser.add_argument('--visualize_mask', action='store_true')

    args = parser.parse_args()
    # import pdb; pdb.set_trace()

    if args.cfg is not None:
        update_config(args.cfg)
    if args.bs is not None:
        config.TEST.BATCH_IMAGES = args.bs
    if args.gpus is not None:
        config.GPUS = ','.join([str(gpu) for gpu in args.gpus])
    if args.split is not None:
        config.DATASET.TEST_IMAGE_SET = args.split
    if args.model_dir is not None:
        config.OUTPUT_PATH = os.path.join(args.model_dir, config.OUTPUT_PATH)
    if args.excl_left_right:
        config.TEST.EXCL_LEFT_RIGHT = True

    return args, config


def main():
    args, config = parse_args()

    accs, loss = test_net(args, config, ckpt_path=args.ckpt, save_path=args.result_path, 
                          save_name=args.result_name)

    print(f'\n\t{args.split} loss = {loss:.4f}')
    print('\ttesting accuracy = %.3f' % accs['overall'])
    # writer.add_scalar('Val/Loss', loss, epoch)
    # writer.add_scalar('Val/Accuracy', accs['overall'], epoch)
    for predi in accs:
        if predi != 'overall':
            print('\t\t%s: %.3f' % (predi, accs[predi]))
            # writer.add_scalar(f'Val/Accuracy_{predi}', accs[predi], epoch)


if __name__ == '__main__':
    main()
