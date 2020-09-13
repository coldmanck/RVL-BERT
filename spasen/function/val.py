from collections import namedtuple
import torch
from common.trainer import to_cuda
from tqdm import tqdm

@torch.no_grad()
def do_validation(net, val_loader, metrics, label_index_in_batch):
    net.eval()
    metrics.reset()
    
    for nbatch, batch in tqdm(enumerate(val_loader)):
        batch = to_cuda(batch)
        # label = batch[label_index_in_batch]
        # datas = [batch[i] for i in range(len(batch)) if i != label_index_in_batch % len(batch)]
        datas = [batch[i] for i in range(len(batch))]
        outputs = net(*datas)
        # outputs.update({'label': label.long()})
        metrics.update(outputs)