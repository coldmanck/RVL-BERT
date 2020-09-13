import torch
from .eval_metric import EvalMetric


class LossLogger(EvalMetric):
    def __init__(self, output_name, display_name=None,
                 allreduce=False, num_replicas=1):
        self.output_name = output_name
        if display_name is None:
            display_name = output_name
        super(LossLogger, self).__init__(display_name, allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            if self.output_name in outputs:
                self.sum_metric += float(outputs[self.output_name].mean().item())
            self.num_inst += 1


class SoftAccuracy(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(SoftAccuracy, self).__init__('SoftAcc', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            cls_logits = outputs['label_logits']
            label = outputs['label']
            bs, num_classes = cls_logits.shape
            batch_inds = torch.arange(bs, device=cls_logits.device)
            self.sum_metric += float(label[batch_inds, cls_logits.argmax(1)].sum().item())
            self.num_inst += cls_logits.shape[0]


class Accuracy(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1):
        super(Accuracy, self).__init__('Acc', allreduce, num_replicas)

    def update(self, outputs):
        with torch.no_grad():
            # _filter = outputs['label'] != -1
            # cls_logits = outputs['label_logits'][_filter]
            # label = outputs['label'][_filter]
            cls_logits = outputs['label_logits'].argmax(1)
            label = outputs['label'].squeeze()
            # import pdb; pdb.set_trace()
            self.sum_metric += float((cls_logits == label).sum().item())
            self.num_inst += cls_logits.shape[0]
            # print('Accuracy this batch:', float((torch.round(cls_logits).long() == label).sum().item()) / cls_logits.shape[0])

class Recall50(EvalMetric):
    def __init__(self, allreduce=False, num_replicas=1, remove_bg=False):
        super(Recall50, self).__init__('Recall50', allreduce, num_replicas)
        self.remove_bg = remove_bg

    def update(self, outputs):
        with torch.no_grad():
            labels = outputs['label']
            labels_so_ids = outputs['labels_so_ids']
            logits = outputs['label_logits']
            if self.remove_bg:
                # do NOT consider background prediction: zero out bg prob
                logits[:, 0] = 0
            rels_cand = outputs['rels_cand']

            pred = logits.argmax(dim=1)            

            logits = logits[pred != 0]
            rels_cand = rels_cand[pred != 0]
            pred = pred[pred != 0]

            pred_conf = logits[[i for i in range(pred.shape[0])], [pred.cpu().tolist()]].squeeze()
            if len(pred_conf.shape) == 0:
                pred_conf = pred_conf.unsqueeze(0)

            # pred_conf = pred_conf.argsort(descending=True).cpu().tolist()
            # THIS SECTION IS FOR REOMVING NONE VALUES!
            values, indices = pred_conf.sort()
            nb_of_non_nan_values = len(values[values==values])
            pred_conf = indices[:nb_of_non_nan_values].cpu().tolist()
            pred_conf.reverse()
            
            pred_conf = pred_conf[:50] # TO compute R@50
            pred = pred[pred_conf].cpu().tolist()
            rels_cand = rels_cand[pred_conf].cpu().tolist()
            rels_cand_pred = {tuple(k): v for k, v in zip(rels_cand, pred)}
            
            correct = 0
            for idx, rel in enumerate(labels_so_ids.cpu().tolist()):
                if not tuple(rel) in rels_cand_pred:
                    continue
                if rels_cand_pred[tuple(rel)] == labels[idx].cpu().item():
                    correct += 1

            self.sum_metric += float(correct)
            self.num_inst += labels_so_ids.shape[0]
