import torch
from common.utils.clip_pad import *


class BatchCollator(object):
    def __init__(self, dataset, append_ind=False):
        self.dataset = dataset
        self.data_names = self.dataset.data_names
        self.append_ind = append_ind

    def __call__(self, batch):
        if not isinstance(batch, list):
            batch = list(batch)

        if batch[0][self.data_names.index('img')] is not None:
            max_shape = tuple(max(s) for s in zip(*[data[self.data_names.index('img')].shape for data in batch]))
            image_none = False
        else:
            image_none = True
        # max_boxes = max([data[self.dcata_names.index('boxes')].shape[0] for data in batch])
        max_spo_ids_length = max([len(data[self.data_names.index('spo_ids')]) for data in batch])

        for i, ibatch in enumerate(batch):
            out = {}

            if image_none:
                out['img'] = None
            else:
                image = ibatch[self.data_names.index('img')]
                out['img'] = clip_pad_images(image, max_shape, pad=0)

            # boxes = ibatch[self.data_names.index('boxes')]
            # out['boxes'] = clip_pad_boxes(boxes, max_boxes, pad=-2)

            spo_ids = ibatch[self.data_names.index('spo_ids')]
            out['spo_ids'] = clip_pad_1d(spo_ids, max_spo_ids_length, pad=0)

            other_names = [data_name for data_name in self.data_names if data_name not in out]
            for name in other_names:
                # print('ibatch[self.data_names.index(name)]:', ibatch[self.data_names.index(name)])
                if isinstance(ibatch[self.data_names.index(name)], str):
                    out[name] = ibatch[self.data_names.index(name)]
                else:
                    out[name] = torch.as_tensor(ibatch[self.data_names.index(name)])

            batch[i] = tuple(out[data_name] for data_name in self.data_names)
            if self.append_ind:
                batch[i] += (torch.tensor(i, dtype=torch.int64),)

        out_tuple = ()
        for items in zip(*batch):
            if items[0] is None:
                out_tuple += (None,)
            elif isinstance(items[0], str):
                out_tuple += (list(items), )
            else:
                out_tuple += (torch.stack(tuple(items), dim=0), )

        return out_tuple