import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from common.backbone.resnet.resnet import *
from common.backbone.resnet.resnet import Bottleneck, BasicBlock
from common.backbone.resnet.resnet import model_urls

from common.lib.roi_pooling.roi_pool import ROIPool
from common.lib.roi_pooling.roi_align import ROIAlign
from common.utils.flatten import Flattener
from common.utils.pad_sequence import pad_sequence
from common.utils.bbox import coordinate_embeddings


class FastRCNN(nn.Module):
    def __init__(self, config, average_pool=True, final_dim=768, enable_cnn_reg_loss=False):
        """
        :param config:
        :param average_pool: whether or not to average pool the representations
        :param final_dim:
        :param is_train:
        """
        super(FastRCNN, self).__init__()
        self.config = config
        self.average_pool = average_pool
        self.enable_cnn_reg_loss = enable_cnn_reg_loss
        self.final_dim = final_dim
        self.image_feat_precomputed = config.NETWORK.IMAGE_FEAT_PRECOMPUTED
        if self.image_feat_precomputed:
            if config.NETWORK.IMAGE_SEMANTIC:
                self.object_embed = torch.nn.Embedding(num_embeddings=81, embedding_dim=128)
            else:
                self.object_embed = None
        else:
            self.stride_in_1x1 = config.NETWORK.IMAGE_STRIDE_IN_1x1
            self.c5_dilated = config.NETWORK.IMAGE_C5_DILATED
            self.num_layers = config.NETWORK.IMAGE_NUM_LAYERS
            # import pdb; pdb.set_trace()
            if config.NETWORK.IMAGE_PRETRAINED != '':
                self.pretrained_model_path = '{}-{:04d}.model'.format(config.NETWORK.IMAGE_PRETRAINED, 
                                                                      config.NETWORK.IMAGE_PRETRAINED_EPOCH)
            else:
                self.pretrained_model_path = None
            self.output_conv5 = config.NETWORK.OUTPUT_CONV5
            if self.num_layers == 18:
                self.backbone = resnet18(pretrained=True, pretrained_model_path=self.pretrained_model_path,
                                         expose_stages=[4])
                block = BasicBlock
            elif self.num_layers == 34:
                self.backbone = resnet34(pretrained=True, pretrained_model_path=self.pretrained_model_path,
                                         expose_stages=[4])
                block = BasicBlock
            elif self.num_layers == 50:
                self.backbone = resnet50(pretrained=True, pretrained_model_path=self.pretrained_model_path,
                                         expose_stages=[4], stride_in_1x1=self.stride_in_1x1)
                block = Bottleneck
            elif self.num_layers == 101:
                self.backbone = resnet101(pretrained=True, pretrained_model_path=self.pretrained_model_path,
                                          expose_stages=[4], stride_in_1x1=self.stride_in_1x1)
                block = Bottleneck
            elif self.num_layers == 152:
                self.backbone = resnet152(pretrained=True, pretrained_model_path=self.pretrained_model_path,
                                          expose_stages=[4], stride_in_1x1=self.stride_in_1x1)
                block = Bottleneck
            else:
                raise NotImplemented

            output_size = (14, 14)
            self.roi_align = ROIAlign(output_size=output_size, spatial_scale=1.0 / 16)

            if config.NETWORK.IMAGE_SEMANTIC:
                self.object_embed = torch.nn.Embedding(num_embeddings=81, embedding_dim=128)
            else:
                self.object_embed = None
                self.mask_upsample = None

            self.roi_head_feature_extractor = self.backbone._make_layer(block=block, planes=512, blocks=3,
                                                                        stride=2 if not self.c5_dilated else 1,
                                                                        dilation=1 if not self.c5_dilated else 2,
                                                                        stride_in_1x1=self.stride_in_1x1)

            if average_pool:
                self.head = torch.nn.Sequential(
                    self.roi_head_feature_extractor,
                    nn.AvgPool2d(7 if not self.c5_dilated else 14, stride=1),
                    Flattener()
                )
            else:
                self.head = self.roi_head_feature_extractor

            if config.NETWORK.IMAGE_FROZEN_BN:
                for module in self.roi_head_feature_extractor.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        for param in module.parameters():
                            param.requires_grad = False

            frozen_stages = config.NETWORK.IMAGE_FROZEN_BACKBONE_STAGES
            if 5 in frozen_stages:
                for p in self.roi_head_feature_extractor.parameters():
                    p.requires_grad = False
                frozen_stages = [stage for stage in frozen_stages if stage != 5]
            self.backbone.frozen_parameters(frozen_stages=frozen_stages,
                                            frozen_bn=config.NETWORK.IMAGE_FROZEN_BN)

            if self.enable_cnn_reg_loss:
                self.regularizing_predictor = torch.nn.Linear(2048, 81)

        self.obj_downsample = torch.nn.Sequential(
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(2 * 2048 + (128 if config.NETWORK.IMAGE_SEMANTIC else 0), final_dim),
            torch.nn.ReLU(inplace=True),
        )

        self.max_nb_boxes = 0

    def init_weight(self):
        if not self.image_feat_precomputed:
            if self.pretrained_model_path is None:
                pretrained_model = model_zoo.load_url(model_urls['resnet{}'.format(self.num_layers)])
            else:
                pretrained_model = torch.load(self.pretrained_model_path, map_location=lambda storage, loc: storage)
            roi_head_feat_dict = {k[len('layer4.'):]: v for k, v in pretrained_model.items() if k.startswith('layer4.')}
            self.roi_head_feature_extractor.load_state_dict(roi_head_feat_dict)
            if self.output_conv5:
                self.conv5.load_state_dict(roi_head_feat_dict)

    def bn_eval(self):
        if not self.image_feat_precomputed:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()

    def forward(self, images, boxes, box_mask, im_info, classes=None, segms=None, mvrc_ops=None, mask_visual_embed=None, copy_images=False, union_boxes=None, rels_cand=None):
        """
        :param images: [batch_size, 3, im_height, im_width]
        :param boxes: [batch_size, max_num_objects, 4] Padded boxes
        :param box_mask: [batch_size, max_num_objects] Mask for whether or not each box is OK
        :return: object reps [batch_size, max_num_objects, dim]
        """

        box_inds = box_mask.nonzero()
        obj_labels = classes[box_inds[:, 0], box_inds[:, 1]].type(torch.long) if classes is not None else None
        assert box_inds.shape[0] > 0

        try:
            if copy_images:
                images = images[0].unsqueeze(0)
            img_feats = self.backbone(images)
            if copy_images:
                img_feats['body4'] = torch.cat([img_feats['body4']] * box_inds.shape[0])
        except:
            import pdb; pdb.set_trace()
        
        if union_boxes is not None: # object pairing after roi pooling
            n_boxes, n_union_boxes = len(boxes), len(union_boxes) # n_boxes includes the first one as full image bbox
            rois = torch.cat((boxes, union_boxes))
            rois = torch.cat((
                torch.zeros_like(rois)[:, 0].view(-1, 1),
                rois
            ), 1)
        else:
            rois = torch.cat((
                box_inds[:, 0, None].type(boxes.dtype),
                boxes[box_inds[:, 0], box_inds[:, 1]],
            ), 1)
        
        roi_align_res = self.roi_align(img_feats['body4'], rois).type(images.dtype)

        if segms is not None:
            pool_layers = self.head[1:]
            post_roialign = self.roi_head_feature_extractor(roi_align_res)
            post_roialign = post_roialign * segms[box_inds[:, 0], None, box_inds[:, 1]].to(dtype=post_roialign.dtype)
            for _layer in pool_layers:
                post_roialign = _layer(post_roialign)
        else:
            # post_roialign = self.head(roi_align_res)
            if self.config.TRAIN.DEBUG and union_boxes is not None:
                print(f'cur max_nb_boxes: {self.max_nb_boxes}, n_boxes: {n_boxes}, n_union_boxes: {n_union_boxes}, rois.shape: {rois.shape}')
                if self.max_nb_boxes < n_boxes:
                    print(f'Update self.max_nb_boxes from {self.max_nb_boxes} to {n_boxes}')
                    self.max_nb_boxes = n_boxes
            try:
                post_roialign = self.head[0](roi_align_res)
                post_roialign_raw = post_roialign.clone().detach() # torch.tensor(post_roialign)
                post_roialign = self.head[1:](post_roialign)
            except:
                import pdb; pdb.set_trace()

        if union_boxes is not None:
            '''
            (Pdb) post_roialign.shape
            torch.Size([81, 2048])
            (Pdb) post_roialign_raw.shape
            torch.Size([81, 2048, 14, 14])
            '''
            full_img_feat = post_roialign[0]
            post_roialign_boxes = post_roialign[1:n_boxes]
            post_roialign_union_boxes = post_roialign[n_boxes:]

            full_img_feat_raw = post_roialign_raw[0]
            post_roialign_raw_boxes = post_roialign_raw[1:n_boxes]
            post_roialign_raw_union_boxes = post_roialign_raw[n_boxes:]
            n_boxes -= 1 # get rid of the count of the first full img bbox
            
            device = boxes.device
            try:
                new_boxes = torch.zeros([rels_cand.shape[0], 4, 4], device=device)
                post_roialign = torch.zeros((box_inds.shape[0], 2048), device=device)
                post_roialign_raw = torch.zeros((box_inds.shape[0], 2048, 14, 14), device=device)
            except:
                import pdb; pdb.set_trace()
            
            for i, (sub_id, obj_id) in enumerate(rels_cand):
                post_roialign[i*4] = full_img_feat
                post_roialign[i*4 + 1] = post_roialign_boxes[sub_id]
                post_roialign[i*4 + 2] = post_roialign_union_boxes[i]
                post_roialign[i*4 + 3] = post_roialign_boxes[obj_id]

                post_roialign_raw[i*4] = full_img_feat_raw
                post_roialign_raw[i*4 + 1] = post_roialign_raw_boxes[sub_id]
                post_roialign_raw[i*4 + 2] = post_roialign_raw_union_boxes[i]
                post_roialign_raw[i*4 + 3] = post_roialign_raw_boxes[obj_id]

                new_boxes[i][0] = boxes[0]
                new_boxes[i][1] = boxes[sub_id + 1]
                new_boxes[i][2] = union_boxes[i]
                new_boxes[i][3] = boxes[obj_id + 1]
            
            boxes = new_boxes

        '''
        (Pdb) boxes.shape
        torch.Size([32, 4, 4])
        (Pdb) post_roialign.shape
        torch.Size([128, 2048])
        (Pdb) post_roialign_raw.shape
        torch.Size([128, 2048, 14, 14])
        '''
        if self.config.TRAIN.DEBUG:
            pass # import pdb; pdb.set_trace() # pass

        # Add some regularization, encouraging the model to keep giving decent enough predictions
        if self.enable_cnn_reg_loss: # False
                obj_logits = self.regularizing_predictor(post_roialign)
                cnn_regularization = F.cross_entropy(obj_logits, obj_labels)[None]

        feats_to_downsample = post_roialign if (self.object_embed is None or obj_labels is None) else \
            torch.cat((post_roialign, self.object_embed(obj_labels)), -1)
        if mvrc_ops is not None and mask_visual_embed is not None: # False
            _to_masked = (mvrc_ops == 1)[box_inds[:, 0], box_inds[:, 1]]
            feats_to_downsample[_to_masked] = mask_visual_embed
        try:
            coord_embed = coordinate_embeddings(
                torch.cat((boxes[box_inds[:, 0], box_inds[:, 1]], im_info[box_inds[:, 0], :2]), 1),
                256
            )
        except:
            import pdb; pdb.set_trace()
        
        feats_to_downsample = torch.cat((coord_embed.view((coord_embed.shape[0], -1)), feats_to_downsample), -1)
        final_feats = self.obj_downsample(feats_to_downsample)

        # Reshape into a padded sequence - this is expensive and annoying but easier to implement and debug...
        if union_boxes is None:
            obj_reps = pad_sequence(final_feats, box_mask.sum(1).tolist())
            post_roialign = pad_sequence(post_roialign, box_mask.sum(1).tolist())
            post_roialign_raw = pad_sequence(post_roialign_raw, box_mask.sum(1).tolist())
        else:
            obj_reps = final_feats.view(-1, 4, final_feats.shape[1])
            post_roialign = post_roialign.view(-1, 4, post_roialign.shape[1])
            post_roialign_raw = post_roialign_raw.view(-1, 4, post_roialign_raw.shape[1])

        # DataParallel compatibility
        if union_boxes is None: # off for VG exps
            obj_reps_padded = obj_reps.new_zeros((obj_reps.shape[0], boxes.shape[1], obj_reps.shape[2]))
            obj_reps_padded[:, :obj_reps.shape[1]] = obj_reps
            obj_reps = obj_reps_padded

            post_roialign_padded = post_roialign.new_zeros((post_roialign.shape[0], boxes.shape[1], post_roialign.shape[2]))
            post_roialign_padded[:, :post_roialign.shape[1]] = post_roialign
            post_roialign = post_roialign_padded

            post_roialign_raw_padded = post_roialign.new_zeros((post_roialign_raw.shape[0], boxes.shape[1], post_roialign_raw.shape[2], post_roialign_raw.shape[3], post_roialign_raw.shape[4]))
            post_roialign_raw_padded[:, :post_roialign_raw.shape[1]] = post_roialign_raw
            post_roialign_raw = post_roialign_raw_padded

        # Output
        output_dict = {
            'obj_reps_raw': post_roialign,
            'obj_reps': obj_reps,
            'obj_reps_rawraw': post_roialign_raw,
        }

        if (not self.image_feat_precomputed) and self.enable_cnn_reg_loss:
            output_dict.update({'obj_logits': obj_logits,
                                'obj_labels': obj_labels,
                                'cnn_regularization_loss': cnn_regularization})

        if (not self.image_feat_precomputed) and self.output_conv5:
            image_feature = self.img_head(img_feats['body4'])
            output_dict['image_feature'] = image_feature

        if union_boxes is not None:
            return output_dict, boxes
        else:
            return output_dict
