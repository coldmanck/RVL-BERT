import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from external.pytorch_pretrained_bert import BertTokenizer
from external.pytorch_pretrained_bert.modeling import BertPredictionHeadTransform
from common.module import Module
from common.fast_rcnn import FastRCNN
from common.visual_linguistic_bert import VisualLinguisticBert
from .simple_spatial_model import SimpleSpatialModel
import numpy as np

BERT_WEIGHTS_NAME = 'pytorch_model.bin'
import cv2

class ResNetVLBERT(Module):
    def __init__(self, config):

        super(ResNetVLBERT, self).__init__(config)

        self.predict_on_cls = config.NETWORK.VLBERT.predict_on_cls # make prediction on [CLS]?

        self.enable_cnn_reg_loss = config.NETWORK.ENABLE_CNN_REG_LOSS
        if not config.NETWORK.BLIND:
            self.image_feature_extractor = FastRCNN(config,
                                                    average_pool=True,
                                                    final_dim=config.NETWORK.IMAGE_FINAL_DIM,
                                                    enable_cnn_reg_loss=self.enable_cnn_reg_loss)
            if config.NETWORK.VLBERT.object_word_embed_mode == 1:
                self.object_linguistic_embeddings = nn.Embedding(81, config.NETWORK.VLBERT.hidden_size)
            elif config.NETWORK.VLBERT.object_word_embed_mode == 2: # default: class-agnostic
                self.object_linguistic_embeddings = nn.Embedding(1, config.NETWORK.VLBERT.hidden_size)
            elif config.NETWORK.VLBERT.object_word_embed_mode == 3:
                self.object_linguistic_embeddings = None
            else:
                raise NotImplementedError
        self.image_feature_bn_eval = config.NETWORK.IMAGE_FROZEN_BN

        self.tokenizer = BertTokenizer.from_pretrained(config.NETWORK.BERT_MODEL_NAME)

        language_pretrained_model_path = None
        if config.NETWORK.BERT_PRETRAINED != '':
            language_pretrained_model_path = '{}-{:04d}.model'.format(config.NETWORK.BERT_PRETRAINED,
                                                                      config.NETWORK.BERT_PRETRAINED_EPOCH)
        elif os.path.isdir(config.NETWORK.BERT_MODEL_NAME):
            weight_path = os.path.join(config.NETWORK.BERT_MODEL_NAME, BERT_WEIGHTS_NAME)
            if os.path.isfile(weight_path):
                language_pretrained_model_path = weight_path
        self.language_pretrained_model_path = language_pretrained_model_path
        if language_pretrained_model_path is None:
            print("Warning: no pretrained language model found, training from scratch!!!")

        self.vlbert = VisualLinguisticBert(config.NETWORK.VLBERT,
                                         language_pretrained_model_path=language_pretrained_model_path) 

        dim = config.NETWORK.VLBERT.hidden_size
        if config.NETWORK.CLASSIFIER_TYPE == "2fc":
            self.final_mlp = torch.nn.Sequential(
                torch.nn.Dropout(config.NETWORK.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(dim, config.NETWORK.CLASSIFIER_HIDDEN_SIZE),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(config.NETWORK.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(config.NETWORK.CLASSIFIER_HIDDEN_SIZE, config.DATASET.ANSWER_VOCAB_SIZE),
            )
        elif config.NETWORK.CLASSIFIER_TYPE == "1fc":
            self.final_mlp = torch.nn.Sequential(
                torch.nn.Dropout(config.NETWORK.CLASSIFIER_DROPOUT, inplace=False),
                torch.nn.Linear(dim, config.DATASET.ANSWER_VOCAB_SIZE)
            )
        elif config.NETWORK.CLASSIFIER_TYPE == 'mlm':
            transform = BertPredictionHeadTransform(config.NETWORK.VLBERT)
            linear = nn.Linear(config.NETWORK.VLBERT.hidden_size, config.DATASET.ANSWER_VOCAB_SIZE)
            self.final_mlp = nn.Sequential(
                transform,
                nn.Dropout(config.NETWORK.CLASSIFIER_DROPOUT, inplace=False),
                linear
            )
        else:
            raise ValueError("Not support classifier type: {}!".format(config.NETWORK.CLASSIFIER_TYPE))

        self.use_spatial_model = False
        if config.NETWORK.USE_SPATIAL_MODEL:
            self.use_spatial_model = True
            # self.simple_spatial_model = SimpleSpatialModel(4, config.NETWORK.VLBERT.hidden_size, 9, config)
            
            self.use_coord_vector = False
            if config.NETWORK.USE_COORD_VECTOR:
                self.use_coord_vector = True
                self.loc_fcs = nn.Sequential(
                    nn.Linear(2*5+9, config.NETWORK.VLBERT.hidden_size),
                    nn.ReLU(True),
                    nn.Linear(config.NETWORK.VLBERT.hidden_size, config.NETWORK.VLBERT.hidden_size)
                )
            else:
                self.simple_spatial_model = SimpleSpatialModel(4, config.NETWORK.VLBERT.hidden_size, 9)

            self.spa_add = True if config.NETWORK.SPA_ADD else False
            self.spa_concat = True if config.NETWORK.SPA_CONCAT else False
            
            if self.spa_add:
                self.spa_feat_weight = 0.5
                if config.NETWORK.USE_SPA_WEIGHT:
                    self.spa_feat_weight = config.NETWORK.SPA_FEAT_WEIGHT
                self.spa_fusion_linear = nn.Linear(config.NETWORK.VLBERT.hidden_size, config.NETWORK.VLBERT.hidden_size)
            elif self.spa_concat:
                if self.use_coord_vector:
                    self.spa_fusion_linear = nn.Linear(config.NETWORK.VLBERT.hidden_size + config.NETWORK.VLBERT.hidden_size, config.NETWORK.VLBERT.hidden_size)
                else:
                    self.spa_fusion_linear = nn.Linear(config.NETWORK.VLBERT.hidden_size * 2, config.NETWORK.VLBERT.hidden_size)
            self.spa_linear = nn.Linear(config.NETWORK.VLBERT.hidden_size, config.NETWORK.VLBERT.hidden_size)
            self.dropout = nn.Dropout(config.NETWORK.CLASSIFIER_DROPOUT)

            self.spa_one_more_layer = config.NETWORK.SPA_ONE_MORE_LAYER
            if self.spa_one_more_layer:
                self.spa_linear_hidden = nn.Linear(config.NETWORK.VLBERT.hidden_size, config.NETWORK.VLBERT.hidden_size)

        self.enhanced_img_feature = False
        if config.NETWORK.VLBERT.ENHANCED_IMG_FEATURE:
            self.enhanced_img_feature = True
            self.mask_weight = config.NETWORK.VLBERT.mask_weight
            self.mask_loss_sum = config.NETWORK.VLBERT.mask_loss_sum
            self.mask_loss_mse = config.NETWORK.VLBERT.mask_loss_mse
            self.no_predicate = config.NETWORK.VLBERT.NO_PREDICATE

        self.all_proposals_test = False
        if config.DATASET.ALL_PROPOSALS_TEST:
            self.all_proposals_test = True

        self.use_uvtranse = False
        if config.NETWORK.USE_UVTRANSE:
            self.use_uvtranse = True
            self.union_vec_fc = nn.Linear(config.NETWORK.VLBERT.hidden_size, config.NETWORK.VLBERT.hidden_size)
            self.uvt_add = True if config.NETWORK.UVT_ADD else False
            self.uvt_concat = True if config.NETWORK.UVT_CONCAT else False
            if not (self.uvt_add ^ self.uvt_concat):
                assert False
            if self.uvt_add:
                self.uvt_feat_weight = config.NETWORK.UVT_FEAT_WEIGHT
                self.uvt_fusion_linear = nn.Linear(config.NETWORK.VLBERT.hidden_size, config.NETWORK.VLBERT.hidden_size)
            elif self.uvt_concat:
                self.uvt_fusion_linear = nn.Linear(config.NETWORK.VLBERT.hidden_size * 2, config.NETWORK.VLBERT.hidden_size)
            self.uvt_linear = nn.Linear(config.NETWORK.VLBERT.hidden_size, config.NETWORK.VLBERT.hidden_size)
            self.dropout_uvt = nn.Dropout(config.NETWORK.CLASSIFIER_DROPOUT)


        # init weights
        self.init_weight()

    def init_weight(self):
        # self.hm_out.weight.data.normal_(mean=0.0, std=0.02)
        # self.hm_out.bias.data.zero_()
        # self.hi_out.weight.data.normal_(mean=0.0, std=0.02)
        # self.hi_out.bias.data.zero_()
        self.image_feature_extractor.init_weight()
        if self.object_linguistic_embeddings is not None:
            self.object_linguistic_embeddings.weight.data.normal_(mean=0.0, std=0.02)
        for m in self.final_mlp.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)
        if self.config.NETWORK.CLASSIFIER_TYPE == 'mlm':
            language_pretrained = torch.load(self.language_pretrained_model_path)
            mlm_transform_state_dict = {}
            pretrain_keys = []
            for k, v in language_pretrained.items():
                if k.startswith('cls.predictions.transform.'):
                    pretrain_keys.append(k)
                    k_ = k[len('cls.predictions.transform.'):]
                    if 'gamma' in k_:
                        k_ = k_.replace('gamma', 'weight')
                    if 'beta' in k_:
                        k_ = k_.replace('beta', 'bias')
                    mlm_transform_state_dict[k_] = v
            print("loading pretrained classifier transform keys: {}.".format(pretrain_keys))
            self.final_mlp[0].load_state_dict(mlm_transform_state_dict)

    def train(self, mode=True):
        super(ResNetVLBERT, self).train(mode)
        # turn some frozen layers to eval mode
        if self.image_feature_bn_eval:
            self.image_feature_extractor.bn_eval()

    def _collect_obj_reps(self, span_tags, object_reps, spo_len):
        """
        Collect span-level object representations
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :return:
        """

        span_tags_fixed = torch.clamp(span_tags, min=0)  # In case there were masked values here
        row_id = span_tags_fixed.new_zeros(span_tags_fixed.shape)
        row_id_broadcaster = torch.arange(0, row_id.shape[0], step=1, device=row_id.device)[:, None]

        # Add extra diminsions to the row broadcaster so it matches row_id
        leading_dims = len(span_tags.shape) - 2
        for i in range(leading_dims):
            row_id_broadcaster = row_id_broadcaster[..., None]
        row_id += row_id_broadcaster
        
        if self.enhanced_img_feature:
            # for i in range(span_tags_fixed.shape[0]):
            #     span_tags_fixed[i, 1:1 + spo_len[i, 0]] = 1
            #     span_tags_fixed[i, 1 + spo_len[i, 0]:1 + spo_len[i, 0] + spo_len[i, 1]] = 2
            #     span_tags_fixed[i, 1 + spo_len[i, 0] + spo_len[i, 1]:1 + spo_len[i, 0] + spo_len[i, 1] + spo_len[i, 2]] = 3
            pass
        
        text_visual_embeddings = object_reps[row_id.view(-1), span_tags_fixed.view(-1)].view(*span_tags_fixed.shape, -1)
        
        return text_visual_embeddings

    def prepare_text_from_qa(self, question, question_tags, question_mask, answer, answer_tags, answer_mask):
        batch_size, max_q_len = question.shape
        _, max_a_len = answer.shape

        if self.predict_on_cls:
            answer_mask = answer_mask.new_zeros(answer_mask.shape) # remove answer_mask
            max_len = (question_mask.sum(1) + answer_mask.sum(1)).max() + 2 # [CLS] & 1*[SEP]
        else:
            max_len = (question_mask.sum(1) + answer_mask.sum(1)).max() + 3 # [CLS] & 2*[SEP]
        cls_id, sep_id = self.tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
        q_end = 1 + question_mask.sum(1, keepdim=True)
        a_end = q_end if self.predict_on_cls else q_end + 1 + answer_mask.sum(1, keepdim=True)
        input_ids = torch.zeros((batch_size, max_len), dtype=question.dtype, device=question.device)
        input_mask = torch.ones((batch_size, max_len), dtype=torch.uint8, device=question.device)
        input_type_ids = torch.zeros((batch_size, max_len), dtype=question.dtype, device=question.device)
        text_tags = input_type_ids.new_zeros((batch_size, max_len))
        grid_i, grid_j = torch.meshgrid(torch.arange(batch_size, device=question.device),
                                        torch.arange(max_len, device=question.device))

        input_mask[grid_j > a_end] = 0
        if not self.predict_on_cls:
            input_type_ids[(grid_j > q_end) & (grid_j <= a_end)] = 1
        q_input_mask = (grid_j > 0) & (grid_j < q_end)
        a_input_mask = (grid_j > q_end) & (grid_j < a_end)
        input_ids[:, 0] = cls_id
        input_ids[grid_j == q_end] = sep_id
        input_ids[grid_j == a_end] = sep_id
        input_ids[q_input_mask] = question[question_mask]
        input_ids[a_input_mask] = answer[answer_mask]
        text_tags[q_input_mask] = question_tags[question_mask]
        text_tags[a_input_mask] = answer_tags[answer_mask]

        ans_pos = a_end.new_zeros(a_end.shape).squeeze(1) if self.predict_on_cls else (a_end - 1).squeeze(1)

        return input_ids, input_type_ids, text_tags, input_mask, ans_pos

    def train_forward(self, img, im_info, boxes, labels, spo_ids, spo_lens, img_path):
        boxes, labels, spo_ids, spo_lens, im_info = boxes.squeeze(0), labels.squeeze(0), spo_ids.squeeze(0), spo_lens.squeeze(0), im_info.squeeze(0)

        images = torch.cat([img for _ in range(boxes.shape[0])]) # (Pdb) images.shape = torch.Size([4, 3, 895, 899])
        box_mask = (boxes[:, :, 0] > - 1.5) # (Pdb) box_mask.shape = torch.Size([4, 54])
        max_len = int(box_mask.sum(1).max().item()) # max_len = 54
        box_mask = box_mask[:, :max_len] # doesn't seem to have effect
        boxes = boxes[:, :max_len] # doesn't seem to have effect
        
        boxes[boxes<0]=0 # rectify those coordinates < 0 to 0

        obj_reps = self.image_feature_extractor(images=images,
                                                boxes=boxes,
                                                box_mask=box_mask,
                                                im_info=im_info,
                                                classes=None,
                                                segms=None,
                                                copy_images=True)
        # obj_reps['obj_reps'].shape = torch.Size([4, 54, 768])

        question_ids = spo_ids
        question_tags = spo_ids.new_zeros(question_ids.shape)
        question_mask = (spo_ids > 0.5)

        answer_ids = question_ids.new_zeros((question_ids.shape[0], 1)).fill_(
            self.tokenizer.convert_tokens_to_ids(['[MASK]'])[0])
        answer_mask = question_mask.new_zeros(answer_ids.shape).fill_(1)
        answer_tags = question_tags.new_zeros(answer_ids.shape)

        ############################################

        # prepare text
        text_input_ids, text_token_type_ids, text_tags, text_mask, ans_pos = self.prepare_text_from_qa(question_ids,
                                                                                                       question_tags,
                                                                                                       question_mask,
                                                                                                       answer_ids,
                                                                                                       answer_tags,
                                                                                                       answer_mask)
        if self.config.NETWORK.NO_GROUNDING: # always False
            obj_rep_zeroed = obj_reps['obj_reps'].new_zeros(obj_reps['obj_reps'].shape)
            text_tags.zero_()
            text_visual_embeddings = self._collect_obj_reps(text_tags, obj_rep_zeroed)
        else:
            text_visual_embeddings = self._collect_obj_reps(text_tags, obj_reps['obj_reps'], spo_lens)

        assert self.config.NETWORK.VLBERT.object_word_embed_mode == 2
        object_linguistic_embeddings = self.object_linguistic_embeddings(
            boxes.new_zeros((boxes.shape[0], boxes.shape[1])).long()
        )
        
        object_vl_embeddings = torch.cat((obj_reps['obj_reps'], object_linguistic_embeddings), -1) # concatenation of obj visual & linguistic

        ###########################################

        # Visual Linguistic BERT

        hidden_states, hc, spo_fused_masks = self.vlbert(text_input_ids,
                                      text_token_type_ids,
                                      text_visual_embeddings,
                                      text_mask,
                                      object_vl_embeddings,
                                      box_mask,
                                      object_visual_feat=obj_reps['obj_reps_rawraw'],
                                      spo_len=spo_lens,
                                      output_all_encoded_layers=False)
        _batch_inds = torch.arange(spo_ids.shape[0], device=spo_ids.device)

        hm = hidden_states[_batch_inds, ans_pos]

        if self.use_spatial_model:
            if self.use_coord_vector:
                # import pdb; pdb.set_trace()
                spa_feat = torch.zeros((boxes.shape[0], 5*2+9), dtype=boxes.dtype, layout=boxes.layout, device=boxes.device)
                for i in range(boxes.shape[0]):
                    area_subj_ratio = (boxes[i,1,2] - boxes[i,1,0]) * (boxes[i,1,3] - boxes[i,1,1]) / (im_info[i, 0] * im_info[i, 1])
                    subj = torch.tensor([boxes[i,1,0] / im_info[i,0], boxes[i,1,1] / im_info[i,1], boxes[i,1,2] / im_info[i,0], boxes[i,1,3] / im_info[i,1], area_subj_ratio])
                    
                    area_pred_ratio = (boxes[i,2,2] - boxes[i,2,0]) * (boxes[i,2,3] - boxes[i,2,1]) / (im_info[i, 0] * im_info[i, 1])
                    w_s = (boxes[i,1,2] - boxes[i,1,0])
                    h_s = (boxes[i,1,3] - boxes[i,1,1])
                    x_s = (boxes[i,1,2] + boxes[i,1,0]) / 2
                    y_s = (boxes[i,1,3] + boxes[i,1,1]) / 2
                    w_o = (boxes[i,3,2] - boxes[i,3,0])
                    h_o = (boxes[i,3,3] - boxes[i,3,1])
                    x_o = (boxes[i,3,2] + boxes[i,3,0]) / 2
                    y_o = (boxes[i,3,3] + boxes[i,3,1]) / 2
                    pred = torch.tensor([(x_s - x_o) / w_o, (y_s - y_o) / h_o, torch.log(w_s / w_o), torch.log(h_s / h_o), (x_o - x_s) / w_s, (y_o - y_s) / h_s, torch.log(w_o / w_s), torch.log(h_o / h_s), area_pred_ratio])
                    
                    area_obj_ratio = (boxes[i,3,2] - boxes[i,3,0]) * (boxes[i,3,3] - boxes[i,3,1]) / (im_info[i, 0] * im_info[i, 1])
                    obj = torch.tensor([boxes[i,3,0] / im_info[i,0], boxes[i,3,1] / im_info[i,1], boxes[i,3,2] / im_info[i,0], boxes[i,3,3] / im_info[i,1], area_obj_ratio])

                    spa_feat[0] = torch.cat((subj, pred, obj)).unsqueeze(0)
                spa_feat = self.loc_fcs(spa_feat)
                # assert self.spa_concat # Currently coord_vec only works with concatenation!
            else:
                for i in range(boxes.shape[0]):
                    boxes[:,:,0][i] /= im_info[:,0][i]
                    boxes[:,:,1][i] /= im_info[:,1][i]
                    boxes[:,:,2][i] /= im_info[:,0][i]
                    boxes[:,:,3][i] /= im_info[:,1][i]
                spa_feat = self.simple_spatial_model(boxes[:, 1], boxes[:, 3], labels)
            
            if self.spa_add:
                hm = hm * (1 - self.spa_feat_weight) + spa_feat * self.spa_feat_weight
            elif self.spa_concat: 
                hm = torch.cat((hm, spa_feat), dim=1)
            hm = self.spa_fusion_linear(hm)
            hm = F.relu(hm)
            hm = self.dropout(hm)
            
            if self.spa_one_more_layer: # if no unfrozen VLBERT add one more layer and lower the dropout rate to 0.2
                hm = self.spa_linear_hidden(hm)
                hm = F.relu(hm)
            
            hm = self.spa_linear(hm)

        if self.use_uvtranse:
            union_vec = obj_reps['obj_reps'][:, 2] - obj_reps['obj_reps'][:, 1] - obj_reps['obj_reps'][:, 3] # pred - subj - obj
            union_vec = self.union_vec_fc(union_vec)
            union_vec = F.relu(union_vec)
            
            if self.uvt_add:
                hm = hm * (1 - self.uvt_feat_weight) + union_vec * self.uvt_feat_weight
            elif self.uvt_concat:
                hm = torch.cat((hm, union_vec), dim=1)

            hm = self.uvt_fusion_linear(hm)
            hm = F.relu(hm)
            hm = self.dropout_uvt(hm)
            hm = self.uvt_linear(hm)
            
            # import pdb; pdb.set_trace()

        ###########################################
        outputs = {}

        # classifier
        logits = self.final_mlp(hm)

        # loss
        # import pdb; pdb.set_trace()
        ans_loss = F.cross_entropy(logits, labels.view(-1)) # * label.size(1)

        # Add sigmoid for binary prediction in spasen_metrics.py
        logits = F.softmax(logits, dim=1)

        # mask loss
        if spo_fused_masks is not None:
            nb_of_tokens = 2 if self.no_predicate else 3
            spo_fused_masks = spo_fused_masks.view(-1, nb_of_tokens, 14, 14)
            # spo_fused_masks_norm = spo_fused_masks.new_zeros(size=spo_fused_masks.shape)
            boxes_mask = torch.zeros_like(spo_fused_masks)
            rounded_14x14_boxes = torch.round(boxes*14).to(torch.int)
            for i in range(boxes.shape[0]): # for each sample
                for j in range(nb_of_tokens): # sub, pred, obj
                    # Create a mask
                    boxes_mask[i, j, rounded_14x14_boxes[i, j+1, 0].item():rounded_14x14_boxes[i, j+1, 2].item(), rounded_14x14_boxes[i, j+1, 1].item():rounded_14x14_boxes[i, j+1, 3].item()] = 1

            if self.mask_loss_sum:
                mask_loss = F.binary_cross_entropy_with_logits(spo_fused_masks, boxes_mask, reduction='sum') /spo_fused_masks.shape[0]
            elif self.mask_loss_mse:
                mask_loss = F.mse_loss(spo_fused_masks, boxes_mask)
            else:
                mask_loss = F.binary_cross_entropy_with_logits(spo_fused_masks, boxes_mask)

            outputs.update({'label_logits': logits,
                            'label': labels,
                            'ans_loss': ans_loss,
                            'mask_loss': mask_loss})
            
            if self.mask_weight < 0:
                loss = (ans_loss + mask_loss).mean()
            else:
                loss = (ans_loss * (1-self.mask_weight) + mask_loss * self.mask_weight).mean()
        else:
            outputs.update({'label_logits': logits,
                            'label': labels,
                            'ans_loss': ans_loss})
            loss = ans_loss.mean()

        return outputs, loss

    def inference_forward(self, img, im_info, boxes, labels, spo_ids, spo_lens, img_path, rels_cand, labels_so_ids, subj_obj_classes):
        boxes, labels, spo_ids, spo_lens, im_info, rels_cand, labels_so_ids, subj_obj_classes = boxes.squeeze(0), labels.squeeze(0), spo_ids.squeeze(0), spo_lens.squeeze(0), im_info.squeeze(0), rels_cand.squeeze(0), labels_so_ids.squeeze(0), subj_obj_classes.squeeze(0)

        # visual feature extraction
        images = torch.cat([img for _ in range(boxes.shape[0])]) 
        box_mask = (boxes[:, :, 0] > - 1.5)
        max_len = int(box_mask.sum(1).max().item())
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]

        obj_reps = self.image_feature_extractor(images=images,
                                                boxes=boxes,
                                                box_mask=box_mask,
                                                im_info=im_info,
                                                classes=None,
                                                segms=None,
                                                copy_images=True)

        question_ids = spo_ids
        question_tags = spo_ids.new_zeros(question_ids.shape)
        question_mask = (spo_ids > 0.5)

        answer_ids = question_ids.new_zeros((question_ids.shape[0], 1)).fill_(
            self.tokenizer.convert_tokens_to_ids(['[MASK]'])[0])
        answer_mask = question_mask.new_zeros(answer_ids.shape).fill_(1)
        answer_tags = question_tags.new_zeros(answer_ids.shape)

        ############################################

        # prepare text
        text_input_ids, text_token_type_ids, text_tags, text_mask, ans_pos = self.prepare_text_from_qa(question_ids,
                                                                                                       question_tags,
                                                                                                       question_mask,
                                                                                                       answer_ids,
                                                                                                       answer_tags,
                                                                                                       answer_mask)
        if self.config.NETWORK.NO_GROUNDING:
            obj_rep_zeroed = obj_reps['obj_reps'].new_zeros(obj_reps['obj_reps'].shape)
            text_tags.zero_()
            text_visual_embeddings = self._collect_obj_reps(text_tags, obj_rep_zeroed)
        else:
            text_visual_embeddings = self._collect_obj_reps(text_tags, obj_reps['obj_reps'], spo_lens)

        assert self.config.NETWORK.VLBERT.object_word_embed_mode == 2
        object_linguistic_embeddings = self.object_linguistic_embeddings(
            boxes.new_zeros((boxes.shape[0], boxes.shape[1])).long()
        )
        object_vl_embeddings = torch.cat((obj_reps['obj_reps'], object_linguistic_embeddings), -1)

        ###########################################

        # Visual Linguistic BERT

        hidden_states, hc, spo_fused_masks = self.vlbert(text_input_ids,
                                      text_token_type_ids,
                                      text_visual_embeddings,
                                      text_mask,
                                      object_vl_embeddings,
                                      box_mask,
                                      object_visual_feat=obj_reps['obj_reps_rawraw'],
                                      spo_len=spo_lens,
                                      output_all_encoded_layers=False)
        _batch_inds = torch.arange(spo_ids.shape[0], device=spo_ids.device)

        hm = hidden_states[_batch_inds, ans_pos]

        if self.use_spatial_model:
            if self.use_coord_vector:
                # import pdb; pdb.set_trace()
                spa_feat = torch.zeros((boxes.shape[0], 5*2+9), dtype=boxes.dtype, layout=boxes.layout, device=boxes.device)
                for i in range(boxes.shape[0]):
                    area_subj_ratio = (boxes[i,1,2] - boxes[i,1,0]) * (boxes[i,1,3] - boxes[i,1,1]) / (im_info[i, 0] * im_info[i, 1])
                    subj = torch.tensor([boxes[i,1,0] / im_info[i,0], boxes[i,1,1] / im_info[i,1], boxes[i,1,2] / im_info[i,0], boxes[i,1,3] / im_info[i,1], area_subj_ratio])
                    
                    area_pred_ratio = (boxes[i,2,2] - boxes[i,2,0]) * (boxes[i,2,3] - boxes[i,2,1]) / (im_info[i, 0] * im_info[i, 1])
                    w_s = (boxes[i,1,2] - boxes[i,1,0])
                    h_s = (boxes[i,1,3] - boxes[i,1,1])
                    x_s = (boxes[i,1,2] + boxes[i,1,0]) / 2
                    y_s = (boxes[i,1,3] + boxes[i,1,1]) / 2
                    w_o = (boxes[i,3,2] - boxes[i,3,0])
                    h_o = (boxes[i,3,3] - boxes[i,3,1])
                    x_o = (boxes[i,3,2] + boxes[i,3,0]) / 2
                    y_o = (boxes[i,3,3] + boxes[i,3,1]) / 2
                    pred = torch.tensor([(x_s - x_o) / w_o, (y_s - y_o) / h_o, torch.log(w_s / w_o), torch.log(h_s / h_o), (x_o - x_s) / w_s, (y_o - y_s) / h_s, torch.log(w_o / w_s), torch.log(h_o / h_s), area_pred_ratio])
                    
                    area_obj_ratio = (boxes[i,3,2] - boxes[i,3,0]) * (boxes[i,3,3] - boxes[i,3,1]) / (im_info[i, 0] * im_info[i, 1])
                    obj = torch.tensor([boxes[i,3,0] / im_info[i,0], boxes[i,3,1] / im_info[i,1], boxes[i,3,2] / im_info[i,0], boxes[i,3,3] / im_info[i,1], area_obj_ratio])

                    spa_feat[0] = torch.cat((subj, pred, obj)).unsqueeze(0)
                spa_feat = self.loc_fcs(spa_feat)
                # assert self.spa_concat # Currently coord_vec only works with concatenation!
            else:
                for i in range(boxes.shape[0]):
                    boxes[:,:,0][i] /= im_info[:,0][i]
                    boxes[:,:,1][i] /= im_info[:,1][i]
                    boxes[:,:,2][i] /= im_info[:,0][i]
                    boxes[:,:,3][i] /= im_info[:,1][i]
                spa_feat = self.simple_spatial_model(boxes[:, 1], boxes[:, 3], labels)
            
            if self.spa_add:
                hm = hm * (1 - self.spa_feat_weight) + spa_feat * self.spa_feat_weight
            elif self.spa_concat: 
                hm = torch.cat((hm, spa_feat), dim=1)
            hm = self.spa_fusion_linear(hm)
            hm = F.relu(hm)
            hm = self.dropout(hm)
            
            if self.spa_one_more_layer: # if no unfrozen VLBERT add one more layer and lower the dropout rate to 0.2
                hm = self.spa_linear_hidden(hm)
                hm = F.relu(hm)
            
            hm = self.spa_linear(hm)

        if self.use_uvtranse:
            union_vec = obj_reps['obj_reps'][:, 2] - obj_reps['obj_reps'][:, 1] - obj_reps['obj_reps'][:, 3] # pred - subj - obj
            union_vec = self.union_vec_fc(union_vec)
            union_vec = F.relu(union_vec)
            
            if self.uvt_add:
                hm = hm * (1 - self.uvt_feat_weight) + union_vec * self.uvt_feat_weight
            elif self.uvt_concat:
                hm = torch.cat((hm, union_vec), dim=1)

            hm = self.uvt_fusion_linear(hm)
            hm = F.relu(hm)
            hm = self.dropout_uvt(hm)
            hm = self.uvt_linear(hm)

        ###########################################
        outputs = {}

        # classifier
        logits = self.final_mlp(hm)
        logits = F.softmax(logits, dim=1)

        # mask loss
        if spo_fused_masks is not None:
            nb_of_tokens = 2 if self.no_predicate else 3
            spo_fused_masks = spo_fused_masks.view(-1, nb_of_tokens, 14, 14)

            boxes_mask = boxes.new_zeros(size=(boxes.shape[0], nb_of_tokens, 14, 14))
            rounded_14x14_boxes = torch.round(boxes*14).to(torch.int)
            for i in range(boxes.shape[0]): # for each sample
                for j in range(nb_of_tokens): # sub, pred, obj
                    # Create a mask
                    boxes_mask[i, j, rounded_14x14_boxes[i, j+1, 0].item():rounded_14x14_boxes[i, j+1, 2].item(), rounded_14x14_boxes[i, j+1, 1].item():rounded_14x14_boxes[i, j+1, 3].item()] = 1

            if self.mask_loss_sum:
                mask_loss = F.binary_cross_entropy_with_logits(spo_fused_masks, boxes_mask, reduction='sum') /spo_fused_masks.shape[0]
            elif self.mask_loss_mse:
                mask_loss = F.mse_loss(spo_fused_masks, boxes_mask)
                # import pdb; pdb.set_trace()
                # self.show_cam_on_image(spo_fused_masks, img_path)
            else:
                mask_loss = F.binary_cross_entropy_with_logits(spo_fused_masks, boxes_mask)

            outputs.update({'label_logits': logits,
                            'label': labels,
                            'labels_so_ids': labels_so_ids,
                            'rels_cand': rels_cand,
                            'mask_loss': mask_loss,
                            'img_path': img_path,
                            'spo_fused_masks': spo_fused_masks,
                            'subj_obj_classes': subj_obj_classes,
                            'prediction': logits.argmax(1)})
        else:
            outputs.update({'label_logits': logits,
                            'label': labels,
                            'labels_so_ids': labels_so_ids,
                            'rels_cand': rels_cand,
                            'prediction': logits.argmax(1)})

        return outputs
