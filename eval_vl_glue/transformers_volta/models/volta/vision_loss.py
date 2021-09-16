# coding: utf-8


import torch


class KL1601(object):
    
    input_dim = 1601
    
    @staticmethod
    def calc(prediction_scores_v, weight, label, image_cls, image_feat, obj_labels, obj_confs, attr_labels, attr_confs):
        if (weight > 0) and (image_cls is not None):
            image_target = image_cls
            loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(prediction_scores_v, dim=2), 
                image_target, 
                reduction="none"
            )
            return weight * torch.sum(loss * (label == 1).unsqueeze(2).float()) / max(torch.sum((label == 1)), 1)
        else:
            return 0


vl_pretraining_losses = {
    "0": KL1601,
    "1": None, #mse_2048,
    "2": None, #nce_2048,
    "3": None, #xent_1600,
    "4": None, #xent_400,
    "5": None, #huber_2048,
    "6": None, #xent_1601,
}
