# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from pprint import pformat


def lr_wd_annealing(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= param_group.get('lr_scale', 1)  # 'lr_scale' could be assigned
        param_group['weight_decay'] *= param_group.get('weight_decay_scale', 1)  # 'weight_decay_scale' could be assigned


def get_param_groups(model, nowd_keys=(), lr_scale=0.0):
    print(lr_scale, hasattr(model, 'get_layer_id_and_scale_exp'))
    using_lr_scale = hasattr(model, 'get_layer_id_and_scale_exp') and 0.0 < lr_scale < 1.0
    assert using_lr_scale
    # print(f'[get_ft_param_groups][lr decay] using_lr_scale={using_lr_scale}, ft_lr_scale={lr_scale}')
    para_groups, para_groups_dbg = {}, {}
    
    for name, para in model.named_parameters():
        if not para.requires_grad:
            continue  # frozen weights
        if len(para.shape) == 1 or name.endswith('.bias') or any(k in name for k in nowd_keys):
            wd_scale, group_name = 0., 'no_decay'
        else:
            wd_scale, group_name = 1., 'decay'
        
        if using_lr_scale:
            layer_id, scale_exp = model.get_layer_id_and_scale_exp(name)
            group_name = f'layer{layer_id}_' + group_name
            this_lr_scale = lr_scale ** scale_exp
            dbg = f'[layer {layer_id}][sc = {lr_scale} ** {scale_exp}]'
        else:
            this_lr_scale = 1
            dbg = f'[no scale]'
        
        if group_name not in para_groups:
            para_groups[group_name] = {'params': [], 'weight_decay_scale': wd_scale, 'lr_scale': this_lr_scale}
            para_groups_dbg[group_name] = {'params': [], 'weight_decay_scale': wd_scale, 'lr_scale': dbg}
        para_groups[group_name]['params'].append(para)
        para_groups_dbg[group_name]['params'].append(name)
    
    for g in para_groups_dbg.values():
        g['params'] = pformat(', '.join(g['params']), width=200)
    
    return list(para_groups.values()), para_groups_dbg
