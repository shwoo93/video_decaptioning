import torch
from torch import nn

from models import resnet, resnet_AE, resnet_mask, resnet_comp, unet_mask, icnet_mask, icnet_res

import pdb

def generate_model(opt):
    assert opt.model in [
        'resnet', 'resnet_AE', 'resnet_mask', 'resnet_comp', 'unet', 'icnet', 'icnet_res', 'icnet_res_2D', 
        'icnet_res_2Dt', 'icnet_DBI', 'icnet_deep', 'icnet_deep_gate', 'icnet_deep_gate_2step'
    ]

    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

        from models.resnet import get_fine_tuning_parameters

        if opt.model_depth == 10:
            model = resnet.resnet10(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 18:
            model = resnet.resnet18(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 34:
            model = resnet.resnet34(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 50:
            model = resnet.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 101:
            model = resnet.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 152:
            model = resnet.resnet152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 200:
            model = resnet.resnet200(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    
    elif opt.model == 'resnet_AE':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

        from models.resnet_AE import get_fine_tuning_parameters

        if opt.model_depth == 18:
            model = resnet_AE.resnet18(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                is_gray=opt.is_gray,
                opt=opt)
        elif opt.model_depth == 34:
            model = resnet_AE.resnet34(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                is_gray=opt.is_gray,
                opt=opt)
        elif opt.model_depth == 50:
            model = resnet_AE.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                is_gray=opt.is_gray,
                opt=opt)

    elif opt.model == 'resnet_mask':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

        from models.resnet_mask import get_fine_tuning_parameters

        if opt.model_depth == 18:
            model = resnet_mask.resnet18(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                is_gray=opt.is_gray,
                opt=opt)
        elif opt.model_depth == 34:
            model = resnet_mask.resnet34(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                is_gray=opt.is_gray,
                opt=opt)
        elif opt.model_depth == 50:
            model = resnet_mask.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                is_gray=opt.is_gray,
                opt=opt)

    elif opt.model == 'resnet_comp':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

        from models.resnet_comp import get_fine_tuning_parameters

        if opt.model_depth == 18:
            model = resnet_comp.resnet18(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                is_gray=opt.is_gray,
                opt=opt)
        elif opt.model_depth == 34:
            model = resnet_comp.resnet34(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                is_gray=opt.is_gray,
                opt=opt)
        elif opt.model_depth == 50:
            model = resnet_comp.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                is_gray=opt.is_gray,
                opt=opt)
    elif opt.model == 'unet':
        model = unet_mask.UNet3D(
            opt=opt
            )

    elif opt.model == 'icnet':
        model = icnet_mask.ICNet3D(
            opt=opt
            )
    elif opt.model == 'icnet_res':
        model = icnet_res.ICNetResidual3D(
            opt=opt
            )
    elif opt.model == 'icnet_res_2D':
        model = icnet_res.ICNetResidual2D(
            opt=opt
            )
    elif opt.model == 'icnet_res_2Dt':
        model = icnet_res.ICNetResidual2Dt(
            opt=opt
            )
    elif opt.model == 'icnet_DBI':
        model = icnet_res.ICNetResidual_DBI(
            opt=opt
            )
    elif opt.model == 'icnet_deep':
        model = icnet_res.ICNetDeep(
            opt=opt
            )
    elif opt.model == 'icnet_deep_gate':
        model = icnet_res.ICNetDeepGate(
            opt=opt
            )
    elif opt.model == 'icnet_deep_gate_2step':
        model = icnet_res.ICNetDeepGate2step(
            opt=opt
            )
    if not opt.no_cuda:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)

        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path)
            print('loading from', pretrain['arch'])

            child_dict = model.state_dict()
            if opt.two_step and opt.test:
                parent_list = pretrain['state_dict_1'].keys()
            else:
                parent_list = pretrain['state_dict'].keys()

            print('Not loaded :')
            parent_dict = {}
            for chi,_ in child_dict.items():
                # pdb.set_trace()
                # if ('coarse' in chi):
                    # chi_ori = chi
                    # chi = 'module.' + ".".join(chi_ori.split('.')[2:])

                if chi in parent_list:
                    if opt.two_step and opt.test:
                        parent_dict[chi] = pretrain['state_dict_1'][chi]
                    else:
                        parent_dict[chi] = pretrain['state_dict'][chi]
                else:
                    print(chi)
            print('length :', len(parent_dict.keys()))
            child_dict.update(parent_dict)
            model.load_state_dict(child_dict)
            
            if not opt.is_AE:
                model.module.fc = nn.Linear(model.module.fc.in_features, opt.n_finetune_classes)
            return model, model.parameters()

    else:          
        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path)
            assert pretrain['arch'] in ['resnet','resnet_AE']

            model.load_state_dict(pretrain['state_dict'])

            model.module.fc = nn.Linear(model.module.fc.in_features,
                                        opt.n_finetune_classes)
            if not opt.no_cuda:
                model.module.fc = model.module.fc.cuda()

            parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
            return model, parameters

    return model, model.parameters()