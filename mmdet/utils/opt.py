import torch

def build_optimizer_serge(model, config):
    parameters = []
    for name, p in model.named_parameters():
        if (config['train_module'] in name) and (config['hard_fix'] not in name):
            parameters.append(p)
        else:
            p.requires_grad = False
    optimizer = torch.optim.SGD(
        parameters, lr=config['lr'], momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )
    return model, optimizer

def build_optimizer_serge_recon(model, config):
    parameters = []
    for name, p in model.named_parameters():
        if ('encode3d' in name) or ('rotate' in name) and (config['fix_voxel'] != 0):
            par = {"params": p,
                "lr": config['lr']*config['fix_voxel']
                }
            parameters.append(par)
        elif ('encode3d' in name) or ('rotate' in name) and (config['fix_voxel'] == 0):
            p.requires_grad = False
        else:
            par = {"params": p,
                "lr": config['lr']
                }
            parameters.append(par)
    optimizer = torch.optim.SGD(
        parameters, lr=config['lr'], momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )
    return model, optimizer