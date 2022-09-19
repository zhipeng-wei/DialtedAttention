import argparse
import os
from tqdm import tqdm
import torch.nn as nn
import torch
import torchvision
from torchvision import models
import pandas as pd
import numpy as np
from utils import Normalize, load_ground_truth, OPT_PATH, NIPS_DATA
import dataset
import attacks

def arg_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--white_box', type=str, default='densenet121', help='inception_v3, resnet50, densenet121, vgg16_bn')
    parser.add_argument('--batch_size', type=int, default=10, metavar='N',
                    help='input batch size for reference (default: 10)')
    parser.add_argument('--dataset', type=str, default='NIPSDataset', help='class in dataset.py')
    parser.add_argument('--attack', type=str, default='DTMI', help='class in attacks.py')
    parser.add_argument('--loss_fn', type=str, default='CE', help='CE or Logit')
    
    parser.add_argument('--target', action='store_true')
    parser.add_argument('--no-target', dest='target', action='store_false')
    parser.set_defaults(target=True)
    parser.add_argument('--MI', action='store_true')
    parser.add_argument('--no-MI', dest='MI', action='store_false')
    parser.set_defaults(MI=True)
    parser.add_argument('--TI', action='store_true')
    parser.add_argument('--no-TI', dest='TI', action='store_false')
    parser.set_defaults(TI=True)
    parser.add_argument('--DI', action='store_true')
    parser.add_argument('--no-DI', dest='DI', action='store_false')
    parser.set_defaults(DI=True)

    # hyper-parameters used in this method
    parser.add_argument('--upper_bound', type=int, default=330, help='the coefficient of feature similarity loss')
    
    parser.add_argument('--saveperts', action='store_true')
    parser.add_argument('--no-saveperts', dest='saveperts', action='store_false')
    parser.set_defaults(saveperts=False)


    parser.add_argument('--file_tailor', type=str, default='experiment', help='')
    args = parser.parse_args()
    args.adv_path = os.path.join(OPT_PATH, 'Paper-{}-{}-{}-Target_{}-{}'.format(args.white_box, args.dataset, args.attack, args.loss_fn, args.file_tailor))
    if not os.path.exists(args.adv_path):
        os.makedirs(args.adv_path)
    return args


if __name__ == '__main__':
    args = arg_parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    torch.manual_seed(1024)
    torch.backends.cudnn.deterministic = True

    # loading models
    all_models = ['inception_v3', 'resnet50', 'densenet121', 'vgg16_bn']
    if args.white_box != 'inception_v3':
        white_model = nn.Sequential(Normalize(), getattr(models, args.white_box)(pretrained=True)).eval()
    else:
        white_model = nn.Sequential(Normalize(), getattr(models, args.white_box)(pretrained=True, transform_input=True)).eval()
    for param in white_model[1].parameters():
        param.requires_grad = False  
    white_model.cuda()

    black_models = [i for i in all_models if i != args.white_box]
    eval_models = []
    for model_name in black_models:
        if model_name == 'inception_v3':
            this_model = nn.Sequential(Normalize(), getattr(models, model_name)(pretrained=True, transform_input=True)).eval()
        else:
            this_model =  nn.Sequential(Normalize(), getattr(models, model_name)(pretrained=True)).eval()
        for param in this_model[1].parameters():
            param.requires_grad = False  
        this_model.cuda()
        eval_models.append(this_model)

    # loading dataset
    attack_dataset = getattr(dataset, args.dataset)()
    print ('length of attack dataset', len(attack_dataset))
    dataloader = torch.utils.data.DataLoader(
                    attack_dataset, batch_size=args.batch_size,
                    num_workers=4, shuffle=False, pin_memory=True)

    # attack
    if args.attack not in  ['DI_wo_prob', 'DI_logits_vary']:
        adversor = getattr(attacks, args.attack)(args.upper_bound, white_model, eval_models, loss_fn=args.loss_fn, eval_steps=[i+20 for i in range(0, 300, 20)], steps=300, \
                target=args.target)
    else:
        adversor = getattr(attacks, args.attack)(white_model, eval_models, loss_fn=args.loss_fn, eval_steps=[i+20 for i in range(0, 300, 20)], steps=300, \
                target=args.target)
    

    # main loop
    if args.attack in ['DI_wo_prob', 'DI_logits_vary']:
        result = None
    else:
        result = []
    save_flag = 0
    gradient_norm_matrix = []
    loss_matrix = []
    logits_infos = []
    for inputs, ori_labels, target_labels in tqdm(dataloader):
        if isinstance(inputs, list):
            for ind in range(len(inputs)):
                inputs[ind] = inputs[ind].cuda()
        else:
            inputs = inputs.cuda()
        ori_labels = ori_labels.cuda()
        target_labels = target_labels.cuda()
        if args.attack == 'DI_wo_prob':
            re, perturbations = adversor.perturb(inputs, ori_labels, target_labels)
            if result is None:
                result = re
            else:
                result += re
        elif args.attack == 'DI_logits_vary':
            re, perturbations, logits_info = adversor.perturb(inputs, ori_labels, target_labels)
            if result is None:
                result = re
            else:
                result += re
            logits_infos.append(logits_info)
        else:
            re, perturbations, norm, loss = adversor.perturb(inputs, ori_labels, target_labels)
            result.append(re)
            gradient_norm_matrix.append(norm)
            loss_matrix.append(loss)

        if args.saveperts:
            for per_ind in range(args.batch_size):
                ori_l, tar_l = ori_labels[per_ind].item(), target_labels[per_ind].item()
                tmp_pert = perturbations[per_ind].detach().clone().cpu().numpy()
                np.save(os.path.join(args.adv_path, 'pert-from_{}_to_{}_{}'.format(ori_l, tar_l, save_flag)), tmp_pert)
                save_flag += 1
        

    if args.attack == 'DI_wo_prob':
        # save result
        df = pd.DataFrame(columns = ['iter'] + black_models)
        for ind, itr in enumerate(adversor.eval_steps):
            df.loc[ind] = [itr] + list(result[:,ind])
        df.to_csv(os.path.join(args.adv_path, 'result.csv'), index=False)
    elif args.attack == 'DI_logits_vary':

        logits_infos = np.concatenate(logits_infos, axis=2)
        np.save(os.path.join(args.adv_path, 'logit_info'), logits_infos)

    else:
        # save result
        result = np.concatenate(result, axis=2)
        np.save(os.path.join(args.adv_path, 'result'), result)

        # save norm and loss
        gradient_norm_matrix = np.concatenate(gradient_norm_matrix, axis=1)
        np.save(os.path.join(args.adv_path, 'gradient_magnitude'), gradient_norm_matrix)
        loss_matrix = np.vstack(loss_matrix)
        np.save(os.path.join(args.adv_path, 'loss'), loss_matrix)
