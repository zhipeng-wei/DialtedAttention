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
from ensemble_attack import Ensemble_CommonWeakness, Ensemble_Baseline

# python IJCV_ensemble_main.py --gpu 0 
def arg_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--black_box', type=str, default='densenet121', help='inception_v3, resnet50, densenet121, vgg16_bn')
    parser.add_argument('--batch_size', type=int, default=5, metavar='N',
                    help='input batch size for reference (default: 10)')
    parser.add_argument('--dataset', type=str, default='ImageDataset', help='class in dataset.py')
    parser.add_argument('--attack', type=str, default='Ensemble_CommonWeakness', help='class in attacks.py')
    parser.add_argument('--loss_fn', type=str, default='Logit', help='CE or Logit')
    parser.add_argument('--step', type=int, default=300)

    parser.add_argument('--target', action='store_true')
    parser.add_argument('--no-target', dest='target', action='store_false')
    parser.set_defaults(target=True)
    
    parser.add_argument('--saveperts', action='store_true')
    parser.add_argument('--no-saveperts', dest='saveperts', action='store_false')
    parser.set_defaults(saveperts=False)

    parser.add_argument('--SI', action='store_true')
    parser.set_defaults(SI=False)

    parser.add_argument('--SU', action='store_true')
    parser.set_defaults(SU=False)
    
    parser.add_argument('--RAP', action='store_true')
    parser.set_defaults(RAP=False)
    
    parser.add_argument('--Ours', action='store_true')
    parser.set_defaults(Ours=False)
    
    parser.add_argument('--part', type=int, default=1)
    parser.add_argument('--part_index', type=int, default=1)
    

    args = parser.parse_args()
    args.adv_path = os.path.join(OPT_PATH, 'Ensemble-{}-{}-{}-Target_SI_{}-SU_{}-RAP_{}-Ours_{}'.format(args.black_box, args.dataset, args.attack, args.SI, args.SU, args.RAP, args.Ours))
    
    os.makedirs(args.adv_path, exist_ok=True)
    return args


if __name__ == '__main__':
    args = arg_parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    torch.manual_seed(1024)
    torch.backends.cudnn.deterministic = True

    # loading models
    all_models = ['inception_v3', 'resnet50', 'densenet121', 'vgg16_bn']
    if args.black_box == 'none':
        black_models = []
    else:
        if args.black_box != 'inception_v3':
            black_model = nn.Sequential(Normalize(), getattr(models, args.black_box)(pretrained=True)).eval()
        else:
            black_model = nn.Sequential(Normalize(), getattr(models, args.black_box)(pretrained=True, transform_input=True)).eval()
        for param in black_model[1].parameters():
            param.requires_grad = False
        black_model.cuda()
        black_models = [black_model]

    white_model_names = [i for i in all_models if i != args.black_box]
    white_models = []
    for model_name in white_model_names:
        if model_name == 'inception_v3':
            this_model = nn.Sequential(Normalize(), getattr(models, model_name)(pretrained=True, transform_input=True)).eval()
        else:
            this_model =  nn.Sequential(Normalize(), getattr(models, model_name)(pretrained=True)).eval()
        for param in this_model[1].parameters():
            param.requires_grad = False  
        this_model.cuda()
        white_models.append(this_model)

    # loading dataset
    attack_dataset = getattr(dataset, args.dataset)(part=args.part, part_index=args.part_index)
    print ('length of attack dataset', len(attack_dataset))
    dataloader = torch.utils.data.DataLoader(
                    attack_dataset, batch_size=args.batch_size,
                    num_workers=4, shuffle=False, pin_memory=True)

    # attack
    if args.attack == 'Ensemble_Baseline':
        adversory = Ensemble_Baseline(None, white_models, black_models, loss_fn=args.loss_fn, eval_steps=[i+20 for i in range(0, args.step, 20)], steps=args.step, target=args.target)
    else:
        adversory = Ensemble_CommonWeakness(white_model_names, white_models, black_models, loss_fn=args.loss_fn, eval_steps=[i+20 for i in range(0, args.step, 20)], steps=args.step, \
            target=args.target, SI=args.SI, SU=args.SU, RAP=args.RAP, Ours=args.Ours)

    # main loop
    result = None
    save_flag = 0
    for inputs, ori_labels, target_labels in tqdm(dataloader):
        if isinstance(inputs, list):
            for ind in range(len(inputs)):
                inputs[ind] = inputs[ind].cuda()
        else:
            inputs = inputs.cuda()
        ori_labels = ori_labels.cuda()
        target_labels = target_labels.cuda()

        re, perturbations = adversory.perturb(inputs, ori_labels, target_labels)
        if result is None:
            result = re
        else:
            result += re
        if args.saveperts:
            for per_ind in range(args.batch_size):
                ori_l, tar_l = ori_labels[per_ind].item(), target_labels[per_ind].item()
                tmp_pert = perturbations[per_ind].detach().clone().cpu().numpy()
                np.save(os.path.join(args.adv_path, 'pert-from_{}_to_{}_{}'.format(ori_l, tar_l, save_flag)), tmp_pert)
                save_flag += 1

        torch.cuda.empty_cache()
    
    # save result
    df = pd.DataFrame(columns = ['iter'] + [args.black_box])
    for ind, itr in enumerate(adversory.eval_steps):
        df.loc[ind] = [itr] + list(result[:,ind])
    df.to_csv(os.path.join(args.adv_path, f'result-{args.part}-{args.part_index}.csv'), index=False)

