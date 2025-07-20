import argparse
import os
import time
from tqdm import tqdm
import torch.nn as nn
import torch
import torchvision
from torchvision import models
import pandas as pd
import numpy as np
from functools import partial
import torchvision.transforms as transforms
import warnings

from utils import Normalize, OPT_PATH
from dataset import ImageDataset
import transfer_attacks
from utils_gradcam import find_vgg_layer, find_resnet_layer, find_densenet_layer, find_inception_layer


def arg_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--white_box', type=str, default='resnet50', help='inception_v3, resnet50, densenet121, vgg16_bn')
    parser.add_argument('--batch_size', type=int, default=10, metavar='N',
                    help='input batch size for reference (default: 10)')
    parser.add_argument('--attack', type=str, default='DTMI', help='class in attacks.py')
    parser.add_argument('--loss_fn', type=str, default='Logit', help='CE, Logit, or Attention')
    parser.add_argument('--step', type=int, default=300)
    parser.add_argument('--epsilon', type=float, default=16)
    parser.add_argument('--target', action='store_true')
    parser.add_argument('--no-target', dest='target', action='store_false')
    parser.set_defaults(target=True)

    # parameters for attack
    parser.add_argument('--baseline_cmd', type=str, default='I', help='DI, TI, MI, ODI, RAP, VT')
    parser.add_argument('--linear_aug', type=str, default='SI', help='SI, Admix, Ours')
    
    # divide dataset to parallelize or accelerate attacks
    parser.add_argument('--part', type=int, default=1, help='total number of dataset partitions')
    parser.add_argument('--part_index', type=int, default=1, help='index of the current partition to process')

    parser.add_argument('--saveimages', action='store_true')
    parser.add_argument('--no-saveimages', dest='saveimages', action='store_false')
    parser.set_defaults(saveimages=False)

    parser.add_argument('--layers', nargs='+')

    parser.add_argument('--file_suffix', type=str, default='experiment', help='suffix of saved files')
    args = parser.parse_args()
    args.adv_path = os.path.join(OPT_PATH, 'Our-Eps_{}-Wb_{}-Func_{}-Target_{}-Loss_{}-Method_{}_{}-{}'.format(args.epsilon, args.white_box, args.attack, args.target, args.loss_fn, args.baseline_cmd, args.linear_aug, args.file_suffix))
    os.makedirs(args.adv_path, exist_ok=True)
    print ('Saving to', args.adv_path)
    return args

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    args = arg_parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    torch.manual_seed(1024)
    torch.backends.cudnn.deterministic = True

    if args.part == 1:
        re_path = os.path.join(args.adv_path, 'result.csv')
    else:
        re_path = os.path.join(args.adv_path, 'result_{}.csv'.format(args.part_index))

    if os.path.exists(re_path):
        print ('Done')
        pass
    else:
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
        attack_dataset = ImageDataset(part=args.part, part_index=args.part_index)
        print ('length of attack dataset', len(attack_dataset))
        dataloader = torch.utils.data.DataLoader(
                        attack_dataset, batch_size=args.batch_size,
                        num_workers=4, shuffle=False, pin_memory=True)
        
        # attack 
        if args.attack == 'DTMI':
            adversor = getattr(transfer_attacks, args.attack)(args.baseline_cmd, args.linear_aug, white_model, eval_models, loss_fn=args.loss_fn, eval_steps=[i+20 for i in range(0, args.step, 20)], steps=args.step, \
                target=args.target)
        elif args.attack == 'DTMI_Untargeted':
            adversor = getattr(transfer_attacks, args.attack)(args.baseline_cmd, args.linear_aug, white_model, eval_models, loss_fn='CE', eval_steps=[10], steps=10, \
                target=False)
        elif args.attack == 'DTMI_DilatedAttention':
            if args.loss_fn == 'Attention':
                if args.white_box == 'inception_v3':
                    target_layers = [find_inception_layer(white_model[1], layer_name) for layer_name in args.layers]
                elif args.white_box == 'resnet50':
                    target_layers = [find_resnet_layer(white_model[1], layer_name) for layer_name in args.layers]
                elif args.white_box == 'densenet121':
                    target_layers = [find_densenet_layer(white_model[1], layer_name) for layer_name in args.layers]
                elif args.white_box == 'vgg16_bn':
                    target_layers = [find_vgg_layer(white_model[1], layer_name) for layer_name in args.layers]
            else:
                target_layers = []
                args.layers = []
            adversor = getattr(transfer_attacks, args.attack)(args.layers, target_layers, args.baseline_cmd, args.linear_aug, white_model, eval_models, loss_fn=args.loss_fn, eval_steps=[i+20 for i in range(0, args.step, 20)], steps=args.step, \
                target=args.target)
        
        # main loop
        result = None
        save_flag = 0
        loss_matrix = []
        grad_cs_saved = []

        all_time = []
        for inputs, ori_labels, target_labels in tqdm(dataloader):
            if isinstance(inputs, list):
                for ind in range(len(inputs)):
                    inputs[ind] = inputs[ind].cuda()
            else:
                inputs = inputs.cuda()
            ori_labels = ori_labels.cuda()
            target_labels = target_labels.cuda()
            

            start = time.time()
            re, adv, loss = adversor.perturb(inputs, ori_labels, target_labels)
            dur_time = time.time()-start
            all_time.append(dur_time)
            loss_matrix.append(loss)

            if args.saveimages:
                save_imgs = adv.detach().cpu()
                for image_id in range(len(save_imgs)):
                    ori_l, tar_l = ori_labels[image_id].item(), target_labels[image_id].item()
                    g_img = transforms.ToPILImage('RGB')(save_imgs[image_id])
                    output_dir = os.path.join(args.adv_path, 'adv_images')
                    os.makedirs(output_dir, exist_ok=True)
                
                    file_path = os.path.join(output_dir, '{}_from_{}_to_{}_{}.png'.format(args.part_index, ori_l, tar_l, save_flag))
                    with open(file_path, 'wb') as f:
                        g_img.save(f)
                    save_flag += 1

            if result is None:
                result = re
            else:
                result += re
            del re, adv
            torch.cuda.empty_cache()

        all_time = np.mean(all_time)
        if args.part == 1:
            # save result
            df = pd.DataFrame(columns = ['iter'] + black_models)
            for ind, itr in enumerate(adversor.eval_steps):
                df.loc[ind] = [itr] + list(result[:,ind])
            df.to_csv(os.path.join(args.adv_path, 'result_{}.csv'.format(all_time)), index=False)
        else:
            # save result
            df = pd.DataFrame(columns = ['iter'] + black_models)
            for ind, itr in enumerate(adversor.eval_steps):
                df.loc[ind] = [itr] + list(result[:,ind])
            df.to_csv(os.path.join(args.adv_path, 'result_{}-time_{}.csv'.format(args.part_index, all_time)), index=False)