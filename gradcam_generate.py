import glob
import os
import torch
from utils import OPT_PATH as opt_path
import numpy as np
import dataset
from utils_gradcam import visualize_cam, Normalize
import torchvision.models as models
from torchvision.utils import make_grid, save_image
from gradcam import GradCAM, GradCAMpp
import torch.nn.functional as F
import torchvision.transforms as T
from transforms import SingleTransforms
import argparse
from functools import partial

def arg_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--transform_name', type=str, default='DI', help='the name of transformation')
    parser.add_argument('--magnitude', type=int, default=3, help='we set the num_bins is 7')
    args = parser.parse_args()
    return args

def DI_upper(img, upper=330):
    rnd = np.random.randint(299, upper,size=1)[0]
    h_rem = upper - rnd
    w_rem = upper - rnd
    pad_top = np.random.randint(0, h_rem,size=1)[0]
    pad_bottom = h_rem - pad_top
    pad_left = np.random.randint(0, w_rem,size=1)[0]
    pad_right = w_rem - pad_left
    img_out = torch.nn.functional.pad(torch.nn.functional.interpolate(img, size=(rnd,rnd)),(pad_left,pad_top,pad_right,pad_bottom),mode='constant', value=0)
    return img_out

if __name__ == '__main__':
    args = arg_parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # runed_path = sorted(glob.glob(os.path.join(opt_path, 'detailDI_Paper*DI_Upper*CE*cam*330')))
    # runed_path = glob.glob(os.path.join(opt_path, 'Paper*DIWithDifferentUpperBound*Target_CE-*exp_f1_*'))
    # runed_path = [path for path in runed_path if path.split('/')[-1].split('_')[-1] in ['0', '330', '450', '570']]
    attack_dataset = getattr(dataset, 'ClassSamples5000')()

    vgg16_bn = models.vgg16_bn(pretrained=True)
    resnet50 = models.resnet50(pretrained=True)
    densenet121 = models.densenet121(pretrained=True)
    inception_v3 = models.inception_v3(pretrained=True)
    vgg16_bn.eval()
    vgg16_bn.cuda()
    resnet50.eval()
    resnet50.cuda()
    densenet121.eval()
    densenet121.cuda()
    inception_v3.eval()
    inception_v3.cuda()

    cam_dict = dict()
    vgg_model_dict = dict(type='vgg', arch=vgg16_bn, input_size=(299, 299))
    vgg_gradcam = GradCAM(vgg_model_dict, True)
    vgg_gradcampp = GradCAMpp(vgg_model_dict, True)
    cam_dict['vgg'] = [vgg_gradcam, vgg_gradcampp]

    resnet_model_dict = dict(type='resnet', arch=resnet50, input_size=(299, 299))
    resnet_gradcam = GradCAM(resnet_model_dict, True)
    resnet_gradcampp = GradCAMpp(resnet_model_dict, True)
    cam_dict['resnet'] = [resnet_gradcam, resnet_gradcampp]

    densenet_model_dict = dict(type='densenet', arch=densenet121, input_size=(299, 299))
    densenet_gradcam = GradCAM(densenet_model_dict, True)
    densenet_gradcampp = GradCAMpp(densenet_model_dict, True)
    cam_dict['densenet'] = [densenet_gradcam, densenet_gradcampp]

    inception_model_dict = dict(type='inception', arch=inception_v3, input_size=(299, 299))
    inception_gradcam = GradCAM(inception_model_dict, True)
    inception_gradcampp = GradCAMpp(inception_model_dict, True)
    cam_dict['inception'] = [inception_gradcam, inception_gradcampp]

    if args.transform_name == 'DI':
        transform_fuc = partial(DI_upper, upper=args.magnitude)
    else:
        transform_fuc = SingleTransforms(args.transform_name, args.magnitude)

    normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    all_image_ious = np.zeros((len(attack_dataset), 2, 4))
    # for image_ind in range(20):
    for image_ind in range(len(attack_dataset)):
        print ('running', image_ind)
        image, ori_label, tar_label = attack_dataset[image_ind]
        # benign attention map of orginal and targeted labels.
        torch_image = image.unsqueeze(dim=0).cuda()
        normed_torch_image = normalizer(torch_image)
        
        aug_torch_image = transform_fuc(torch_image)
        aug_normed_torch_image = normalizer(aug_torch_image)

        # Calculate attention maps of targeted labels. And calculate theirs IoUs with benign images
        all_ious = np.zeros((2, 4))
        for label, lat in zip([ori_label, tar_label], ['ori', 'adv']):
            # for benign image
            benign_mask = []
            for model_name in cam_dict.keys():
                gradcam, gradcam_pp = cam_dict[model_name]
                mask, _ = gradcam(normed_torch_image, class_idx=label)
                benign_mask.append(mask>0.5)

            # for transform_func
            cam_flag = 0
            for model_name in cam_dict.keys():
                gradcam, gradcam_pp = cam_dict[model_name]
                mask, _ = gradcam(aug_normed_torch_image, class_idx=label)

                ref_mask = benign_mask[cam_flag]
                mask = T.functional.resize(mask, size=(299, 299))
                overlap_mask = (ref_mask) & (mask>0.5)
                union_mask = (ref_mask) | (mask>0.5)
                mask_iou = (torch.sum(overlap_mask)/(torch.sum(union_mask)+1e-3)).item()
                if lat == 'ori':
                    all_ious[0][cam_flag] = mask_iou
                elif lat == 'adv':
                    all_ious[1][cam_flag] = mask_iou
                cam_flag+=1
        all_image_ious[image_ind] = all_ious
    np.save('./augmentation-gradcam/{}-{}'.format(args.transform_name, args.magnitude), all_image_ious)
                # heatmap_pp, result_pp = visualize_cam(mask_pp.cpu(), aug_torch_image)
                # images.append(torch.stack([aug_torch_image.squeeze().cpu(), heatmap, heatmap_pp, result, result_pp], 0))
            # images = make_grid(torch.cat(images, 0), nrow=5)
            # save_image(images, './augmentation-gradcam/{}-benign-{}-{}_{}_{}.png'.format(image_ind, lat, args.transform_name, args.hierarchy, args.magnitude))

        # for label, lat in zip([ori_label, tar_label], ['ori', 'adv']):
        #     images = []
        #     for gradcam, gradcam_pp in cam_dict.values():
        #         mask, _ = gradcam(normed_torch_image, class_idx=label)
        #         heatmap, result = visualize_cam(mask.cpu(), torch_image)

        #         mask_pp, _ = gradcam_pp(normed_torch_image, class_idx=label)
        #         heatmap_pp, result_pp = visualize_cam(mask_pp.cpu(), torch_image)
        #         images.append(torch.stack([torch_image.squeeze().cpu(), heatmap, heatmap_pp, result, result_pp], 0))
        #     images = make_grid(torch.cat(images, 0), nrow=5)
        #     save_image(images, './gradcam/{}-benign-{}.png'.format(image_ind, lat))

        # adversarial attention map of orginal and targeted labels.
        # pert_path = 'pert-from_{}_to_{}_{}.npy'.format(ori_label, tar_label, image_ind)
        # for path in runed_path:
        #     white_pert_path = os.path.join(path, pert_path)
        #     pert = np.load(white_pert_path)
        #     torch_adv = image + torch.from_numpy(pert)
        #     torch_adv = torch_adv.unsqueeze(dim=0).cuda()
        #     normed_torch_adv = normalizer(torch_adv)
        #     for label, lat in zip([ori_label, tar_label], ['ori', 'adv']):
        #         images = []
        #         for gradcam, gradcam_pp in cam_dict.values():
        #             mask, _ = gradcam(normed_torch_adv, class_idx=label)
        #             heatmap, result = visualize_cam(mask.cpu(), torch_adv)

        #             mask_pp, _ = gradcam_pp(normed_torch_adv, class_idx=label)
        #             heatmap_pp, result_pp = visualize_cam(mask_pp.cpu(), torch_adv)
        #             images.append(torch.stack([torch_adv.squeeze().cpu(), heatmap, heatmap_pp, result, result_pp], 0))
        #         images = make_grid(torch.cat(images, 0), nrow=5)
        #         save_image(images, './gradcam/aaaaa_{}-{}-{}-{}.png'.format(image_ind, path.split('/')[-1].split('-')[1], path.split('/')[-1].split('-')[-1], lat))