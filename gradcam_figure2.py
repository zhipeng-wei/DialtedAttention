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

def arg_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--transform_name', type=str, default='DI', help='the name of transformation')
    parser.add_argument('--magnitude', type=int, default=3, help='we set the num_bins is 7')
    parser.add_argument('--hierarchy', action='store_true')
    parser.add_argument('--no-hierarchy', dest='hierarchy', action='store_false')
    parser.set_defaults(hierarchy=False)
    args = parser.parse_args()
    return args

def DI_upper(X_in, upper=330):
    rnd = np.random.randint(299, upper, size=1)[0]
    h_rem = upper - rnd
    w_rem = upper - rnd
    pad_top = np.random.randint(0, h_rem,size=1)[0]
    pad_bottom = h_rem - pad_top
    pad_left = np.random.randint(0, w_rem,size=1)[0]
    pad_right = w_rem - pad_left

    X_out = F.pad(F.interpolate(X_in, size=(rnd,rnd)),(pad_left,pad_top,pad_right,pad_bottom),mode='constant', value=0)
    return  X_out

if __name__ == '__main__':
    args = arg_parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # runed_path = sorted(glob.glob(os.path.join(opt_path, 'detailDI_Paper*DI_Upper*CE*cam*330')))
    runed_path = glob.glob(os.path.join(opt_path, 'Paper*DIWithDifferentUpperBound*Target_CE-*exp_f1_*'))
    runed_path = [path for path in runed_path if path.split('/')[-1].split('_')[-1] in ['0', '330', '450', '570']]
    attack_dataset = getattr(dataset, 'NIPSDataset')()

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


    normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    all_image_ious = []
    # for image_ind in range(20):
    for image_ind in [818]:
        print ('running', image_ind)
        image, ori_label, tar_label = attack_dataset[image_ind]
        torch_image = image.unsqueeze(dim=0).cuda()
        normed_torch_image = normalizer(torch_image)
        save_image(torch_image, './figure2/benign-{}.png'.format(image_ind))
        # for bengin images
        for label, lat in zip([ori_label, tar_label], ['ori', 'adv']):
            images = []
            for model_name in cam_dict.keys():
                gradcam, gradcam_pp = cam_dict[model_name]
                mask, _ = gradcam(normed_torch_image, class_idx=label)
                heatmap, result = visualize_cam(mask.cpu(), torch_image)

                # mask_pp, _ = gradcam_pp(normed_torch_image, class_idx=label)
                # heatmap_pp, result_pp = visualize_cam(mask_pp.cpu(), torch_image)
                # images.append(torch.stack([torch_image.squeeze().cpu(), result], 0))
                save_image(result, './figure2/benign-cam-{}-{}-{}.png'.format(model_name, image_ind, lat))

        for augsize in [330, 450, 570]:
            aug_torch_image = DI_upper(torch_image, augsize)
            aug_normed_torch_image = normalizer(aug_torch_image)
            save_image(aug_torch_image, './figure2/aug-{}-{}.png'.format(image_ind, augsize))
            for label, lat in zip([ori_label, tar_label], ['ori', 'adv']):
                images = []
                for model_name in cam_dict.keys():
                    gradcam, gradcam_pp = cam_dict[model_name]
                    mask, _ = gradcam(aug_normed_torch_image, class_idx=label)
                    heatmap, result = visualize_cam(mask.cpu(), aug_torch_image)

                    # mask_pp, _ = gradcam_pp(aug_normed_torch_image, class_idx=label)
                    # heatmap_pp, result_pp = visualize_cam(mask_pp.cpu(), aug_torch_image)
                    save_image(result, './figure2/aug-cam-{}-{}-{}-{}.png'.format(model_name, image_ind, augsize, lat))

        # adversarial attention map of orginal and targeted labels.
        pert_path = 'pert-from_{}_to_{}_{}.npy'.format(ori_label, tar_label, image_ind)
        for path in runed_path:
            white_pert_path = os.path.join(path, pert_path)
            pert = np.load(white_pert_path)
            torch_adv = image + torch.from_numpy(pert)
            torch_adv = torch_adv.unsqueeze(dim=0).cuda()
            normed_torch_adv = normalizer(torch_adv)
            augsize = path.split('/')[-1].split('-')[-1]
            save_image(torch_adv, './figure2/adv-{}-{}.png'.format(image_ind, augsize))
            for label, lat in zip([ori_label, tar_label], ['ori', 'adv']):
                images = []
                for model_name in cam_dict.keys():
                    gradcam, gradcam_pp = cam_dict[model_name]
                    mask, _ = gradcam(normed_torch_adv, class_idx=label)
                    heatmap, result = visualize_cam(mask.cpu(), torch_adv)

                    # mask_pp, _ = gradcam_pp(normed_torch_adv, class_idx=label)
                    # heatmap_pp, result_pp = visualize_cam(mask_pp.cpu(), torch_adv)
                    save_image(result, './figure2/adv-cam-{}-{}-{}-{}.png'.format(model_name, image_ind, augsize, lat))
