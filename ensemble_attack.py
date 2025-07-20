import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import scipy.stats as st
import torchvision
from functools import partial

from losses import LogitLoss
from utils_gradcam import find_vgg_layer, find_resnet_layer, find_densenet_layer, find_squeezenet_layer, find_inception_layer


class Base(object):
    def __init__(self, loss_fn, eval_steps, steps, target=False, random_start=False, epsilon=16./255., alpha=2./255.):
        self.eval_steps = eval_steps   
        self.steps = steps
        self.target = target
        self.random_start = random_start
        self.epsilon = epsilon
        self.alpha = alpha
        if loss_fn == 'CE':
            self.loss_fn = nn.CrossEntropyLoss()
        elif loss_fn == 'Logit':
            self.loss_fn = LogitLoss()

    def _DI(self, X_in):
        rnd = np.random.randint(299, 330,size=1)[0]
        h_rem = 330 - rnd
        w_rem = 330 - rnd
        pad_top = np.random.randint(0, h_rem,size=1)[0]
        pad_bottom = h_rem - pad_top
        pad_left = np.random.randint(0, w_rem,size=1)[0]
        pad_right = w_rem - pad_left

        c = np.random.rand(1)
        if c <= 0.7:
            X_out = F.pad(F.interpolate(X_in, size=(rnd,rnd)),(pad_left,pad_top,pad_right,pad_bottom),mode='constant', value=0)
            return  X_out 
        else:
            return  X_in

    def _DI_WO_Prob(self, X_in):
        rnd = np.random.randint(299, 330,size=1)[0]
        h_rem = 330 - rnd
        w_rem = 330 - rnd
        pad_top = np.random.randint(0, h_rem,size=1)[0]
        pad_bottom = h_rem - pad_top
        pad_left = np.random.randint(0, w_rem,size=1)[0]
        pad_right = w_rem - pad_left

        X_out = F.pad(F.interpolate(X_in, size=(rnd,rnd)),(pad_left,pad_top,pad_right,pad_bottom),mode='constant', value=0)
        X_out = F.interpolate(X_out, size=(X_in.shape[-2], X_in.shape[-1]))
        return  X_out 

    def _TI_kernel(self):
        def gkern(kernlen=15, nsig=3):
            x = np.linspace(-nsig, nsig, kernlen)
            kern1d = st.norm.pdf(x)
            kernel_raw = np.outer(kern1d, kern1d)
            kernel = kernel_raw / kernel_raw.sum()
            return kernel
        channels=3
        kernel_size=5
        kernel = gkern(kernel_size, 3).astype(np.float32)
        gaussian_kernel = np.stack([kernel, kernel, kernel])
        gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
        gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()
        return gaussian_kernel
    
    def _target_layer(self, model_names, depth):
        '''
        self.model_list, 
        'inception_v3', 'resnet50', 'densenet121', 'vgg16_bn'
        depth: [1, 2, 3, 4]
        '''
        target_layers = []
        for model_name, model in zip(model_names, self.model_list):
            if model_name == 'resnet50':
                target_layers.append(getattr(model[1], 'layer{}'.format(depth))[-1])
            elif model_name == 'vgg16_bn':
                depth_to_layer = {1:12,2:22,3:32,4:42}
                target_layers.append(getattr(model[1], 'features')[depth_to_layer[depth]])
            elif model_name == 'densenet121':
                target_layers.append(getattr(getattr(model[1], 'features'), 'denseblock{}'.format(depth)))
            elif model_name == 'inception_v3':
                depth_to_layer = {1:'Conv2d_4a_3x3', 2:'Mixed_5d', 3:'Mixed_6e', 4:'Mixed_7c'}
                target_layers.append(getattr(model[1], '{}'.format(depth_to_layer[depth])))
        return target_layers

    def perturb(self, images, ori_labels, target_labels):
        raise NotImplementedError

class Ensemble_Baseline(Base):
    '''
    Use TI, MI and DI as the baseline attack.
    '''
    def __init__(self, transform_func, model_list, eval_models, loss_fn, eval_steps, steps, target, DI=False, TI=False, MI=False):
        super(Ensemble_Baseline, self).__init__(loss_fn, eval_steps, steps, target)
        self.transform_func = transform_func
        self.model_list = model_list
        self.eval_models = eval_models

        self.DI, self.TI, self.MI = DI, TI, MI
        if self.TI:
            self.gaussian_kernel = self._TI_kernel()

    def perturb(self, images, ori_labels, target_labels):
        adv_images = images.clone()
        if self.target:
            used_label = target_labels
            used_coef = -1
        else:
            used_label = ori_labels
            used_coef = 1

        if self.random_start:
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.epsilon, self.epsilon)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
            delta = torch.tensor((adv_images - images).clone().detach(), requires_grad=True).cuda()
        else:
            delta = torch.zeros_like(adv_images,requires_grad=True).cuda()

        result_matrix = np.zeros((len(self.eval_models), len(self.eval_steps)))
        iter_flag = 0
        if self.MI:
            grad_pre = 0
        for itr in range(self.steps):
            if self.DI:
                logits = []
                for model in self.model_list:
                    logit = model(self.transform_func(adv_images+delta))
                    logits.append(logit)
                logits = torch.mean(torch.stack(logits, dim=0), dim=0)
                loss_label = used_label
            else:
                logits = []
                for model in self.model_list:
                    logit = model(adv_images+delta)
                    logits.append(logit)
                logits = torch.mean(torch.stack(logits, dim=0), dim=0)
                loss_label = used_label
            loss = self.loss_fn(logits, loss_label)
            loss.backward()
            grad_c = delta.grad.clone()
            if self.TI:
                grad_c = F.conv2d(grad_c, self.gaussian_kernel, bias=None, stride=1, padding=(2,2), groups=3)
            else:
                pass
            if self.MI:
                grad_a = grad_c / torch.mean(torch.abs(grad_c), (1, 2, 3), keepdim=True) + 1 * grad_pre
                grad_pre = grad_a
            else:
                grad_a = grad_c.clone()

            delta.grad.zero_()
            delta.data = delta.data + used_coef * self.alpha * torch.sign(grad_a)
            delta.data = delta.data.clamp(-self.epsilon, self.epsilon) 
            delta.data = ((adv_images + delta.data).clamp(0,1)) - adv_images
            if itr+1 in self.eval_steps:
                for m_id, model in enumerate(self.eval_models):
                    with torch.no_grad():
                        this_logit = model(adv_images+delta)
                    if self.target:
                        success_nums = torch.sum(torch.argmax(this_logit, dim=1) == used_label).cpu().item()
                    else:
                        success_nums = torch.sum(torch.argmax(this_logit, dim=1) != used_label).cpu().item()
                    result_matrix[m_id, iter_flag] = success_nums
                iter_flag += 1
        return result_matrix, delta.data

class Ensemble_CommonWeakness(Base):
    '''
    refer to https://github.com/huanranchen/AdversarialAttacks/blob/dbac7e7e32844f440876b57e6c0d3bbb95040a80/attacks/AdversarialInput/CommonWeakness.py#L175
    '''
    def __init__(self, model_name, model_list, eval_models, loss_fn, eval_steps, steps, target, reverse_step_size=16./255./15., inner_step_size=50, step_size=2./255., SI=False, SU=False, RAP=False, Ours=False):
        super(Ensemble_CommonWeakness, self).__init__(loss_fn, eval_steps, steps, target)
        self.model_list = model_list
        self.eval_models = eval_models
        self.model_name = model_name

        self.reverse_step_size = reverse_step_size
        self.inner_step_size = inner_step_size
        self.step_size = step_size

        self.SI, self.SU, self.RAP, self.Ours = SI, SU, RAP, Ours

        if self.SU:
            self.local_transform = torchvision.transforms.RandomResizedCrop(299, scale=(0.1, 0.1))
            self.depth = 3
            self._register_forward()

        if self.Ours:
            self.gamma = torch.ones(1)
            self.eta = torch.zeros(1)
            
            self.model_to_layers = {
                'densenet121': ['features_denseblock1', 'features_denseblock2', 'features_denseblock3', 'features_norm5'],
                'inception_v3':['Conv2d_4a_3x3', 'Mixed_5d', 'Mixed_6e', 'Mixed_7c'],
                'resnet50':['layer1', 'layer2', 'layer3', 'layer4',],
                'vgg16_bn':['features_12', 'features_22', 'features_32', 'features_42']
                }
            
            self.target_layer = []
            self.layer_names = []
            for model_id, one_model in enumerate(model_name):
                if one_model == 'inception_v3':
                    target_layers = [find_inception_layer(self.model_list[model_id][1], layer_name) for layer_name in self.model_to_layers[one_model]]
                elif one_model == 'resnet50':
                    target_layers = [find_resnet_layer(self.model_list[model_id][1], layer_name) for layer_name in self.model_to_layers[one_model]]
                elif one_model == 'densenet121':
                    target_layers = [find_densenet_layer(self.model_list[model_id][1], layer_name) for layer_name in self.model_to_layers[one_model]]
                elif one_model == 'vgg16_bn':
                    target_layers = [find_vgg_layer(self.model_list[model_id][1], layer_name) for layer_name in self.model_to_layers[one_model]]
                self.target_layer += target_layers
                self.layer_names += self.model_to_layers[one_model]
            self._attention_hook()

    def _si(self, inputs):
        inputs = torch.cat([inputs/(2**i) for i in range(5)], dim=0)
        return inputs

    def _cal_coef(self, para, mode, lower, upper, itr):
        if mode == 'uniform':
            return para.uniform_(lower, upper).item()
        elif mode == 'increment_linear':
            return torch.linspace(lower+1e-3, upper, self.steps, dtype = torch.float64)[itr].item()

    def _our_linear(self, inputs, itr):
        our_inputs = []
        for our_i in range(5):
            gamma = self._cal_coef(self.gamma, 'uniform', 0.0, 1.8, itr)
            eta = self._cal_coef(self.eta, 'increment_linear', 0.0, 2.2, itr)
            noise = torch.empty_like(inputs).uniform_(-self.epsilon, self.epsilon)
            our_inputs.append(torch.clamp(gamma*(inputs) + eta*noise, min=0, max=1))
        inputs = torch.cat(our_inputs, dim=0)
        return inputs

    def _initilize_gra_act(self):
        self.gradients = dict()
        self.activations = dict()

    def _attention_hook(self):
        self._initilize_gra_act()
        def backward_hook(module, grad_input, grad_output, layer_name):
            self.gradients[layer_name] = grad_output[0]
            return None
        def forward_hook(module, input, output, layer_name):
            self.activations[layer_name] = output
            return None
        if isinstance(self.target_layer, list):
            for i,j in zip(self.target_layer, self.layer_names):
                fuc_forward = partial(forward_hook, layer_name = j)
                fuc_backward = partial(backward_hook, layer_name = j)
                i.register_forward_hook(fuc_forward)
                i.register_backward_hook(fuc_backward) # register_full_backward_hook
        else:
            fuc_forward = partial(forward_hook, layer_name = self.layer_names)
            fuc_backward = partial(backward_hook, layer_name = self.layer_names)   
            self.target_layer.register_forward_hook(fuc_forward)
            self.target_layer.register_backward_hook(fuc_backward)

    def _attention_map(self, logits, delta, labels, retain_graph=True, create_graph=False):
        # return sum_losses (num_layers, num_images)
        # refer to https://stackoverflow.com/questions/54727099/activation-gradient-penalty
        attention_b, c, h, w = delta.size()
        score = logits.gather(1,labels.unsqueeze(1)).squeeze(1).sum()
        for model in self.model_list:
            model.zero_grad()

        sum_losses = []
        for layer_name in self.layer_names:
            grad = torch.autograd.grad(score, self.activations[layer_name], create_graph=True, only_inputs=True)[0]
            act = self.activations[layer_name]
            grad_b, k, u, v = grad.size()
            alpha = grad.view(grad_b, k, -1).mean(2)
            weights = alpha.view(grad_b, k, 1, 1)

            saliency_map = (weights*act).sum(1, keepdim=True)
            sum_losses.append(saliency_map.view(grad_b,-1).sum(-1))
        sum_losses = torch.stack(sum_losses)
        return sum_losses
    
    def _single_model_attention_map(self, model_name, logits, delta, labels, retain_graph=True, create_graph=False):
        # return sum_losses (num_layers, num_images)
        # refer to https://stackoverflow.com/questions/54727099/activation-gradient-penalty
        attention_b, c, h, w = delta.size()
        score = logits.gather(1,labels.unsqueeze(1)).squeeze(1).sum()

        index_of_model = self.model_name.index(model_name)
        used_model = self.model_list[index_of_model]
        used_layer_names = self.model_to_layers[model_name]
        used_model.zero_grad()

        sum_losses = []
        for layer_name in used_layer_names:
            grad = torch.autograd.grad(score, self.activations[layer_name], create_graph=True, only_inputs=True)[0]
            act = self.activations[layer_name]
            grad_b, k, u, v = grad.size()
            alpha = grad.view(grad_b, k, -1).mean(2)
            weights = alpha.view(grad_b, k, 1, 1)

            saliency_map = (weights*act).sum(1, keepdim=True)
            sum_losses.append(saliency_map.view(grad_b,-1).sum(-1))
        sum_losses = torch.stack(sum_losses)
        return sum_losses
    
    def _pgd_adv_example(self, model_list,
             x_natural,
             y,
             step_size=2/255,
             epsilon=16/255,
             perturb_steps=8,
             random_start=True):
        ''' refer to https://github.com/dongyp13/memorization-AT/blob/main/train.py'''
        batch_size = len(x_natural)
        # generate adversarial example
        if random_start:
            x_adv = x_natural.clone().detach() + torch.FloatTensor(*x_natural.shape).uniform_(-epsilon, epsilon).cuda()
        else:
            x_adv = x_natural.clone().detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                this_logit = 0
                for model in model_list:
                    this_logit += model(x_adv)
            loss = F.cross_entropy(this_logit, y)
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        pert = x_adv - x_natural
        return pert
    
    def _register_forward(self):
        '''
        'inception_v3', 'resnet50', 'densenet121', 'vgg16_bn'
        '''
        self.activations = []
        def forward_hook(module, input, output):
            self.activations += [output]
            return None
        target_layers = self._target_layer(self.model_name, self.depth)    
        for target_layer in target_layers:
            target_layer.register_forward_hook(forward_hook)

    def perturb(self, images, ori_labels, target_labels):
        adv_images = images.clone()
        b,c,h,w = images.shape
        N = adv_images.shape[0]
        if self.target:
            used_label = target_labels
            used_coef = -1
        else:
            used_label = ori_labels
            used_coef = 1
        
        if self.random_start:
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.epsilon, self.epsilon)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
            delta = torch.tensor((adv_images - images).clone().detach(), requires_grad=True).cuda()
        else:
            delta = torch.zeros_like(adv_images,requires_grad=True).cuda()

        result_matrix = np.zeros((len(self.eval_models), len(self.eval_steps)))
        iter_flag = 0
        
        inner_momentum = torch.zeros_like(images)
        self.outer_momentum = torch.zeros_like(images)
        self.mu = 1
        for itr in range(self.steps):
            before_delta = delta.data.clone()
            if self.Ours:
                self._initilize_gra_act()

            # data augmentation.
            if self.SU:
                # print ('Runing with SU')
                self.activations = []
                li_inputs = self.local_transform(adv_images)
                used_adv = torch.cat([adv_images+delta, li_inputs+delta], dim=0)
                loss_label = torch.cat([used_label, used_label], dim=0)
            elif self.RAP:
                # print ('Runing with RAP.')
                pert = self._pgd_adv_example(self.model_list, adv_images+delta.detach().clone(), used_label)
                used_adv = adv_images + delta + pert
                loss_label = used_label
            elif self.SI:
                # print ('Runing with SI')
                used_adv = adv_images + delta
                used_adv = self._si(used_adv)
                loss_label = torch.cat([used_label]*int(used_adv.shape[0]/b))
            elif self.Ours:
                # print ('Runing with Ours')
                used_adv = adv_images + delta
                used_adv = self._our_linear(used_adv, itr)
                loss_label = torch.cat([used_label]*int(used_adv.shape[0]/b))
            else:
                used_adv = adv_images + delta
                loss_label = used_label

            logits = 0
            for model in self.model_list:
                logits += model(used_adv)
            
            # classifiy loss
            if self.Ours:
                sum_loss = self._attention_map(logits, delta, loss_label)
                eff_all = torch.ones_like(sum_loss)
                classify_loss = -torch.sum(eff_all * sum_loss)
                # print ('loss for ours', classify_loss)
            else:
                classify_loss = self.loss_fn(logits, loss_label)
                # print ('loss for normal', classify_loss)

            # additional loss
            if self.SU:
                fs_losses = []
                for feature_ind in range(len(self.activations)):
                    fs_loss = torch.nn.functional.cosine_similarity(self.activations[feature_ind][:b].view(b, -1), self.activations[feature_ind][-b:].view(b, -1))
                    fs_loss = torch.mean(fs_loss)
                    fs_losses.append(fs_loss)
                fs_loss = torch.mean(torch.stack(fs_losses))
                loss = classify_loss + 0.001 * -1 * fs_loss
                # print ('loss for su', fs_loss)
            else:
                loss = classify_loss
                
            loss.backward()
            grad_ensemble = delta.grad.clone()
            delta.grad.zero_()
            delta.data += self.reverse_step_size * grad_ensemble.sign() # Increase the loss
            delta.data = delta.data.clamp(-self.epsilon, self.epsilon) 
            delta.data = ((adv_images + delta.data).clamp(0,1)) - adv_images
            
            # individual updating
            for model_name, model in zip(self.model_name, self.model_list):
                if self.SU:
                    self.activations = []
                if self.Ours:
                    self._initilize_gra_act()

                # data augmentation.
                if self.SU:
                    # print ('Runing with SU')
                    self.activations = []
                    li_inputs = self.local_transform(adv_images)
                    used_adv = torch.cat([adv_images+delta, li_inputs+delta], dim=0)
                    loss_label = torch.cat([used_label, used_label], dim=0)
                elif self.RAP:
                    # print ('Runing with RAP.')
                    pert = self._pgd_adv_example(self.model_list, adv_images+delta.detach().clone(), used_label)
                    used_adv = adv_images + delta + pert
                    loss_label = used_label
                elif self.SI:
                    # print ('Runing with SI')
                    used_adv = adv_images + delta
                    used_adv = self._si(used_adv)
                    loss_label = torch.cat([used_label]*int(used_adv.shape[0]/b))
                elif self.Ours:
                    # print ('+'*100)
                    # print ('Runing with Ours')
                    used_adv = adv_images + delta
                    used_adv = self._our_linear(used_adv, itr)
                    loss_label = torch.cat([used_label]*int(used_adv.shape[0]/b))
                else:
                    used_adv = adv_images + delta
                    loss_label = used_label

                # using the outer data augmentation
                logits = model(used_adv)
                # logits = model(adv_images+delta)
                # print (self.activations.keys())

                if self.Ours:
                    sum_loss = self._single_model_attention_map(model_name, logits, delta, loss_label)
                    eff_all = torch.ones_like(sum_loss)
                    classify_loss = -torch.sum(eff_all * sum_loss)
                    # print ('*'*100)
                    # print ('second', classify_loss)
                else:
                    classify_loss = self.loss_fn(logits, loss_label)

                if self.SU:
                    fs_losses = []
                    for feature_ind in range(len(self.activations)):
                        fs_loss = torch.nn.functional.cosine_similarity(self.activations[feature_ind][:b].view(b, -1), self.activations[feature_ind][-b:].view(b, -1))
                        fs_loss = torch.mean(fs_loss)
                        fs_losses.append(fs_loss)
                    fs_loss = torch.mean(torch.stack(fs_losses))
                    loss = classify_loss + 0.001 * -1 * fs_loss
                else:
                    loss = classify_loss
                
                loss.backward()
                grad_individual = delta.grad.clone()
                delta.grad.zero_()
                inner_momentum = self.mu * inner_momentum - grad_individual / torch.norm(grad_individual.reshape(N, -1), p=2, dim=1).view(N, 1, 1, 1)
                delta.data += self.inner_step_size * inner_momentum
                delta.data = delta.data.clamp(-self.epsilon, self.epsilon)
                delta.data = ((adv_images + delta.data).clamp(0,1)) - adv_images
            
            # outer momentum

            fake_grad = delta.data - before_delta
            self.outer_momentum = self.mu * self.outer_momentum + fake_grad / torch.norm(fake_grad, p=1)
            update_delta = self.step_size * self.outer_momentum.sign()
            delta.data = before_delta + update_delta
            delta.data = delta.data.clamp(-self.epsilon, self.epsilon)
            delta.data = ((adv_images + delta.data).clamp(0,1)) - adv_images

            torch.cuda.empty_cache()

            if itr+1 in self.eval_steps:
                for m_id, model in enumerate(self.eval_models):
                    with torch.no_grad():
                        this_logit = model(adv_images+delta)
                    if self.target:
                        success_nums = torch.sum(torch.argmax(this_logit, dim=1) == used_label).cpu().item()
                    else:
                        success_nums = torch.sum(torch.argmax(this_logit, dim=1) != used_label).cpu().item()
                    result_matrix[m_id, iter_flag] = success_nums
                iter_flag += 1
        return result_matrix, delta.data