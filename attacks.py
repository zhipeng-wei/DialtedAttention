import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import scipy.stats as st
import torchvision

from losses import LogitLoss

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

    def _DI(self, X_in, di_prob=0.7):
        rnd = np.random.randint(299, 330,size=1)[0]
        h_rem = 330 - rnd
        w_rem = 330 - rnd
        pad_top = np.random.randint(0, h_rem,size=1)[0]
        pad_bottom = h_rem - pad_top
        pad_left = np.random.randint(0, w_rem,size=1)[0]
        pad_right = w_rem - pad_left

        c = np.random.rand(1)
        if c <= di_prob:
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
        # X_out = F.interpolate(X_out, size=(X_in.shape[-2], X_in.shape[-1]))
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
    
    def _target_layer(self, model_name, depth):
        '''
        'inception_v3', 'resnet50', 'densenet121', 'vgg16_bn'
        depth: [1, 2, 3, 4]
        '''
        if model_name == 'resnet50':
            return getattr(self.model[1], 'layer{}'.format(depth))[-1]
        elif model_name == 'vgg16_bn':
            depth_to_layer = {1:12,2:22,3:32,4:42}
            return getattr(self.model[1], 'features')[depth_to_layer[depth]]
        elif model_name == 'densenet121':
            return getattr(getattr(self.model[1], 'features'), 'denseblock{}'.format(depth))
        elif model_name == 'inception_v3':
            depth_to_layer = {1:'Conv2d_4a_3x3', 2:'Mixed_5d', 3:'Mixed_6e', 4:'Mixed_7c'}
            return getattr(self.model[1], '{}'.format(depth_to_layer[depth]))

    def perturb(self, images, ori_labels, target_labels):
        raise NotImplementedError

class DTMI(Base):
    '''
    Use TI, MI and DI as the baseline attack.
    '''
    def __init__(self, prob, model, eval_models, loss_fn, eval_steps, steps, target, DI=False, TI=False, MI=False):
        super(DTMI, self).__init__(loss_fn, eval_steps, steps, target)
        self.prob = prob
        self.model = model
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
                logits = self.model(self._DI(adv_images+delta, di_prob=self.prob))
                loss_label = used_label
            else:
                logits = self.model(adv_images+delta)
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

class DTMI_Local(Base):
    '''
    Incorporate locality of images.
    '''
    def __init__(self, scale, model, eval_models, loss_fn, eval_steps, steps, target, DI=False, TI=False, MI=False):
        super(DTMI_Local, self).__init__(loss_fn, eval_steps, steps, target)
        self.start, self.interval = scale
        self.model = model
        self.eval_models = eval_models

        self.DI, self.TI, self.MI = DI, TI, MI
        if self.TI:
            self.gaussian_kernel = self._TI_kernel()

        self.local_transform = torchvision.transforms.RandomResizedCrop(299, scale=(self.start, self.start+self.interval))
    
    def _LI_WO_prob(self, X_in):
        return self.local_transform(X_in)

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
            delta = torch.Tensor(adv_images - images, requires_grad=True).cuda()
        else:
            delta = torch.zeros_like(adv_images,requires_grad=True).cuda()

        result_matrix = np.zeros((len(self.eval_models), len(self.eval_steps)))
        iter_flag = 0
        if self.MI:
            grad_pre = 0
        for itr in range(self.steps):
            if self.DI:
                li_inputs = self._LI_WO_prob(adv_images)
                accom_inputs = torch.cat([adv_images+delta, li_inputs+delta], dim=0)
                logits = self.model(self._DI(accom_inputs))
                loss_label = torch.cat([used_label, used_label], dim=0)
            else:
                li_inputs = self._LI_WO_prob(adv_images)
                accom_inputs = torch.cat([adv_images+delta, li_inputs+delta], dim=0)
                logits = self.model(accom_inputs)
                loss_label = torch.cat([used_label, used_label], dim=0)
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

class DTMI_Random(Base):
    '''
    Incorporate random noise.
    '''
    def __init__(self, scale, model, eval_models, loss_fn, eval_steps, steps, target, DI=False, TI=False, MI=False):
        super(DTMI_Random, self).__init__(loss_fn, eval_steps, steps, target)
        self.start, self.interval = scale
        self.model = model
        self.eval_models = eval_models

        self.DI, self.TI, self.MI = DI, TI, MI
        if self.TI:
            self.gaussian_kernel = self._TI_kernel()

    def _random_noise(self, X_in):
        return torch.empty_like(X_in).uniform_(0, 1)

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
            delta = torch.Tensor(adv_images - images, requires_grad=True).cuda()
        else:
            delta = torch.zeros_like(adv_images,requires_grad=True).cuda()

        result_matrix = np.zeros((len(self.eval_models), len(self.eval_steps)))
        iter_flag = 0
        if self.MI:
            grad_pre = 0
        for itr in range(self.steps):
            if self.DI:
                li_inputs = self._random_noise(adv_images)
                accom_inputs = torch.cat([adv_images+delta, li_inputs+delta], dim=0)
                logits = self.model(self._DI(accom_inputs))
                loss_label = torch.cat([used_label, used_label], dim=0)
            else:
                logits = self.model(adv_images+delta)
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

class DTMI_Local_FeatureSimilarityLoss(Base):
    '''
    Incorporate locality of images and the feature similarity loss between global features and local features.
    '''
    def __init__(self, model_name, depth, coef, scale, model, eval_models, loss_fn, eval_steps, steps, target, DI=False, TI=False, MI=False):
        super(DTMI_Local_FeatureSimilarityLoss, self).__init__(loss_fn, eval_steps, steps, target)
        self.start, self.interval = scale
        self.coef = coef
        self.model_name = model_name
        self.depth = depth
        self.model = model
        self.eval_models = eval_models

        self.DI, self.TI, self.MI = DI, TI, MI
        if self.TI:
            self.gaussian_kernel = self._TI_kernel()

        self.local_transform = torchvision.transforms.RandomResizedCrop(299, scale=(self.start, self.start+self.interval))
        self._register_forward()

    def _register_forward(self):
        '''
        'inception_v3', 'resnet50', 'densenet121', 'vgg16_bn'
        '''
        self.activations = []
        def forward_hook(module, input, output):
            self.activations += [output]
            return None
        target_layer = self._target_layer(self.model_name, self.depth)    
        target_layer.register_forward_hook(forward_hook)

    def _LI_WO_prob(self, X_in):
        return self.local_transform(X_in)

    def perturb(self, images, ori_labels, target_labels):
        adv_images = images.clone()
        batch_size = adv_images.shape[0]
        if self.target:
            used_label = target_labels
            used_coef = -1
        else:
            used_label = ori_labels
            used_coef = 1
        
        if self.random_start:
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.epsilon, self.epsilon)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
            delta = torch.Tensor(adv_images - images, requires_grad=True).cuda()
        else:
            delta = torch.zeros_like(adv_images,requires_grad=True).cuda()

        result_matrix = np.zeros((len(self.eval_models), len(self.eval_steps)))
        iter_flag = 0
        if self.MI:
            grad_pre = 0
        for itr in range(self.steps):
            self.activations = []
            if self.DI:
                li_inputs = self._LI_WO_prob(adv_images)
                accom_inputs = torch.cat([adv_images+delta, li_inputs+delta], dim=0)
                logits = self.model(self._DI(accom_inputs))
                loss_label = torch.cat([used_label, used_label], dim=0)
            else:
                li_inputs = self._LI_WO_prob(adv_images)
                accom_inputs = torch.cat([adv_images+delta, li_inputs+delta], dim=0)
                logits = self.model(accom_inputs)
                loss_label = torch.cat([used_label, used_label], dim=0)
            
            classifier_loss = self.loss_fn(logits, loss_label)
            
            fs_loss = torch.nn.functional.cosine_similarity(self.activations[0][:batch_size].view(batch_size, -1), self.activations[0][-batch_size:].view(batch_size, -1))
            fs_loss = torch.mean(fs_loss)
            loss = classifier_loss + self.coef * used_coef * fs_loss
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

#*********************************************
#             The split line
#*********************************************
class DIWithDifferentUpperBound(Base):
    '''
    Perform DI with differnt upper bounds. 
    When the upper bound = 0, it degrades to I-FGSM.
    '''
    def __init__(self, upper_bound, model, eval_models, loss_fn, eval_steps, steps, target):
        super(DIWithDifferentUpperBound, self).__init__(loss_fn, eval_steps, steps, target)
        self.model = model
        self.eval_models = eval_models
        self.upper_bound = upper_bound
    
    def _DI_With_UpperBound(self, X_in):
        if self.upper_bound == 0:
            X_out = X_in.clone()
        else:
            rnd = np.random.randint(299, self.upper_bound, size=1)[0]
            h_rem = self.upper_bound - rnd
            w_rem = self.upper_bound - rnd
            pad_top = np.random.randint(0, h_rem,size=1)[0]
            pad_bottom = h_rem - pad_top
            pad_left = np.random.randint(0, w_rem,size=1)[0]
            pad_right = w_rem - pad_left
            X_out = F.pad(F.interpolate(X_in, size=(rnd,rnd)),(pad_left,pad_top,pad_right,pad_bottom),mode='constant', value=0)
        return  X_out

    def perturb(self, images, ori_labels, target_labels):
        adv_images = images.clone()
        batch_size = images.shape[0]
        if self.target:
            used_label = target_labels
            used_coef = -1
        else:
            used_label = ori_labels
            used_coef = 1
        
        if self.random_start:
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.epsilon, self.epsilon)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
            delta = torch.Tensor(adv_images - images, requires_grad=True).cuda()
        else:
            delta = torch.zeros_like(adv_images,requires_grad=True).cuda()

        result_matrix = np.zeros((len(self.eval_models), len(self.eval_steps), batch_size))
        norm_matrix = np.zeros((self.steps, batch_size))
        loss_matrix = np.zeros((self.steps))

        iter_flag = 0
        for itr in range(self.steps):
            di_inputs = self._DI_With_UpperBound(adv_images+delta)
            logits = self.model(di_inputs)
            loss_label = used_label
            loss = self.loss_fn(logits, loss_label)
            loss.backward()
            grad_c = delta.grad.clone()
            grad_a = grad_c.clone()

            loss_matrix[itr] = loss.detach().cpu().item()
            grad_norm = torch.norm(grad_a.view(batch_size, -1), dim=1).cpu().numpy()
            norm_matrix[itr] = grad_norm

            delta.grad.zero_()
            delta.data = delta.data + used_coef * self.alpha * torch.sign(grad_a)
            delta.data = delta.data.clamp(-self.epsilon, self.epsilon) 
            delta.data = ((adv_images + delta.data).clamp(0,1)) - adv_images
            if itr+1 in self.eval_steps:
                for m_id, model in enumerate(self.eval_models):
                    with torch.no_grad():
                        this_logit = model(adv_images+delta)
                    if self.target:
                        # success_nums = torch.sum(torch.argmax(this_logit, dim=1) == used_label).cpu().item()
                        success_info = (torch.argmax(this_logit, dim=1) == used_label).type(torch.int).cpu().numpy()
                    else:
                        # success_nums = torch.sum(torch.argmax(this_logit, dim=1) != used_label).cpu().item()
                        success_info = (torch.argmax(this_logit, dim=1) != used_label).type(torch.int).cpu().numpy()
                    result_matrix[m_id, iter_flag] = success_info
                iter_flag += 1
        return result_matrix, delta.data, norm_matrix, loss_matrix

class DI_wo_prob(Base):
    '''
    Use TI, MI and DI as the baseline attack.
    '''
    def __init__(self, model, eval_models, loss_fn, eval_steps, steps, target, DI=True, TI=False, MI=False):
        super(DI_wo_prob, self).__init__(loss_fn, eval_steps, steps, target)
        self.model = model
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
                logits = self.model(self._DI_WO_Prob(adv_images+delta))
                loss_label = used_label
            else:
                logits = self.model(adv_images+delta)
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


class DI_logits_vary(Base):
    '''
    Use TI, MI and DI as the baseline attack.
    '''
    def __init__(self, model, eval_models, loss_fn, eval_steps, steps, target, DI=True, TI=False, MI=False):
        super(DI_logits_vary, self).__init__(loss_fn, eval_steps, steps, target)
        self.model = model
        self.eval_models = [model] + eval_models

        self.DI, self.TI, self.MI = DI, TI, MI
        if self.TI:
            self.gaussian_kernel = self._TI_kernel()

    def perturb(self, images, ori_labels, target_labels):
        adv_images = images.clone()
        batch_size = adv_images.shape[0]
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
        logits_info = np.zeros((len(self.eval_models), len(self.eval_steps), batch_size, 1000))
        iter_flag = 0
        if self.MI:
            grad_pre = 0
        for itr in range(self.steps):
            if self.DI:
                logits = self.model(self._DI_WO_Prob(adv_images+delta))
                loss_label = used_label
            else:
                logits = self.model(adv_images+delta)
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
                    logits_info[m_id, iter_flag] = this_logit.cpu().numpy()
                    if self.target:
                        success_nums = torch.sum(torch.argmax(this_logit, dim=1) == used_label).cpu().item()
                    else:
                        success_nums = torch.sum(torch.argmax(this_logit, dim=1) != used_label).cpu().item()
                    result_matrix[m_id, iter_flag] = success_nums
                iter_flag += 1
        return result_matrix, delta.data, logits_info
