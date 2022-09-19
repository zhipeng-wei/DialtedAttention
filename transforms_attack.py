import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import scipy.stats as st
import torchvision
import random

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
            self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        elif loss_fn == 'Logit':
            self.loss_fn = LogitLoss(reduction='none')
    
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

class SingleTransormsAttack(Base):
    '''
    Use TI, MI and DI as the baseline attack.
    '''
    def __init__(self, transform_func, model, eval_models, loss_fn, eval_steps, steps, target):
        super(SingleTransormsAttack, self).__init__(loss_fn, eval_steps, steps, target)
        self.transform_func = transform_func
        self.model = model
        self.eval_models = eval_models

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
            delta = torch.tensor((adv_images - images).clone().detach(), requires_grad=True).cuda()
        else:
            delta = torch.zeros_like(adv_images,requires_grad=True).cuda()

        result_matrix = np.zeros((len(self.eval_models), len(self.eval_steps)))        
        norm_matrix = np.zeros((self.steps, batch_size))
        loss_matrix = np.zeros((self.steps, batch_size))
        iter_flag = 0
        for itr in range(self.steps):            
            logits = self.model(self.transform_func(adv_images+delta))
            loss_label = used_label
            loss = self.loss_fn(logits, loss_label)
            mean_loss = torch.mean(loss)
            mean_loss.backward()

            grad_c = delta.grad.clone()
            grad_a = grad_c.clone()

            loss_matrix[itr] = loss.detach().cpu().numpy()

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
                        success_nums = torch.sum(torch.argmax(this_logit, dim=1) == used_label).cpu().item()
                    else:
                        success_nums = torch.sum(torch.argmax(this_logit, dim=1) != used_label).cpu().item()
                    result_matrix[m_id, iter_flag] = success_nums
                iter_flag += 1            
        return result_matrix, delta.data, norm_matrix, loss_matrix

class SingleTransormsAttack_TMI(Base):
    '''
    Use TI, MI and DI as the baseline attack.
    '''
    def __init__(self, transform_func, model, eval_models, loss_fn, eval_steps, steps, target):
        super(SingleTransormsAttack_TMI, self).__init__(loss_fn, eval_steps, steps, target)
        self.transform_func = transform_func
        self.model = model
        self.eval_models = eval_models

        self.gaussian_kernel = self._TI_kernel()
        
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
            delta = torch.tensor((adv_images - images).clone().detach(), requires_grad=True).cuda()
        else:
            delta = torch.zeros_like(adv_images,requires_grad=True).cuda()

        grad_pre = 0
        result_matrix = np.zeros((len(self.eval_models), len(self.eval_steps)))        
        norm_matrix = np.zeros((self.steps, batch_size))
        loss_matrix = np.zeros((self.steps, batch_size))
        iter_flag = 0
        for itr in range(self.steps):            
            logits = self.model(self.transform_func(adv_images+delta))
            loss_label = used_label
            loss = self.loss_fn(logits, loss_label)
            mean_loss = torch.mean(loss)
            mean_loss.backward()

            grad_c = delta.grad.clone()
            grad_c = F.conv2d(grad_c, self.gaussian_kernel, bias=None, stride=1, padding=(2,2), groups=3)
            grad_a = grad_c / torch.mean(torch.abs(grad_c), (1, 2, 3), keepdim=True) + 1 * grad_pre
            grad_pre = grad_a

            loss_matrix[itr] = loss.detach().cpu().numpy()

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
                        success_nums = torch.sum(torch.argmax(this_logit, dim=1) == used_label).cpu().item()
                    else:
                        success_nums = torch.sum(torch.argmax(this_logit, dim=1) != used_label).cpu().item()
                    result_matrix[m_id, iter_flag] = success_nums
                iter_flag += 1            
        return result_matrix, delta.data, norm_matrix, loss_matrix


class SingleTransormsAttack_LogitSave(Base):
    '''
    Use TI, MI and DI as the baseline attack.
    '''
    def __init__(self, transform_func, model, eval_models, loss_fn, eval_steps, steps, target):
        super(SingleTransormsAttack_LogitSave, self).__init__(loss_fn, eval_steps, steps, target)
        self.transform_func = transform_func
        self.model = model
        self.eval_models = [model] + eval_models

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
            delta = torch.tensor((adv_images - images).clone().detach(), requires_grad=True).cuda()
        else:
            delta = torch.zeros_like(adv_images,requires_grad=True).cuda()

        result_matrix = np.zeros((len(self.eval_models), len(self.eval_steps)))        
        norm_matrix = np.zeros((self.steps, batch_size))
        loss_matrix = np.zeros((self.steps, batch_size))
        logits_info = np.zeros((len(self.eval_models), len(self.eval_steps), batch_size, 1000))
        iter_flag = 0
        for itr in range(self.steps):            
            logits = self.model(self.transform_func(adv_images+delta))
            loss_label = used_label
            loss = self.loss_fn(logits, loss_label)
            mean_loss = torch.mean(loss)
            mean_loss.backward()

            grad_c = delta.grad.clone()
            grad_a = grad_c.clone()

            loss_matrix[itr] = loss.detach().cpu().numpy()

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
                    logits_info[m_id, iter_flag] = this_logit.cpu().numpy()
                    if self.target:
                        success_nums = torch.sum(torch.argmax(this_logit, dim=1) == used_label).cpu().item()
                    else:
                        success_nums = torch.sum(torch.argmax(this_logit, dim=1) != used_label).cpu().item()
                    result_matrix[m_id, iter_flag] = success_nums
                iter_flag += 1            
        return result_matrix, delta.data, norm_matrix, loss_matrix, logits_info

class SingleTransormsAttack_TMI_LogitSave(Base):
    '''
    Use TI, MI and DI as the baseline attack.
    '''
    def __init__(self, transform_func, model, eval_models, loss_fn, eval_steps, steps, target):
        super(SingleTransormsAttack_TMI_LogitSave, self).__init__(loss_fn, eval_steps, steps, target)
        self.transform_func = transform_func
        self.model = model
        self.eval_models = [model] + eval_models

        self.gaussian_kernel = self._TI_kernel()
        
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
            delta = torch.tensor((adv_images - images).clone().detach(), requires_grad=True).cuda()
        else:
            delta = torch.zeros_like(adv_images,requires_grad=True).cuda()

        grad_pre = 0
        result_matrix = np.zeros((len(self.eval_models), len(self.eval_steps)))        
        norm_matrix = np.zeros((self.steps, batch_size))
        loss_matrix = np.zeros((self.steps, batch_size))
        logits_info = np.zeros((len(self.eval_models), len(self.eval_steps), batch_size, 1000))
        iter_flag = 0
        for itr in range(self.steps):            
            logits = self.model(self.transform_func(adv_images+delta))
            loss_label = used_label
            loss = self.loss_fn(logits, loss_label)
            mean_loss = torch.mean(loss)
            mean_loss.backward()

            grad_c = delta.grad.clone()
            grad_c = F.conv2d(grad_c, self.gaussian_kernel, bias=None, stride=1, padding=(2,2), groups=3)
            grad_a = grad_c / torch.mean(torch.abs(grad_c), (1, 2, 3), keepdim=True) + 1 * grad_pre
            grad_pre = grad_a

            loss_matrix[itr] = loss.detach().cpu().numpy()

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
                    logits_info[m_id, iter_flag] = this_logit.cpu().numpy()
                    if self.target:
                        success_nums = torch.sum(torch.argmax(this_logit, dim=1) == used_label).cpu().item()
                    else:
                        success_nums = torch.sum(torch.argmax(this_logit, dim=1) != used_label).cpu().item()
                    result_matrix[m_id, iter_flag] = success_nums
                iter_flag += 1            
        return result_matrix, delta.data, norm_matrix, loss_matrix, logits_info