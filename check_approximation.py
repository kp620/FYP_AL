import time 
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset



def _get_quadratic_approximation(self, coreset, weights):
    """
    Compute the quadratic approximation of the loss function
    :param epoch: current epoch
    :param training_step: current training step
    """

    # 1. Load coreset: 
    approx_loader = DataLoader(
                Subset(self.train_dataset, indices=coreset),
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                pin_memory=True,
                sampler=None,
            )

    self.start_loss = 0

    for approx_batch, (input, target, idx) in enumerate(approx_loader):
        target = target.cuda()
        input_var = input.cuda()
        target_var = target

        # train coresets(with weights)
        output = self.model(input_var)
            
        if self.args.approx_with_coreset:
            loss = self.train_criterion(output, target_var)
            batch_weight = weights[idx.long()]
            loss = (loss * batch_weight).mean()
        else:
            loss = self.val_criterion(output, target_var)
        self.model.zero_grad()

        # approximate with hessian diagonal
        loss.backward(create_graph=True)
        gf_tmp, ggf_tmp, ggf_tmp_moment = self.gradient_approx_optimizer.step(momentum=True)

        if approx_batch == 0:
            self.gf = gf_tmp * len(idx)
            self.ggf = ggf_tmp * len(idx)
            self.ggf_moment = ggf_tmp_moment * len(idx)
        else:
            self.gf += gf_tmp * len(idx)
            self.ggf += ggf_tmp * len(idx)
            self.ggf_moment += ggf_tmp_moment * len(idx)

        #TODO: update start_loss

    self.gf /= len(approx_loader.dataset)
    self.ggf /= len(approx_loader.dataset)
    self.ggf_moment /= len(approx_loader.dataset)





def _check_approx_error(self) -> torch.Tensor:
    """
    Check the approximation error of the current batch
    :param epoch: current epoch
    :param training_step: current training step
    """
    
    # calculate true loss: TODO
    
    # calculate delta: TODO


    approx_loss = torch.matmul(self.delta, self.gf) + self.start_loss
    approx_loss += 1 / 2 * torch.matmul(self.delta * self.ggf, self.delta)

    loss_diff = abs(true_loss - approx_loss.item())
    thresh = self.args.check_thresh_factor * true_loss


        
    if loss_diff > thresh:
        self.reset_step = training_step
        log_str += f" is larger than threshold {thresh:.3f}. "
    self.args.logger.info(log_str)


