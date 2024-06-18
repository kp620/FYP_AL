"""
CITE: Yao Z, Gholami A, Shen S, Mustafa M, Keutzer K, Mahoney M. Adahessian: An adaptive second order optimizer for machine learning. Inproceedings of the AAAI conference on artificial intelligence 2021 May 18 (Vol. 35, No. 12, pp. 10665-10673).
"""

import math
import torch
from torch.optim.optimizer import Optimizer

class Adahessian(Optimizer):
    """Approximates local gradients and 2nd order information in the same way as Adahessian does.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 0.15)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-4)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        hessian_power (float, optional): Hessian power (default: 1). You can also try 0.5. For some tasks we found this to result in better performance.
        single_gpu (Bool, optional): Do you use distributed training or not "torch.nn.parallel.DistributedDataParallel" (default: True)
    """

    def __init__(
        self,
        params,
        lr=0.15,
        betas=(0.9, 0.999),
        eps=1e-4,
        weight_decay=0,
        hessian_power=1,
        single_gpu=True,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= hessian_power <= 1.0:
            raise ValueError("Invalid Hessian power value: {}".format(hessian_power))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            hessian_power=hessian_power,
        )
        self.single_gpu = single_gpu
        super(Adahessian, self).__init__(params, defaults)

    def get_trace(self, params, grads):
        """
        compute the Hessian vector product with a random vector v, at the current gradient point,
        i.e., compute the gradient of <gradsH,v>.
        :param params: list of parameters
        :param grads: list of gradients
        :return: list of Hessian diagonal elements and list of reduced gradients
        """

        # Check backward was called with create_graph set to True
        # Ensure the gradients have been computed with the graph of operations that produced them
        # Typically done by calling the 'backward()' function with 'create_graph=True'
        # Necessary because the hessian is a second-order derivative, we need to have the computational graph that produced the gradients
        for i, grad in enumerate(grads):
            if grad.grad_fn is None:
                raise RuntimeError(
                    "Gradient tensor {:} does not have grad_fn. When calling\n".format(
                        i
                    )
                    + "\t\t\t  loss.backward(), make sure the option create_graph is\n"
                    + "\t\t\t  set to True."
                )

        # Random vector generation(same shape as the parameter)
        # This vector is used to perform a stochastic estimation of the Hessian
        v = [2 * torch.randint_like(p, high=2) - 1 for p in params]

        if not params or not grads:
            raise ValueError("params or grads are empty, ensure that model parameters require gradients and a backward pass has been executed.")
        # Calculates the gradient of the dot product <grads,v>, which gives the Hessian-vector product 'hv'
        # grad_outputs=v indicates we want to backpropagate through 'v'
        # only_inputs=True makes sure the gradients are computed only for the input(parameters)
        # retain_graph=True keeps the computation graph around after computing the gradients
        hvs = torch.autograd.grad(
            grads, params, grad_outputs=v, only_inputs=True, retain_graph=True
        )

        reduced_grads = []
        hutchinson_trace = []
        for grad, hv, vi in zip(grads, hvs, v):
          # Computes the element wise product of Hessian-vector product('hv') and the random vector('vi')
            tmp_output = hv * vi
          # Flattened gradient for that parameter
          # Turn the gradient tensors into a 1D vector, which is often required for optimizer updates
            tmp_grad = grad

            hutchinson_trace.append(tmp_output.detach())
            reduced_grads.append(tmp_grad.detach().flatten())

        # hutchison_trace: contains the estimated Hessian diagonals for all parameters
        # reduced_grad: contains the flattened gradients for all parameters
        return hutchinson_trace, reduced_grads
    
    def step(self, momentum=False):
        """Performs a single approximation step.
        Arguments:
            momentum (bool, optional): enables the momentum technique (default: False)
        """

        params = []
        groups = []
        grads = []


        # Works with parameter groups, which allows for different parameters to have different optimization settings

        # Flatten groups into lists, so that
        #  hut_traces can be called with lists of parameters
        #  and grads
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    params.append(p)
                    groups.append(group)
                    grads.append(p.grad)


        # get the Hessian diagonal

        hutchinson_trace_moment = []
        # Compute Hutchinson trace, which approximates the Hessian diagonal and reduced gradients
        if momentum:
            hut_traces, reduced_grads = self.get_trace(params, grads)
            reduced_grads = []
            # Initialize or update the state for each parameter
            for (p, group, grad, hut_trace) in zip(params, groups, grads, hut_traces):
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of Hessian diagonal square values
                    state["exp_hessian_diag_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_hessian_diag_sq = (
                    state["exp_avg"],
                    state["exp_hessian_diag_sq"],
                )


                # For each parameter, the function updates the state dictionary.
                # Involves incrementing the step count and computing the biased averages of the gradient and the hessian diagonal
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad.detach_(), alpha=1 - beta1)
                exp_hessian_diag_sq.mul_(beta2).addcmul_(hut_trace, hut_trace, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # The denominator of the parameter update
                # make the square root, and the Hessian power
                k = group['hessian_power']
                denom = (
                    (exp_hessian_diag_sq.sqrt() ** k) /
                    math.sqrt(bias_correction2) ** k).add_(
                    group['eps'])

                # Actual parameter update is computed by dividing the biased-corrected gradient by the computed denominator
                reduced_grads.append((exp_avg / bias_correction1).detach().flatten())
                hutchinson_trace_moment.append(denom.detach().flatten())
            hutchinson_trace_moment = torch.cat(hutchinson_trace_moment)
        else:
            reduced_grads = []
            hutchinson_trace = []
            for grad in grads:
                reduced_grads.append(grad.detach().flatten())
            reduced_grads = torch.cat(reduced_grads).detach()
            return reduced_grads, hutchinson_trace, hutchinson_trace_moment

        hutchinson_trace = []
        for hut_trace in hut_traces:
            hutchinson_trace.append(hut_trace.detach().flatten())

        hutchinson_trace = torch.cat(hutchinson_trace).detach()
        reduced_grads = torch.cat(reduced_grads).detach()
        return reduced_grads, hutchinson_trace, hutchinson_trace_moment