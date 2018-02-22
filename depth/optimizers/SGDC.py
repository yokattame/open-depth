import torch

class SGDC(torch.optim.SGD):
  
  '''
  ATTENTION: lr is a required parameter.
  '''

  def __init__(self, params, lr, momentum=0, dampening=0, weight_decay=0, neterov=Fase, gradient_threshold=None):
    super().__init__(params, lr, momentum, dampening, weight_decay, nesterov)
    self.gradient_threshold = gradient_threshold

  def step(self, closure=None):
    
    for param_group in self.param_groups:
      if self.gradient_threshold is not None:
        torch.nn.utils.clip_grad.clip_grad_norm(param_group['params'], self.gradient_threshold)
      else:
        print('Param norm:', torch.nn.utils.clip_grad.clip_grad_norm(param_group['params'], float('inf')))

    super().step(closure)

