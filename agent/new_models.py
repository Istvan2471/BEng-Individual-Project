import torch
from torch import jit, nn
from torch.nn import functional as F
import torch.distributions
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from agent.models import SampleDist, TanhBijector
from gfn.modules import GFNModule

class GFlowNetWrapperModel(jit.ScriptModule):
  def __init__(self, belief_size, state_size, hidden_size, action_size, dist='tanh_normal',
                activation_function='elu', min_std=1e-4, init_std=5, mean_scale=5):
    super().__init__()
    self.module = GFNModule()

  @jit.script_method
  def forward(self, belief, state):
    self.module.forward(belief, state)

  def get_action(self, belief, state, det=False):
    action_mean, action_std = self.forward(belief, state)
    dist = Normal(action_mean, action_std)
    dist = TransformedDistribution(dist, TanhBijector())
    dist = torch.distributions.Independent(dist,1)
    dist = SampleDist(dist)
    if det: return dist.mode()
    else: return dist.rsample()