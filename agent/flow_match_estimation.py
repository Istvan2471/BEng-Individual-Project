from dataclasses import dataclass
from typing import Tuple

import torch
from torchtyping import TensorType

from gfn.containers.states import correct_cast
from agent.modified_states import States
from agent.modified_hyper_grid import ModifiedHyperGrid
from gfn.distributions import EmpiricalTrajectoryDistribution, TrajectoryDistribution
from gfn.envs import Env, HyperGrid
from gfn.estimators import LogEdgeFlowEstimator
from gfn.losses.base import Parametrization, StateDecomposableLoss
from gfn.losses import FMParametrization
from gfn.samplers import DiscreteActionsSampler, TrajectoriesSampler
from gfn.modules import NeuralNet

ScoresTensor = TensorType["n_states", float]
LossTensor = TensorType[0, float]

def estimate_fm_value(states, imged_rewards):
    env = ModifiedHyperGrid(ndim=2, height=64, device_str="cuda", preprocessor_name="Identity")
    estimator = LogEdgeFlowEstimator(env=env, module_name="NeuralNet")
    parametrization = FMParametrization(logF=estimator)
    loss_fn = FlowMatching(parametrization=parametrization)
    states_container = env.make_States_class()(states_tensor=states.to(torch.long))
    losses = loss_fn(states_container, imged_rewards)
    return losses

class FlowMatching(StateDecomposableLoss):
    def __init__(self, parametrization: FMParametrization, alpha=1.0) -> None:
        "alpha is the weight of the reward matching loss"
        self.parametrization = parametrization
        self.env = parametrization.logF.env
        self.alpha = alpha

    def flow_matching_loss(self, states: States) -> ScoresTensor:
        """
        Compute the FM for the given states, defined as the log-sum incoming flows minus log-sum outgoing flows.
        The states should not include s0. The batch shape should be (n_states,).
        As of now, only discrete environments are handled.
        """

        assert len(states.batch_shape) == 1
        # assert not torch.any(states.is_initial_state)

        states.forward_masks, states.backward_masks = correct_cast(
            states.forward_masks, states.backward_masks
        )

        incoming_log_flows = torch.full_like(
            states.backward_masks, -float("inf"), dtype=torch.float
        )
        outgoing_log_flows = torch.full_like(
            states.forward_masks, -float("inf"), dtype=torch.float
        )

        for action_idx in range(self.env.n_actions - 1):
            # TODO: can this be done in a vectorized way? Maybe by "repeating" the states and creating a big actions tensor?
            valid_backward_mask = states.backward_masks[:, action_idx]
            valid_forward_mask = states.forward_masks[:, action_idx]
            valid_backward_states = states[valid_backward_mask]
            valid_forward_states = states[valid_forward_mask]
            _, valid_backward_states.backward_masks = correct_cast(
                valid_backward_states.forward_masks,
                valid_backward_states.backward_masks,
            )
            backward_actions = torch.full_like(
                valid_backward_states.backward_masks[:, 0], action_idx, dtype=torch.long
            )

            valid_backward_states_parents = self.env.backward_step(
                valid_backward_states, backward_actions
            )

            incoming_log_flows[
                valid_backward_mask, action_idx
            ] = self.parametrization.logF(valid_backward_states_parents)[:, action_idx]
            outgoing_log_flows[
                valid_forward_mask, action_idx
            ] = self.parametrization.logF(valid_forward_states)[:, action_idx]

        # Now the exit action
        valid_forward_mask = states.forward_masks[:, -1]
        outgoing_log_flows[valid_forward_mask, -1] = self.parametrization.logF(
            states[valid_forward_mask]
        )[:, -1]

        log_incoming_flows = torch.logsumexp(incoming_log_flows, dim=-1)
        log_outgoing_flows = torch.logsumexp(outgoing_log_flows, dim=-1)

        return log_incoming_flows - log_outgoing_flows

    def reward_matching_loss(self, terminating_states: States, rewards) -> LossTensor:
        log_edge_flows = self.parametrization.logF(terminating_states)
        terminating_log_edge_flows = log_edge_flows[:, -1]
        log_rewards = torch.nan_to_num(torch.log(rewards), nan=0.0, neginf=-1000000.0)
        return terminating_log_edge_flows - log_rewards

    def __call__(self, states: States, rewards) -> LossTensor:
        fm_loss = self.flow_matching_loss(states)
        rm_loss = self.reward_matching_loss(states, rewards)
        return fm_loss + self.alpha * rm_loss