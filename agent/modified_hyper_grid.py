from gfn.containers.states import States
from gfn.envs import HyperGrid
from gfn.envs.env import TensorLong, TensorBool, correct_cast, NonValidActionsError
from torch import clone, gather
from copy import deepcopy

class ModifiedHyperGrid(HyperGrid):
    def backward_step(self, states: States, actions: TensorLong) -> States:
        """Function that takes a batch of states and actions and returns a batch of next
        states and a boolean tensor indicating initial states in the new batch."""
        new_states = self.make_States_class(states.states_tensor.clone())
        valid_states: TensorBool = ~new_states.is_initial_state
        valid_actions = actions[valid_states]

        if new_states.backward_masks is not None:
            _, new_backward_masks = correct_cast(
                new_states.forward_masks, new_states.backward_masks
            )
            valid_states_masks = new_backward_masks[valid_states]
            valid_actions_bool = all(
                gather(valid_states_masks, 1, valid_actions.unsqueeze(1))
            )
            if not valid_actions_bool:
                raise NonValidActionsError("Actions are not valid")

        not_done_states = new_states.states_tensor[valid_states]
        self.maskless_backward_step(not_done_states, valid_actions)

        new_states.states_tensor[valid_states] = not_done_states

        new_states.update_masks()
        return new_states