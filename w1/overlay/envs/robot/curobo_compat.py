"""Compatibility fix for the bundled cuRobo matrix self-collision kernel.

The fallback kernel can leave a positive distance/gradient in reused output
buffers when the following configuration has no collision. Clear its outputs
on every evaluation, including CUDA graph capture. Collision pairs, radii,
weights and thresholds are unchanged.
"""


def clear_fallback_self_collision_outputs():
    import torch
    from curobo.rollout.cost.self_collision_cost import SelfCollisionCost
    original=SelfCollisionCost.forward
    if getattr(original,"_w1_clears_fallback_outputs",False):return

    def forward(self,robot_spheres):
        self.update_batch_size(robot_spheres)
        if not self.self_collision_kin_config.experimental_kernel:
            with torch.no_grad():
                self._out_distance.zero_()
                self._out_vec.zero_()
                self._sparse_sphere_idx.zero_()
        return original(self,robot_spheres)

    forward._w1_clears_fallback_outputs=True
    SelfCollisionCost.forward=forward
