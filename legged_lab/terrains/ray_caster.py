from isaaclab.sensors.ray_caster import RayCaster as BaseRayCaster
from collections.abc import Sequence


class RayCaster(BaseRayCaster):
    def reset(self, env_ids: Sequence[int] | None = None):
        # reset the timers and counters
        super().reset(env_ids)
        # resolve None
        if env_ids is None:
            env_ids = slice(None)
        # resample the drift
        self.drift[env_ids, 0] = self.drift[env_ids, 0].uniform_(*self.cfg.drift_range)
        self.drift[env_ids, 1] = self.drift[env_ids, 1].uniform_(*self.cfg.drift_range)
        self.drift[env_ids, 2] = self.drift[env_ids, 2].uniform_(
            *(self.cfg.drift_range[0] * 0.1, self.cfg.drift_range[1] * 0.1)
        )
