"""Keep the RSL-RL 3.x entry points usable with Isaac Lab 2.3 snapshots."""

from packaging import version

from isaaclab_rl import rsl_rl as lab_rsl_rl


def handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version):
    handler = getattr(lab_rsl_rl, "handle_deprecated_rsl_rl_cfg", None)
    if handler is not None:
        return handler(agent_cfg, installed_version)
    if version.parse(installed_version) >= version.parse("4.0.0"):
        raise RuntimeError("RSL-RL >=4 requires a newer Isaac Lab. Use this workspace's rsl_rl submodule.")
    # These Isaac Lab versions already use the native RSL-RL 3.x policy layout.
    return agent_cfg
