from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download
from openpi.policies.libero_policy import make_libero_example


config = _config.get_config("pi0_ours_low_mem_finetune")
# checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_fast_droid")
checkpoint_dir = "/scratch2/whwjdqls99/pi/pi0_ours_low_mem_finetune/debug/410"

policy = policy_config.create_trained_policy(config, checkpoint_dir)

example = make_libero_example()
result = policy.infer(example)
# print(result)
action_chunks = result["actions"]
print(action_chunks.shape)