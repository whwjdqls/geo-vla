from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_libero")
# config = _config.get_config("pi05_base")

checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_base")
# checkpoint_dir = "/scratch2/whwjdqls99/pi/pi_zero"

policy = policy_config.create_trained_policy(config, checkpoint_dir)

# # print information on policy
print(policy)
# exit()
# action_chunks = policy.infer(example)["actions"]



