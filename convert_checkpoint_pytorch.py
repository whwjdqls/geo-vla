from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi0_libero")

print(config)
# save the config as json
config_json_path = "pi0_libero_config.json"
with open(config_json_path, "w") as f:
    f.write(config)
print(f"Config saved to {config_json_path}")
# checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_base")
# checkpoint_dir = "/scratch2/whwjdqls99/pi/pi_zero"

# policy = policy_config.create_trained_policy(config, checkpoint_dir)

# # print information on policy
# print(policy)
# exit()
# action_chunks = policy.infer(example)["actions"]



