from .delivery import DeliveryEnv


def test_delivery_env_reset():
  env = DeliveryEnv()

  obs, score_reward, done, info = env.reset()

  # action = [
  # [0, 0, 0, 0, 1],
  # [0] * 20,
  # [0] * 5,
  # [1] * 10,
  # ]
  action = {'move_action': 4, 'grab_action': 0}

  joint_act = [action, action]

  for i in range(10):
    obs, score_reward, done, info = env.step(joint_act)
