from .replay_buffer import ReplayBuffer
import pytest


@pytest.mark.skip("")
def test_replay_buffer_append():
  buf = ReplayBuffer(1)

  assert len(buf) == 0

  buf.append(1)
  assert len(buf) == 1
  assert buf.sample() == 1

  buf.append(2)
  assert len(buf) == 1
  assert buf.sample() == 2
