from ebb.env.luxenv import seed_to_transpose


def test_seed_to_transpose():
  assert seed_to_transpose(100) == (False, False)
  assert seed_to_transpose(101) == (True, False)
  assert seed_to_transpose(102) == (False, True)
  assert seed_to_transpose(103) == (True, True)
