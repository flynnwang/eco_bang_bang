from ebb.env.luxenv import NebulaEnergyReduction


def test_nebula_energy_redunction():
  ner = NebulaEnergyReduction()

  assert ner.is_valid(25)
  assert not ner.is_valid(15)

  ner.add(25)
  ner.add(10)
  ner.add(10)

  # ignore invalid values
  ner.add(8)
  ner.add(8)
  ner.add(8)
  assert ner.best_guess() == 10
