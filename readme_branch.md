# Computing partition energy

Computed on the same EZ-SP features and only for the first level of the partition.

on S3DIS Fold 5
- EZ-SP: `python src/partition_energy.py experiment=partition_energy_EZSP logger=csv datamodule.mini=True`
- PCP (on EZ-SP features): `python src/partition_energy.py experiment=partition_energy_PCP logger=csv datamodule.mini=True`