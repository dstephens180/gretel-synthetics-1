import numpy as np
from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig

attributes = np.random.rand(10000, 3)
features = np.random.rand(10000, 20, 2)

config = DGANConfig(
    max_sequence_len=20,
    sample_len=5,
    batch_size=1000,
    epochs=10
)



model = DGAN(config)

model.train_dataframe(attributes, features)

synthetic_attributes, synthetic_features = model.generate(100)
