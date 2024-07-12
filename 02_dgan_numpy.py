import numpy as np
from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig, OutputType

attributes = np.random.randint(0, 3, size=(1000,3))
features = np.random.random(size=(1000,20,2))

model = DGAN(DGANConfig(
    max_sequence_len=20,
    sample_len=4,
    batch_size=1000,
    epochs=10,  # For real data sets, 100-1000 epochs is typical
))




model.train_numpy(
    attributes, features,
    attribute_types = [OutputType.DISCRETE] * 3,
    feature_types = [OutputType.CONTINUOUS] * 2,
)

synthetic_attributes, synthetic_features = model.generate(100)
