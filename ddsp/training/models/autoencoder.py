# Copyright 2022 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model that encodes audio features and decodes with a ddsp processor group."""

import ddsp
from ddsp.training.models.model import Model
from codetiming import Timer

class Autoencoder(Model):
  """Wrap the model function for dependency injection with gin."""

  def __init__(self,
               preprocessor=None,
               encoder=None,
               decoder=None,
               processor_group=None,
               losses=None,
               **kwargs):
    super().__init__(**kwargs)
    self.preprocessor = preprocessor
    self.encoder = encoder
    self.decoder = decoder
    self.processor_group = processor_group
    self.loss_objs = ddsp.core.make_iterable(losses)

  def encode(self, features, training=True):
    """Get conditioning by preprocessing then encoding."""
    with Timer("Autoencoder.preprocessor", logger=None):
      if self.preprocessor is not None:
        features.update(self.preprocessor(features, training=training))

    with Timer("Autoencoder.encoder", logger=None):
      if self.encoder is not None:
        features.update(self.encoder(features))

    return features

  def decode(self, features, training=True):
    """Get generated audio by decoding than processing."""
    with Timer("Autoencoder.decoder", logger=None):
      features.update(self.decoder(features, training=training))

    # with Timer("Autoencoder.processor_group", logger=None):
    res = self.processor_group(features)

    return res

  def get_audio_from_outputs(self, outputs):
    """Extract audio output tensor from outputs dict of call()."""
    return outputs['audio_synth']

  def call(self, features, training=True):
    """Run the core of the network, get predictions and loss."""
    features = self.encode(features, training=training)

    with Timer("Autoencoder.decoder", logger=None):
      features.update(self.decoder(features, training=training))

    # Run through processor group.
    with Timer("Autoencoder.processor_group", logger=None):
      pg_out = self.processor_group(features, return_outputs_dict=True)

    # Parse outputs
    outputs = pg_out['controls']
    outputs['audio_synth'] = pg_out['signal']

    if training or True:
      self._update_losses_dict(
        self.loss_objs, features['audio_16k'], outputs['audio_synth'])

    return outputs

