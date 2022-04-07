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

# Lint as: python3
"""Library of preprocess functions."""

import ddsp
from ddsp.training import nn
import gin
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config

F0_RANGE = ddsp.spectral_ops.F0_RANGE
DB_RANGE = ddsp.spectral_ops.DB_RANGE

tfkl = tf.keras.layers
np_config.enable_numpy_behavior()


# ---------------------- Preprocess Helpers ------------------------------------
def at_least_3d(x):
  """Optionally adds time, batch, then channel dimension."""
  x = x[tf.newaxis] if not x.shape else x
  x = x[tf.newaxis, :] if len(x.shape) == 1 else x
  x = x[:, :, tf.newaxis] if len(x.shape) == 2 else x
  return x


def scale_db(db):
  """Scales [-DB_RANGE, 0] to [0, 1]."""
  return (db / DB_RANGE) + 1.0


def inv_scale_db(db_scaled):
  """Scales [0, 1] to [-DB_RANGE, 0]."""
  return (db_scaled - 1.0) * DB_RANGE


def scale_f0_hz(f0_hz):
  """Scales [0, Nyquist] Hz to [0, 1.0] MIDI-scaled."""
  return ddsp.core.hz_to_midi(f0_hz) / F0_RANGE


def inv_scale_f0_hz(f0_scaled):
  """Scales [0, 1.0] MIDI-scaled to [0, Nyquist] Hz."""
  return ddsp.core.midi_to_hz(f0_scaled * F0_RANGE)


# ---------------------- Preprocess objects ------------------------------------
@gin.register
class F0LoudnessPreprocessor(nn.DictLayer):
  """Resamples and scales 'f0_hz' and 'loudness_db' features."""

  def __init__(self,
               time_steps=1000,
               frame_rate=250,
               sample_rate=16000,
               compute_loudness=True,
               compute_f0=False,
               **kwargs):
    super().__init__(**kwargs)
    self.time_steps = time_steps
    self.frame_rate = frame_rate
    self.sample_rate = sample_rate
    self.compute_loudness = compute_loudness
    self.compute_f0 = compute_f0

  def call(self, loudness_db, f0_hz, f0_confidence, audio=None) -> [
      'f0_hz', 'loudness_db', 'f0_scaled', 'ld_scaled', 'f0_confidence']:
    # We have to pass f0_confidence for the case where `self.compute_f0` is off
    # and we need to return the original confidence unchanged.

    # Compute loudness fresh (it's fast).
    if self.compute_loudness:
      loudness_db = ddsp.spectral_ops.compute_loudness(
          audio,
          sample_rate=self.sample_rate,
          frame_rate=self.frame_rate)

    # Recomputing f0 might be desirable if we want to include it in the timing info
    if self.compute_f0:
      assert len(audio) == 1, "Batch size must be 1 when recomputing f0."

      f0_hz, f0_confidence = ddsp.spectral_ops.compute_f0(
          audio[0],
          frame_rate=self.frame_rate)

    # Resample features to the frame_rate.
    f0_hz = self.resample(f0_hz)
    loudness_db = self.resample(loudness_db)
    # For NN training, scale frequency and loudness to the range [0, 1].
    # Log-scale f0 features. Loudness from [-1, 0] to [1, 0].
    f0_scaled = scale_f0_hz(f0_hz)
    ld_scaled = scale_db(loudness_db)
    return f0_hz, loudness_db, f0_scaled, ld_scaled, f0_confidence

  @staticmethod
  def invert_scaling(f0_scaled, ld_scaled):
    """Takes in scaled f0 and loudness, and puts them back to hz & db scales."""
    f0_hz = inv_scale_f0_hz(f0_scaled)
    loudness_db = inv_scale_db(ld_scaled)
    return f0_hz, loudness_db

  def resample(self, x):
    x = at_least_3d(x)
    return ddsp.core.resample(x, self.time_steps)


@gin.register
class F0PowerPreprocessor(F0LoudnessPreprocessor):
  """Dynamically compute additional power_db feature."""

  def __init__(self,
               time_steps=1000,
               frame_rate=250,
               sample_rate=16000,
               frame_size=64,
               **kwargs):
    super().__init__(time_steps, **kwargs)
    self.frame_rate = frame_rate
    self.sample_rate = sample_rate
    self.frame_size = frame_size

  def call(self, f0_hz, power_db=None, audio=None) -> [
      'f0_hz', 'pw_db', 'f0_scaled', 'pw_scaled']:
    """Compute power on the fly if it's not in the inputs."""
    # For NN training, scale frequency and loudness to the range [0, 1].
    f0_hz = self.resample(f0_hz)
    f0_scaled = scale_f0_hz(f0_hz)

    if power_db is not None:
      # Use dataset values if present.
      pw_db = power_db
    elif audio is not None:
      # Otherwise, compute power on the fly.
      pw_db = ddsp.spectral_ops.compute_power(audio,
                                              sample_rate=self.sample_rate,
                                              frame_rate=self.frame_rate,
                                              frame_size=self.frame_size)
    else:
      raise ValueError('Power preprocessing requires either '
                       '"power_db" or "audio" keys to be provided '
                       'in the dataset.')
    # Resample.
    pw_db = self.resample(pw_db)
    # Scale power.
    pw_scaled = scale_db(pw_db)

    return f0_hz, pw_db, f0_scaled, pw_scaled

  @staticmethod
  def invert_scaling(f0_scaled, pw_scaled):
    """Puts scaled f0, loudness, and power back to hz & db scales."""
    f0_hz = inv_scale_f0_hz(f0_scaled)
    power_db = inv_scale_db(pw_scaled)
    return f0_hz, power_db


