import tensorflow_hub as hub
import tensorflow as tf

models = {
  "v1": hub.load("https://tfhub.dev/google/spice/1"),
  "v2": hub.load("https://tfhub.dev/google/spice/2"),
}

def output_to_hz(pitch_output):
  # Calibration constants
  PT_OFFSET = 25.58
  PT_SLOPE = 63.07
  FMIN = 10.0
  BINS_PER_OCTAVE = 12.0
  cqt_bin = pitch_output * PT_SLOPE + PT_OFFSET
  return FMIN * 2.0 ** (1.0 * cqt_bin / BINS_PER_OCTAVE)


def predict(audio, version):
  try:
    model = models[version]
  except KeyError:
    raise ValueError(f"Unrecognized SPICE version: {version}. "
                     f"Valid values are {models.keys()}.")

  input = tf.constant(audio)
  output = model.signatures["serving_default"](input)
  f0 = output["pitch"]
  confidence = 1 - output["uncertainty"]

  f0_hz = output_to_hz(f0)

  return f0_hz, confidence
