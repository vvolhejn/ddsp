from typing import Optional

import numpy as np
import torch
import tensorflow as tf

import jukebox.hparams
import jukebox.make_models
from jukebox.vqvae.vqvae import VQVAE

vqvae: Optional[VQVAE] = None
levels = 3
strides = (8, 32, 128)
embeddings_params = None

def setup():
  global vqvae, embeddings_params

  sr = 44100
  secs = 4

  hps = jukebox.hparams.setup_hparams(
    "vqvae",
    dict(sample_length=sr * secs, sample_length_in_seconds=secs),
  )
  vqvae = jukebox.make_models.make_vqvae(hps, device='cpu')

  embeddings_params = [
    tf.convert_to_tensor(vqvae.bottleneck.level_blocks[i].k)
    for i in range(3)
  ]


setup()


def encode(audio):
  audio_torch = torch.as_tensor(audio).reshape(1, -1, 1)
  encodings = vqvae.encode(audio_torch)
  return encodings


def embedding_lookup(indices):
  assert isinstance(indices, list)
  assert len(indices) == levels

  embeddings = []

  for i in range(levels):
    embeddings.append(tf.nn.embedding_lookup(
      embeddings_params[i], indices[i].astype(tf.int32),
    ))

  # print(embeddings[2].shape)

  return embeddings


  #
  # is_tf = isinstance(indices[0], tf.Tensor)
  #
  # def preprocess(e):
  #   if is_tf:
  #     e = torch.as_tensor(np.array(e.astype(tf.int32)))
  #
  #   return e
  #
  # indices = [preprocess(e) for e in indices]
  #
  # def postprocess(e):
  #   if is_tf:
  #     e = tf.convert_to_tensor(e.permute(0, 2, 1))
  #
  #   return e
  #
  # embeddings = vqvae.bottleneck.decode(indices)
  # embeddings = [postprocess(e) for e in embeddings]
  #
  # return embeddings
