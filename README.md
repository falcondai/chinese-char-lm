# chinese-char-lm
This is the code associated with the publication _Glyph-aware Embedding of Chinese Characters_ by Dai and Cai. Please consider to cite the paper if you find the code useful in some way for your research.

```
Dai, Falcon Z., and Zheng Cai. "Glyph-aware Embedding of Chinese Characters." EMNLP 2017 (2017): 64.
```

```bibtex
@article{dai2017glyph,
  title={Glyph-aware Embedding of Chinese Characters},
  author={Dai, Falcon Z and Cai, Zheng},
  journal={EMNLP 2017},
  pages={64},
  year={2017}
}
```

# usage

- We used Google Noto font for all of our experiments. Download Google Noto Simplified Chinese fonts (https://www.google.com/get/noto/#sans-hans). Unzip it under the project directory. It is needed to render the glyphs.
- Requires Tensorflow v1.1 and Python 2.7.x
- Clone the repo and check out a particular branch or a specific commit with `$ git checkout <branch-name or git-tag>`

# replication

In favor of replicability, we git-tagged the original git commits we used to obtain the published figures. Please see the [release](https://github.com/falcondai/chinese-char-lm/releases) for a complete list of git tags (compare with the model names in the paper). Please use the issues page to contact us with code issues so more people can benefit from the conversations.

## summary of our implementation

Commit [msr-m1](https://github.com/falcondai/chinese-char-lm/tree/msr-m1) is a good place to start for language modeling. See https://github.com/falcondai/chinese-char-lm/blob/msr-m1/train_id_cnn_lm.py#L35 for a few related models (they differ by whether they use character-id embedding, glyph embedding, or both). For the Chinese segmentation task (tokenizing Chinese sentences which lack whitespaces by convention), you probably want to consult https://github.com/falcondai/chinese-char-lm/blob/segmentation/train_cnn_segmentation.py.

On a high level, our implementation uses no pre-trained embeddings and render the characters into glyphs on-the-fly. Glyph rendering calls are slow, so we cache the glyphs of seen characters which gives a dramatic speedup (see https://github.com/falcondai/chinese-char-lm/blob/msr-m1/train_cnn_lm.py#L18). We consider the input activation, - the combined output of a CNN over the glyph and a trained character-id embedding -, to the RNN as the _effective embedding_ for an input character. 

In terms of implementation: 
1. It takes in the path to a text file (utf-8 encoded) and the path to a vocabulary as input (see https://github.com/falcondai/chinese-char-lm/blob/msr-m1/train_id_cnn_lm.py#L61) to build a tensorflow input pipeline. In the case of segmentation, an additional path to the ground truth segmentation annotations. 
2. The characters are rendered into glyphs (see https://github.com/falcondai/chinese-char-lm/blob/msr-m1/train_id_cnn_lm.py#L13) and pass to the CNN. In parallel, we also look up the embedding using its vocabulary id. (We do both for all models, and then simply use a 0/1 multiplier to shutdown the path we don't need before outputting to the RNN in the specific model variant. See https://github.com/falcondai/chinese-char-lm/blob/msr-m1/train_id_cnn_lm.py#L40)
3. Lastly the output is fed into an standard RNN as common in other contemporary works.
4. Train end-to-end for the given task.

# authors
Falcon Dai (dai@ttic.edu)

Zheng Cai (jontsai@uchicago.edu)
