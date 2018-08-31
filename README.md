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

In favor of replicability, we git-tagged the original git commits we used to obtain the published figures. Please see the [release](https://github.com/falcondai/chinese-char-lm/releases). Please use the issues page to contact us with code issues so more people can benefit from the conversations.

# authors
Falcon Dai (dai@ttic.edu)

Zheng Cai (jontsai@uchicago.edu)
