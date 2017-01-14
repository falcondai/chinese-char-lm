#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import kenlm

LM = os.path.join('char-kn-lm.arpa')
model = kenlm.Model(LM)
print('{0}-gram model'.format(model.order))

sentence = '捷 克 總 統 克 勞 斯 上 週 在 智 利 一 項 簽 約 儀 式'
print(sentence)
print('score %g' % model.score(sentence))
print('perplexity %g' % model.perplexity(sentence))

# Show scores and n-gram matches
words = ['<s>'] + sentence.split() + ['</s>']
for i, (prob, length, oov) in enumerate(model.full_scores(sentence)):
    print('{0} {1}: {2}'.format(prob, length, ' '.join(words[i+2-length:i+2])))
    if oov:
        print('\t"{0}" is an OOV'.format(words[i+1]))

# Find out-of-vocabulary words
for w in words:
    if not w in model:
        print('"{0}" is an OOV'.format(w))

#Stateful query
state = kenlm.State()
state2 = kenlm.State()
#Use <s> as context.  If you don't want <s>, use model.NullContextWrite(state).
model.BeginSentenceWrite(state)
accum = 0.0
accum += model.BaseScore(state, "a", state2)
accum += model.BaseScore(state2, "sentence", state)
#score defaults to bos = True and eos = True.  Here we'll check without the end
#of sentence marker.
assert (abs(accum - model.score("a sentence", eos = False)) < 1e-3)
accum += model.BaseScore(state, "</s>", state2)
assert (abs(accum - model.score("a sentence")) < 1e-3)
