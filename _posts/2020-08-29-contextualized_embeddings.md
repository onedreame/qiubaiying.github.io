---
layout:     post
title:      自然语言处理中的embeddings
subtitle:   Contextualized Embeddings
date:       2020-08-29
author:     OD
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - nlp
    - Contextualized Embeddings
---

# 自然语言处理中的embeddings

## 1. Introduction

​	语境词嵌入（contextualized word (CW) ）是新一代词嵌入技术，它的突出特点就是词的表示对其所处的上下文具有敏感性，一个目标词的embedding可以根据他出现的上下文而变化，这些动态embeddings减轻了许多与静态词embeddings相关的问题，并为捕捉自然语言在上下文中的语义和句法属性提供了可靠的手段。语境化词嵌入尽管历史不长，但在几乎所有被应用的下游NLP任务中，都提供了显著的收益。

​	自被引入以来，预训练的word embeddings便在语义表示领域获得了支配性的地位。通常，一个nlp系统会被提供一个包含目标语言词汇中所有单词的大型预训练word embeddings，对于输入的句子，会通过对embeddings进行lookup操作获得对应word的embedding向量，这个流程从最初的one-hot编码到连续的词嵌入空间，这些进化提高了系统的泛化能力，进而提高了性能。

​	不过，预训练的word embeddings也有自己的局限性，那就是他们为每个词提供的是一个单一的静态的表示，在任何上下文中均是采用这个固定的embedding，而这很可能是不够的，比如，"the cells of a honeycomb","mobile cell","prison cell",这三种语境下虽然都有"cell"，不过其含义显然是不同的，而静态的embedding则无法表示这种不同。

​	静态的word embeddings困扰于两方面的限制：(1)忽略掉有特殊含义的单词所处的上下文很显然是对问题的一个简化，而这个简化并不是人类解释文本中单词的方式。(2)由于对单词施加了限制，所以模型很难捕捉到高层的语义特征，比如成分性和长期依赖性。因而，静态的word-based embeddings很显然会损害nlp系统理解输入文本的语义信息，因为在这种情况下，从输入序列中推断意义的重担就落在了nlp系统上面，它不得不处理诸如消歧，语义差别等等问题。

### 2.

​	

