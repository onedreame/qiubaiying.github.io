---
layout:     post
title:      自然语言处理中的embeddings
subtitle:   Sense Embeddings
date:       2020-08-29
author:     OD
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - Pytorch
    - finetune
---

# Sense Embeddings

## 1. Introduction

​	word Embedding方式虽然在很多NLP任务中都已经成为标配，但是这种方法不能很好的区分一词多义(polysems)，因此近年来有很多关于Sense Embedding的研究。sense embedding技术要解决的一个突出问题就是meaning conflation deficiency，它希望通过直接建模单词的不同含义来缓解这种缺陷。目前，关于这个技术，有两种方向：unsupervised&knowledge-based。

> meaning conflation deficiency：将每个词映射到语义空间中的一个点会产生一个比较严重的问题：这种做法忽略了词可以有多种含义的事实，它把单词所有的含义都合并进一个表示向量中去，将不同的（甚至可能是不相关的）含义混淆到一个单一的表示中，会妨碍一个以这些表示为核心的nlp系统的语义理解。这个语义合并操作也会对准确的语义建模产生额外的负面影响，比如，一些语义不相关而含义相似的词在语义空间中被拉近，比如两个语义不相关的单词，"老鼠"和"屏幕"，因为他们与"mouse"这个单词的两种含义的相似性（啮齿动物和计算机输入设备），会在语义空间被拉近。

![](https://github.com/onedreame/onedreame.github.io/blob/master/img/embeddings/meaning_conflation_deficiency.png)	

<center>meaning conflation deficiency例子</center>

### 2. 技术路线

#### 2.1 UNSUPERVISED SENSE EMBEDDINGS

​	无监督的词义表征仅基于从文本语料库中提取的信息来构建。词义诱导(word sense induction)，即自动识别词的可能含义，是这些技术的核心。无监督模型通过分析文本语料库中的上下文语义，推导出一个词的不同意义，并根据从语料库中获得的统计知识来表示每个意义。根据模型所使用的文本语料类型，我们可以将无监督意义表示分为两大类：

- 只利用单语语料库的技术 

  这种技术同样可以分为两个流派：

  （1）clustering-based (也称为two-stage) models

  这类模型首先推导含义，然后为这些含义学习表示。

  这方面的开创性工作是语境组辨析(context-group discrimination, CGD),该方法是为了解决语义标注数据的知识获取瓶颈和对外部资源的依赖，尝试自动进行词义辨析。CGD方法的基本思想是通过对出现歧义词的语境进行聚类计算，从语境相似性中自动推导含义。更具体的说，一个歧义词$w$的每个上下文$C$都可以表示为一个上下文向量$\vec v_{C}$,计算方式为其内容词向量的中心点$\vec v_{c}(c \in C)$.对给定语料中的每个词计算出上下文向量，然后用EM算法将其聚成预定数量的群组(context group)。尽管基于聚类的方法很简单，但它构成了许多后续技术的基础，这些技术主要是在上下文的表示或下卧聚类算法上有所不同。

  ![](https://github.com/onedreame/onedreame.github.io/blob/master/img/embeddings/two_stages.png)

  <center>CGD算法例子</center>

  由于针对每个词都要计算它所有的上下文的向量表示，这显然是比较耗费资源的，因而在数据量大了以后就不再适用了，后续有研究针对这一问题进行了改进，即直接聚类上下文，上下文向量通过unigrams的特征向量来表示。2012的一个工作使用了三种技术来进一步提高了性能：通过对词向量进行idf加权平均来计算上下文向量，使用球形k-means进行聚类，一个词的出现会被标记上其聚类并采用二遍学习来学习意义表示。

  （2）joint training

  同时做含义推导以及表示学习。基于聚类的词义表示方法存在局限性，聚类和词义表示是相互独立完成的，因此，这两个阶段并不能从利用语言内在的相似性，而且，其计算开销也比较大。

  嵌入模型的引入是词义向量空间模型最具革命性的变化之一，sense embedding技术显然也从中受益良多。许多研究者提出了对Skip-gram模型的各种扩展，这将使捕获特定意义的区分成为可能。

  这类技术的一个例子就是Multiple-Sense Skip-Gram (MSSG)，与早期的工作类似，它将一个词的上下文表示为它词向量的中心点，并将其聚类形成目标词的意义表示，不过，根本的区别在于聚类和意义嵌入学习是联合进行的。在训练过程中，每个词的目的语义被动态选择为最接近上下文的语义，并且只对该语义进行权重更新。不过有研究认为，上述技术的局限性在于，它们只考虑到一个词的局部语境来推导其意义表征。为了解决这一局限性，出现了一种称为Topical Word Embeddings(TWE)的新技术，即允许每个词在不同的主题下有不同的嵌入，其中主题是利用 latent topic modelling进行全局计算的。该模型共有三种变体。(1)TWE-1，将每个话题视为一个伪词（pseudo word），分别学习topic embeddings和word embeddings；(2)TWE-2，将每个word-topic视为一个伪词（pseudo word），直接学习话题词嵌入；(3)TWE-3，为每个词和每个话题分配不同的嵌入，并通过concat相应的word-eembeddings和topic-embeddings建立每个词-话题对的嵌入。TWE模型的各种扩展已经被提出。神经张量跳格(Neural Tensor Skip-gram，NTSG)模型将话题建模的思想同样应用于意义表示，但引入了一个张量，以更好地学习单词和话题之间的相互作用。另一个扩展是MSWE，它认为在给定的上下文中，一个词可能会触发多个意义，并将TWE中最合适的意义的选择替换为反映该词与上下文中多个意义不同关联度的混合权重。

  joint-training也有自己的问题，第一：那就是为了易于实现，他们会假定每个词拥有固定数目的意义，而这显然是不合理的，因为不同的词的意义数目差别是非常大的。第二：大多数无监督模型的一个共同点是，它们扩展了Skip-gram模型，将一个词对其上下文的建模（就像在原始模型中那样）替换为对预期意义的附加建模。然而，这些模型中的语境词并没有进行消歧。因此，一个意义嵌入是以其上下文的词嵌入为条件的。

- 利用多语语料库的技术。

