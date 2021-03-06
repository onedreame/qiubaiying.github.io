---
layout:     post
title:      RNN中的Teacher Forcing
subtitle:   Teacher Forcing妙用
date:       2020-06-07
author:     OD
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - RNN
    - Teacher Forcing
---

# RNN中的Teacher Forcing

## 1. 什么是Teacher Forcing

​	Teacher Forcing（下面简称tf）是一种快速有效地训练递归神经网络模型的方法，这个方法名字听着很高端，其实应用非常简单，就是使用上一时间步的输入的groud truth作为输入，而取代了真实的目标输入。该技术广泛使用在了机器翻译，文本摘要，图像描述（ image captioning）等，在训练递归网络时，tf可以加快模型的收敛速度以及提升模型的不稳定性。

## 2.序列预测问题

​	我们实际生活中的很多问题都是序列问题，就拿我们熟悉的机器翻译来说，它对应的就是一种源语言的序列到目标语言的序列转换问题，在这类型的任务中，由于序列是依次产生的，所以天然适合[seq2seq](https://onedreame.github.io/2020/06/02/%E8%81%8A%E8%81%8Achatbot%E9%82%A3%E4%BA%9B%E4%BA%8B/)结构，而这类型的任务，如果直接使用目标序列作为输入指导训练，则有着收敛慢，模型稳定性差的问题。

​	以实际例子演示一下该方法的使用情况：

​	假如我们有一个句子“ Your plan sounds good.”，我们希望输入当前的单词，可以得到下一个单词，比如当输入“plan”的时候，我们期待能够输出“good", 让我们演示一遍：

1. 针对句子添加开始符号"sos"与终止符号"eos"，变成 “ sos Your plan sounds good eos”

2. 将“sos"送入模型，来产生第一个单词

3. 假如模型模型产生的预测结果是“hello”，则很显然，模型输入错了，因为我们期待的输出是“Your”，

   | 输入单词 | 预测单词 |
   | -------- | -------- |
   | sos      | hello    |

4. 在普通的序列预测中，显然是直接把预测的“hello”作为下一步的输入，这样做的话很显然模型已经偏离了轨道，并且每生成一个后续的单词都会受到惩罚。这使得学习速度变慢，模型不稳定。

   | 输入单词（普通的序列输入） | 预测单词 |
   | -------------------------- | -------- |
   | sos，hello                 | ？       |

5. 而采用tf的时候，我们不在使用预测的“hello”来作为输入，而是直接使用真实的groud truth “Your”作为输入，后续的步骤都可以采用这种方式。这样下来，该模型将可以快速学习正确的序列，或序列的正确统计属性。

| 输入单词        | 预测单词 |
| --------------- | -------- |
| sos，Your       | ？       |
| sos,  Your plan | ? ?      |

## 3.注意点

​	tf虽然可以加快收敛，提升稳定性，但是这种方法也会导致模型在实际使用时，当生成的序列与模型在训练过程中看到的情况不同时，模型可能会很脆弱或受到限制，这是因为RNN的调理上下文（之前生成的样本序列）与训练期间看到的序列存在分歧。

​	关于这个问题，有几种拓展方式来解决：

1. 搜索候选输出序列

   这种方式是对每个单词的预测概率进行搜索，以生成一些可能的候选输出序列。其缺点是只适用于具有离散输出值的预测问题，不能用于实值输出。

   这个方法的典型例子就是机器翻译中的beam search。

2. Curriculum Learning

   这种方法在训练过程中引入由先前时间步骤产生的输出，以鼓励模型学习如何纠正自己的错误。它可以通过随机选择使用groud truth或前一个时间步的生成输出作为当前时间步的输入来实现。在实际应用的时候，可以通过开始使用较高的tf率，后续逐渐降低tf率来改善效果。

   ```python
   # curriculum learning代码片段
   use_teacher_forcing = random.random() < teacher_forcing_ratio
   for t in range(max_target_length):
     decoder_output, decoder_hidden, decoder_attn = self.decoder(
         decoder_input, decoder_hidden, encoder_outputs
     )
   
     decoder_outputs.append(decoder_output)
     decoder_input = tgt[t] if use_teacher_forcing else decoder_output.argmax(-1)
   ```

   