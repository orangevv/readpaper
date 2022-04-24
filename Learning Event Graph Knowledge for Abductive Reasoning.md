@[TOC](Learning Event Graph Knowledge for Abductive Reasoning)
[原文链接](https://aclanthology.org/2021.acl-long.403.pdf).

# Abstract
溯因推理目的是对观察到的事件推断出最合理的解释。为此，本文提出了一种基于叙事文本的溯因推理任务αNLI。已经有许多预训练模型用于这项任务推理，均取得了不错的效果。然而，丰富的事件常识并没有很好地用于这项任务。为了填补这一空白，我们提出了一种基于变分自编码器的模型ege-RoBERTa，该模型利用潜在变量从事件图中获取必要的常识知识，以指导溯因推理任务。

#  1.Introduction
Bhagavatula等人(2019)提出了一种基于自然语言的溯因推理任务αNLI。如图1 (a)所示，给定两个观测事件O1和O2， αNLI任务要求预测模型从两个候选假设事件H1和H2中选择一个更合理的解释。更直白点说从H1和H2中选择一个假设解释从O1到O2的变化。
<img src="https://img-blog.csdnimg.cn/2c4871095c15477db137e8ba2b505f3a.png?x-oss-process=image#pic_center =400x200" alt="图1（a）" style="zoom:50%;" />
然而，尽管预先训练的语言模型可以获得丰富的语言知识，以帮助理解事件的语义，但额外的常识知识仍然是溯及推理所必需的。例如，如图1(b)所示。可以发现通过引入中间变量I1和I2，使得事件链的发展更具有逻辑了。具体地，如图1 (c)所示，在RoBERTa框架的基础上，我们额外引入了一个潜在变量z来对关于中间事件的信息建模。
<img src="https://img-blog.csdnimg.cn/2c82307273ac43dd8ec4c603d32362e0.png?x-oss-process=image#pic_center =500x230" alt="在这里插入图片描述" style="zoom:50%;" />

# 2.Background
## Problem Formalization
我们将溯因推理任务形式化为条件分布p(Y|O1,Hi, O2)，其中Hi∈{H1,H2}，和Y∈[0,1]是衡量Hi合理性的相关性得分。O1、O2和Hi形成事件时间序列O1 Hi O2。我们将事件序列表示为X=(O1,Hi, O2)。因此，考虑到事件顺序，我们进一步将溯因推理任务刻画为p(Y|X)。
## Event Graph
形式上，事件图可以表示为G={V, R}，其中V为节点集，R为边集。每个节点Vi∈V对应一个事件，而Rij∈R是有向边Vi→Vj，有权值Wij，表示Vj是Vi的后续事件的概率。从事件图中我们可以获得额外的常识知识:(1)中间事件，(2)事件之间的关系。
为了清晰起见，我们将这种事件序列定义为后验事件序列X '，其中X ' = (O1, I1,Hi, I2, O2)。X '内事件之间的关系可以用邻接矩阵A∈R5×5描述，每个元素使用事件图的边权值进行初始化，关于这里的权值，会在实验的部分加以介绍:
<img src="https://img-blog.csdnimg.cn/6974f223b01f451aacd19e0fd95b03da.png?x-oss-process=image#pic_center =350x100" alt="在这里插入图片描述" style="zoom:50%;" />
这样邻接矩阵A可以用来描述在X‘中两个事件的关系。

# 3.Ege-RoBERTa as a Conditional Variational Autoencoder Based Reasoning Framework
为此，我们引入一个潜在变量z，通过两个阶段的训练过程从事件图中学习这些知识。我们将ege-RoBERTa模型构建为一个条件变分自动编码器(CVAE)。具体来说，对于潜在变量z, ege-RoBERTa使用三个神经网络来表征条件分布P(Y|X):先验网络P θ(z|X)，认知网络qφ(z|X '， a)和神经似然网络Pθ(Y|X, z)。
## Pre-training Stage: Learning Event Graph
在预训练阶段，在一个预构建的基于事件图的Pseudo实例集上预先训练ege-RoBERTa，该Pseudo实例集包含关于中间事件和事件关系的丰富信息。
如图2 (a)所示，潜在变量z直接以X '和a为条件，因此可以利用z学习事件图知识。
<img src="https://img-blog.csdnimg.cn/5b4fbf00adc8432ebab9f63768eeecd4.png?x-oss-process=image#pic_center =300x200" alt="在这里插入图片描述" style="zoom:50%;" />

## Finetuning Stage: Adapt Event Graph
如图2 (b)所示，在微调阶段，ege-RoBERTa在没有附加信息X '和a的αNLI数据集上接受训练。在这个阶段，模型学习将捕获的事件图知识适应于溯因推理任务。那么如图2 (c)所示，经过两阶段的训练过程，ege-RoBERTa可以基于潜在变量z预测事件相关分数Y。
<img src="https://img-blog.csdnimg.cn/bdbb8973e2d648f295247204fb4df4aa.png?x-oss-process=image#pic_center =500x240" alt="在这里插入图片描述" style="zoom:50%;" />

# 4.Architecture of ege-RoBERTa
我们介绍了ege-RoBERTa的具体实现。除了RoBERTa框架，它还引入了四个模块:(1)一个聚合器，为X和X '内的任何事件提供表示;(2) pθ(z|X)的注意先验网络模型;(3)基于图神经网络建模的qφ(z|X '， a)识别网络;(4)合并将潜在变量z合并到RoBERTa框架中进行下游溯因推理任务。

<img src="https://img-blog.csdnimg.cn/3db69b3c094d49b2a48b5b6be0135883.png?x-oss-process=image#pic_center =600x600" alt="在这里插入图片描述" style="zoom:50%;" />

## Event Representation Aggregator
<img src="https://img-blog.csdnimg.cn/b54344b005bb4067b232ffe7395e8001.png?x-oss-process=image#pic_center =400x400" alt="在这里插入图片描述" style="zoom:50%;" />
给定一个由令牌组成的事件序列X，入到嵌入层嵌入表示如下：
<img src="https://img-blog.csdnimg.cn/95ca1c3fe1f94a7bb4b01dfbe9fbd1aa.png#pic_center =300x40" alt="在这里插入图片描述" style="zoom:50%;" />
而xjk是第j个事件中的第k个令牌，RoBERTa的transformer第m层将这些令牌编码为上下文化的分布式表示.
<img src="https://img-blog.csdnimg.cn/5067f7d7ee1c459890aa6a0b33d13d42.png#pic_center =400x40" alt="在这里插入图片描述" style="zoom:50%;" />
其中，hjk∈R1×d是第j个事件中第k个token的分布表示。
对于每一个事件的更深层次的表示为
<img src="https://img-blog.csdnimg.cn/76e244de5be641039a435341267ff042.png#pic_center =160x40" alt="在这里插入图片描述" style="zoom:50%;" />
采用多头注意机制(MultiAttn)从H(M)中选择信息，得到每个事件的表示:
<img src="https://img-blog.csdnimg.cn/e6ac70f8666e4e82b3d6e424d7993a49.png#pic_center =360x50" alt="在这里插入图片描述" style="zoom:50%;" />
这里的X为3个事件构成的序列，因此将Ex = {e1，e2，e3}最为最终表示。
同理X‘的情况与上文的描述一致，只是多了两个中间事件，变成5个事件的序列表示为Ex’。
这里的X‘进入到Event Graph中在事件图中获取到邻接矩阵A的相关信息。

## recognition Network
<img src="https://img-blog.csdnimg.cn/d6190bf143544e39b5bdc5f34f3715af.png?x-oss-process=image#pic_center =300x200" alt="在这里插入图片描述" style="zoom:50%;" />
识别网络基于EX '和A对qφ(z|X '， A)进行建模，其中EX '是X '内事件的表示。遵循传统的VAE，假设qφ(z|X '， A)为多元高斯分布
，即为多元正态分布:
<img src="https://img-blog.csdnimg.cn/1fe707c388ba4873b11a22f44d70d0d2.png#pic_center =360x60" alt="在这里插入图片描述" style="zoom:50%;" />
为了获得µ' (X '， A)，我们首先使用GNN将EX '和邻接矩阵A结合起来。
<img src="https://img-blog.csdnimg.cn/1c132cf120424a07a24e1cf17517951e.png#pic_center =320x50" alt="在这里插入图片描述" style="zoom:50%;" />
式中，σ(·)为s型函数;W(u)∈Rd×d为权值矩阵，E(u) '为关系信息更新的事件表示形式。
<img src="https://img-blog.csdnimg.cn/1fa83800eb7d496dafadf1eb585af71e.png#pic_center =400x50" alt="在这里插入图片描述" style="zoom:50%;" />
最后，为了估计µ' (X '， A)，我们使用读出函数g(·)聚合E(U) '中的信息，我们将g(·)设为一个均值池化操作:
<img src="https://img-blog.csdnimg.cn/cff9f627487c47c6a2ebc428787484c2.png#pic_center =360x50" alt="在这里插入图片描述" style="zoom:50%;" />

## Prior Network
<img src="https://img-blog.csdnimg.cn/c1c3926046e447d1b30080864c0e17bd.png?x-oss-process=image#pic_center =300x200" alt="在这里插入图片描述" style="zoom:50%;" />

与识别网络一样，pθ(z|X)也服从多元正态分布，但参数不同:
<img src="https://img-blog.csdnimg.cn/b619199c6da142f0a2d27177df2d2a1a.png#pic_center =360x50" alt="在这里插入图片描述" style="zoom:50%;" />
为了获得µ(X)，与识别网络不同，先验网络从使用多头自我注意更新EX开始:
<img src="https://img-blog.csdnimg.cn/d37e6c52b9824ca8b18ab97bb727cbac.png#pic_center =360x50" alt="在这里插入图片描述" style="zoom:50%;" />
然后执行一个额外的多头自我注意操作来获得更深层次的表示:
<img src="https://img-blog.csdnimg.cn/ebf3124dcb764f50a585adfb1cf842ea.png#pic_center =390x35" alt="在这里插入图片描述" style="zoom:50%;" />
最后，通过聚合E(U)的信息来估计µ(X):
<img src="https://img-blog.csdnimg.cn/77d16f3603264574b0c5c22fff656f19.png#pic_center =360x50" alt="在这里插入图片描述" style="zoom:50%;" />
其中g(·)是一个平均池化操作。

## Merger
合并模块将潜在变量z和事件的更新(深度)表示合并到RoBERTa帧的第n个转换器层，用于预测相关性评分。我们采用多头注意机制从z和E(U)中选择相关信息。
<img src="https://img-blog.csdnimg.cn/20291706401a413896071ecae859507e.png?x-oss-process=image#pic_center =360x300" alt="在这里插入图片描述" style="zoom: 50%;" />
具体来说，在训练前阶段:
<img src="https://img-blog.csdnimg.cn/ea52bf734e174039962f0ef3e4faa5d3.png#pic_center =360x50" alt="在这里插入图片描述" style="zoom:50%;" />
在预训练阶段和预测阶段的时候：

<img src="https://img-blog.csdnimg.cn/86d8a0d72f81491e88f84b0309462619.png#pic_center =360x50" alt="在这里插入图片描述" style="zoom:50%;" />
以* H(N)作为RoBERTa后续的第(N + 1)transformer层的输入，根据附加的事件图知识进行外推推理，输出相关分数Y。

# 5.Experiments
## αNLI Dataset

αNLI数据集在训练集、开发集和测试集上分别包含169,654、1,532和4,056个<o1、o2、h1、h2>四元组。<o1、o2、h1、h2>观察事件是从一个短篇故事语料库中收集的，而所有的假设事件都是通过众包独立生成的。
## Construction of Event Graph
VIST和TimeTravel都是由五句话组成的短篇故事。辅助数据集中共有121,326个句子。为了得到两个节点Vi和Vj之间的边权Wij，我们通过下一个句子预测任务对RoBERTa-large模型进行微调。我们将故事文本中相邻的句子组(例如，故事的[1,2]句，[4,5]句)定义为正向实例，将不相邻的句子组或倒序的句子组(例如故事的[1,3]句，[5,4]句)定义为负向实例。
## Event Graph Based Pseudo Instance Set for Pretraining ege-RoBERTa 
为了有效地利用事件图知识，我们引入了一组伪实例对ege-RoBERTa模型进行预训练。具体来说，在辅助数据集中有一个五句话的故事，如表1所示，我们将故事的第1句和第5句分别定义为两个观察事件，第3句定义为假设事件，第2句和第4句定义为中间事件。这样就可以得到伪实例的后验事件序列X '和事件序列X。另外，给定X '，我们利用事件图的边权初始化邻接矩阵A的元素，并对A进行缩放，使其行和等于1。
<img src="https://img-blog.csdnimg.cn/76870fe12d7d4354bdc3b0d25e106941.png?x-oss-process=image#pic_center =500x220" alt="在这里插入图片描述" style="zoom:50%;" />

## Result
<img src="https://img-blog.csdnimg.cn/ad38e777aaaa46a8be3e7792bc7c10e5.png?x-oss-process=image#pic_center =600x400" alt="在这里插入图片描述" style="zoom: 50%;" />
从图中我们可以看的出来，与SVM和Infersent相比较，其他几个预训练的语言模型效果要明显更好一些。其中Baseline是RoBERTa-large。本文提出的模型ege-RoBERTa-large效果比目前最新的模型RoBERTa-GPT-MHKA的效果还要好一些，由于已经很接近人类的效果，此时的极小提升都是很困难的。
<img src="https://img-blog.csdnimg.cn/4492162485fb44a399a39ed0541d0829.png?x-oss-process=image#pic_center =360x140" alt="在这里插入图片描述" style="zoom:50%;" />
消融实验中，去除中间事件以及矩阵关系效果都是有所下降的。足见常识带来的效果提升。
<img src="https://img-blog.csdnimg.cn/3276912239744821b7e92c0e1495ec85.png?x-oss-process=image#pic_center =450x180" alt="讨论" style="zoom:50%;" />
对于X’的形式，在前文中提到理想的五元组状态是符合人类的思维逻辑的，在实验中也表明这样的假设效果是最好的，此外，相较于没有使用常识，使用常识的部分均是有一定的效果提升的。

# Conclusion

在本文中，我们提出了一个基于变分自编码器的框架ege-RoBERTa，该框架具有两个阶段的溯因推理任务训练过程。在预训练阶段，ege-RoBERTa能够通过潜变量从事件图中学习常识知识，然后在接下来的阶段，学习到的事件图知识可以适应溯因推理任务。实验结果表明，αNLI任务的性能较基线有所提高。