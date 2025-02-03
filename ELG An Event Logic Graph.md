# ELG: An Event Logic Graph

[原文链接](https://arxiv.org/pdf/1907.08015.pdf)

## Abstract
传统的知识图谱主要关注实体及其关系，忽略了现实世界中的事件。ELG可以揭示现实世界事件的进化模式和发展逻辑。

## 1. Introduction
本文的主要贡献有三个方面：
1. 我们是最早提出ELG定义的国家之一。
2. 我们提出了一个从大规模非结构化文本语料库中构建ELG的框架。
3. 实验结果表明，ELG能够提高脚本事件预测等下游应用的性能。

## 2. Event Logic Graph

### 2.1 Definition
ELG是一个有向无环图，它的节点是事件，边代表事件之间的顺序关系、因果关系、条件关系或上下关系。ELG本质上是一个事件逻辑知识库，揭示了现实世界事件的演化模式和发展逻辑。

### 2.2 Three structures of ELG
- (a) 树形
- (b) 链式
- (c) 图

![ELG结构](https://i-blog.csdnimg.cn/blog_migrate/7d93add9672044d330d0e6582a2a9763.png#pic_center =740x520)

### 2.3 Form of Expression
在ELG中，事件被表示为抽象的、广义的、语义完整的事件元组E = (S, P, O)，其中P是动作，S是行动者，O是执行动作的对象。在我们的定义中，每个事件必须包含一个触发词(即P)，例如“run”，它主要表示事件的类型。

### 2.4 Abstract and Generalized
事件抽象和泛化，我们不关心事件的确切参与者、地点和时间。“谁看电影”和“看哪部电影”在ELG中并不重要。语义完全性是指人们能够毫无歧义地理解事件的意义。

### 2.5 Relationship between Events
上下位关系

![上下位关系](https://i-blog.csdnimg.cn/blog_migrate/e878770eb6089dedfaec2c09b70455c6.png#pic_center =640x380)

## 3. Architecture
我们提出了一个从大规模非结构化文本中构建语篇的框架。在清理数据之后，进行一系列自然语言处理步骤，包括分割、词性标记和依赖关系解析，用于事件提取。

![架构](https://i-blog.csdnimg.cn/blog_migrate/f136d3f63adf25ebe099908813623d95.png#pic_center =480x380)

### 3.1 Sequential Relation and Direction Recognition
给定事件对候选对象(A, B)，序列关系识别就是判断其是否具有序列关系。如果有顺序关系，则应进行方向识别，以区分方向。

基于属性的PMI值的计算：
\[ PMI(x, y) = \frac{p(x, y)}{p(x) \times p(y)} \]

![序列关系识别](https://i-blog.csdnimg.cn/blog_migrate/7aa7268330aa685b36de567ab3d9e7ac.png#pic_center =640x560)

### 3.2 Transition Probability Computation
\[ P(B|A) = \frac{f(A, B)}{f(A)} \]
其中f(A, B)为事件对(A, B)的共现频率，f(A)为事件A在整个语料库中的频率。

### 3.3 Unsupervised Causality Extraction
构建因果关系语义表的第一步是从非结构化的自然语言文本中识别因果关系对。我们构建了一套规则来提取因果事件的提及。每个规则遵循模板<Pattern, Constraint, Priority>，其中Pattern是一个包含选定连接器的正则表达式，Constraint是一个可以应用该模式的句子的语法约束，Priority是多个规则匹配时规则的优先级。

### 3.4 Supervised Causality Extraction
如图4所示，我们还使用Bert和BiLSTM+CRF模型提取因果关系。我们用以下标记来注释句子中的每个标记:B-cause, I-cause, beeffect, I-effect和O。我们为每个令牌i提供隐藏表示Ti作为BiLSTM的输入层。然后将BiLSTM的输出表示层送入分类层，预测因果标签。分类层中的预测以使用CRF方法进行的周围预测为条件。

![因果关系提取](https://i-blog.csdnimg.cn/blog_migrate/41d4cc38bd468adb7fcc18e75f80c467.png#pic_center =640x520)

### 3.5 Event Generalization
我们建议找到语义上相似的事件(A和A ')并将它们联系起来。为此，我们提出学习每个事件的分布式表示，并利用余弦相似度来衡量两个事件向量之间的语义相似度。

![事件泛化](https://i-blog.csdnimg.cn/blog_migrate/517545dbe74a9534fda5f680fbd867fe.png#pic_center =600x300)

## 4. Experiments
我们进行了三种实验：
1. 识别两个事件是否有顺序关系和方向。
2. 基于我们提出的无监督和监督方法提取事件之间的因果关系。
3. 使用下游的任务:脚本事件预测来证明ELG的有效性。

### 4.1 Dataset Description
我们从数据集中注释了2173个高共出现频率(≥5)的事件对。可以看到正向的实例和反向实例数量有所不同，需要抽取实例使得正负例平衡。

![数据集描述](https://i-blog.csdnimg.cn/blog_migrate/0fa7d4703f97620bc87f4e5126ea1299.png#pic_center =400x160)

因果关系实验，我们从腾讯、网易等在线网站抓取了1362345篇中文财经新闻文献。脚本事件预测是一项具有挑战性的基于事件的常识推理任务。我们在标准的多项选择叙述完形填空(MCNC)数据集上进行评估。

### 4.2 Result
![实验结果](https://i-blog.csdnimg.cn/blog_migrate/66229ac01270dc95ccf88e6b3e79b373.png#pic_center =640x580)

可以发现基于统计的方法效果比嵌入的方法要差。主要原因是学习事件的低维密集嵌入要比稀疏特征表示更有效，以进行脚本事件预测。点对和基于链的模型相比，基于图的模型具有更好的性能。这证实了我们的假设，即事件图结构比事件对和链更有效，并且可以为脚本事件的预测提供更多的事件交互信息。这主要是因为结对结构，链结构和图结构各有优势，并且它们可以互相补充。

![模型比较](https://i-blog.csdnimg.cn/blog_migrate/c5d6666422af1485f589f178d98c1ade.png#pic_center =640x580)

本文提出的模型是要更加稳定，更加优秀的。

## Conclusion
我们还提出了一个从大规模非结构化文本中构建ELG的框架，并利用该框架来提高脚本事件预测的性能。ELG中使用的所有技术都是独立于语言的。因此，我们可以很容易地构建其他语言版本的ELG。