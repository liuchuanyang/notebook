# 编写自己的GNN模块

有时，您的模型不仅仅是简单地堆叠现有的 GNN 模块。 例如，您想发明一种通过考虑节点重要性或边权重来聚合邻居信息的新方法。

在本教程结束时，您将能够

+ 了解 DGL 的消息传递 API。
+ 自己实现GraphSAGE卷积模块。

本教程假设您已经了解[训练用于节点分类的 GNN 的基础知识](https://docs.dgl.ai/tutorials/blitz/1_introduction.html)。

（时间估计：10分钟）

```python
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
```

## 消息传递和 GNNs

DGL 遵循由 [Gilmer 等人](https://arxiv.org/abs/1704.01212)提出的消息传递神经网络启发的消息传递范式。 从本质上讲，他们发现许多 GNN 模型可以适合以下框架：

![image-20211125151031126](/Users/huan/Library/Application Support/typora-user-images/image-20211125151031126.png)
$$
M^{(l)}表示消息传递函数，\sum 表示聚合函数，U^{(l)}表示更新函数；
提示\sum表示一个函数，并不一定是求和
$$
例如，[GraphSAGE 卷积（Hamilton 等人，2017 年）](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf)采用以下数学形式：

![image-20211125151637658](/Users/huan/Library/Application Support/typora-user-images/image-20211125151637658.png)

你可以看到消息传递是有方向的：从一个节点𝑢发送到另一个节点𝑣的消息不一定与从节点𝑣发送到节点𝑢的另一条消息以相反的方向相同。



尽管 DGL 通过 `dgl.nn.SAGEConv` 内置了对 `GraphSAGE `的支持，但您可以通过以下方式在 DGL 中自行实现 GraphSAGE 卷积。

```python
import dgl.function as fn

class SAGEConv(nn.Module):
    """Graph convolution module used by the GraphSAGE model.

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    """
    def __init__(self, in_feat, out_feat):
        super(SAGEConv, self).__init__()
        # A linear submodule for projecting the input and neighbor feature to the output.
        self.linear = nn.Linear(in_feat * 2, out_feat)

    def forward(self, g, h):
        """Forward computation

        Parameters
        ----------
        g : Graph
            The input graph.
        h : Tensor
            The input node feature.
        """
        with g.local_scope():
            g.ndata['h'] = h
            # update_all is a message passing API.
            g.update_all(message_func=fn.copy_u('h', 'm'), reduce_func=fn.mean('m', 'h_N'))
            h_N = g.ndata['h_N']
            h_total = torch.cat([h, h_N], dim=1)
            return self.linear(h_total)
```

这段代码的核心部分是` g.update_all `函数，它收集和平均相邻特征。 这里有三个概念：

+ 消息函数 **fn.copy_u('h', 'm') **将名称为 **'h'** 的节点特征复制为发送给邻居的消息。
+ Reduce（聚合） 函数 **fn.mean('m', 'h_N') **对名称为 **'m' **的所有接收消息进行平均，并将结果保存为新的节点特征 **'h_N'**。
+ **update_all **告诉 DGL 为所有节点和边触发***消息***和***聚合函数***。



之后，您可以堆叠自己的 GraphSAGE 卷积层以形成多层 GraphSAGE 网络。

```python
class Model(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(Model, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats)
        self.conv2 = SAGEConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
```

## 训练

以下数据加载和训练循环的代码直接复制自介绍教程。

```python
import dgl.data

dataset = dgl.data.CoraGraphDataset()
g = dataset[0]

def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    all_logits = []
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    for e in range(200):
        # Forward
        logits = model(g, features)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        # Note that we should only compute the losses of the nodes in the training set,
        # i.e. with train_mask 1.
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all_logits.append(logits.detach())

        if e % 5 == 0:
            print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                e, loss, val_acc, best_val_acc, test_acc, best_test_acc))

model = Model(g.ndata['feat'].shape[1], 16, dataset.num_classes)
train(g, model)
```



Out:

```
  NumNodes: 2708
  NumEdges: 10556
  NumFeats: 1433
  NumClasses: 7
  NumTrainingSamples: 140
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
In epoch 0, loss: 1.949, val acc: 0.114 (best 0.114), test acc: 0.103 (best 0.103)
In epoch 5, loss: 1.904, val acc: 0.160 (best 0.160), test acc: 0.154 (best 0.154)
In epoch 10, loss: 1.804, val acc: 0.432 (best 0.432), test acc: 0.454 (best 0.454)
In epoch 15, loss: 1.644, val acc: 0.462 (best 0.462), test acc: 0.469 (best 0.469)
In epoch 20, loss: 1.423, val acc: 0.488 (best 0.488), test acc: 0.499 (best 0.499)
In epoch 25, loss: 1.154, val acc: 0.578 (best 0.578), test acc: 0.584 (best 0.584)
In epoch 30, loss: 0.866, val acc: 0.680 (best 0.680), test acc: 0.676 (best 0.676)
In epoch 35, loss: 0.600, val acc: 0.726 (best 0.726), test acc: 0.740 (best 0.740)
In epoch 40, loss: 0.389, val acc: 0.738 (best 0.744), test acc: 0.754 (best 0.747)
In epoch 45, loss: 0.242, val acc: 0.742 (best 0.744), test acc: 0.759 (best 0.747)
In epoch 50, loss: 0.150, val acc: 0.742 (best 0.748), test acc: 0.759 (best 0.756)
In epoch 55, loss: 0.095, val acc: 0.752 (best 0.752), test acc: 0.760 (best 0.760)
In epoch 60, loss: 0.063, val acc: 0.746 (best 0.752), test acc: 0.764 (best 0.760)
In epoch 65, loss: 0.044, val acc: 0.750 (best 0.752), test acc: 0.763 (best 0.760)
In epoch 70, loss: 0.032, val acc: 0.750 (best 0.752), test acc: 0.766 (best 0.760)
In epoch 75, loss: 0.025, val acc: 0.750 (best 0.752), test acc: 0.766 (best 0.760)
In epoch 80, loss: 0.020, val acc: 0.750 (best 0.752), test acc: 0.764 (best 0.760)
In epoch 85, loss: 0.017, val acc: 0.750 (best 0.752), test acc: 0.766 (best 0.760)
In epoch 90, loss: 0.015, val acc: 0.750 (best 0.752), test acc: 0.766 (best 0.760)
In epoch 95, loss: 0.013, val acc: 0.756 (best 0.756), test acc: 0.767 (best 0.767)
In epoch 100, loss: 0.012, val acc: 0.756 (best 0.756), test acc: 0.765 (best 0.767)
In epoch 105, loss: 0.011, val acc: 0.756 (best 0.756), test acc: 0.764 (best 0.767)
In epoch 110, loss: 0.010, val acc: 0.756 (best 0.756), test acc: 0.764 (best 0.767)
In epoch 115, loss: 0.009, val acc: 0.758 (best 0.758), test acc: 0.764 (best 0.764)
In epoch 120, loss: 0.008, val acc: 0.758 (best 0.758), test acc: 0.764 (best 0.764)
In epoch 125, loss: 0.008, val acc: 0.758 (best 0.758), test acc: 0.765 (best 0.764)
In epoch 130, loss: 0.007, val acc: 0.758 (best 0.758), test acc: 0.764 (best 0.764)
In epoch 135, loss: 0.007, val acc: 0.758 (best 0.758), test acc: 0.764 (best 0.764)
In epoch 140, loss: 0.006, val acc: 0.758 (best 0.758), test acc: 0.764 (best 0.764)
In epoch 145, loss: 0.006, val acc: 0.758 (best 0.758), test acc: 0.763 (best 0.764)
In epoch 150, loss: 0.006, val acc: 0.760 (best 0.760), test acc: 0.763 (best 0.763)
In epoch 155, loss: 0.005, val acc: 0.758 (best 0.760), test acc: 0.765 (best 0.763)
In epoch 160, loss: 0.005, val acc: 0.756 (best 0.760), test acc: 0.764 (best 0.763)
In epoch 165, loss: 0.005, val acc: 0.756 (best 0.760), test acc: 0.766 (best 0.763)
In epoch 170, loss: 0.004, val acc: 0.754 (best 0.760), test acc: 0.766 (best 0.763)
In epoch 175, loss: 0.004, val acc: 0.756 (best 0.760), test acc: 0.765 (best 0.763)
In epoch 180, loss: 0.004, val acc: 0.756 (best 0.760), test acc: 0.765 (best 0.763)
In epoch 185, loss: 0.004, val acc: 0.756 (best 0.760), test acc: 0.765 (best 0.763)
In epoch 190, loss: 0.004, val acc: 0.756 (best 0.760), test acc: 0.765 (best 0.763)
In epoch 195, loss: 0.004, val acc: 0.756 (best 0.760), test acc: 0.765 (best 0.763)
```

## 更多自定义

在 DGL 中，我们在 `dgl.function `包下提供了许多内置的 message 和 reduce 函数。 您可以在[ API](https://docs.dgl.ai/api/python/dgl.function.html#apifunction) 文档中找到更多详细信息。

这些 API 允许人们快速实现新的图卷积模块。 例如，下面实现了一个新的 `SAGEConv`，它使用加权平均聚合邻居表示。 请注意， `edata `成员可以保存边缘特征，这些特征也可以参与消息传递。

```python
class WeightedSAGEConv(nn.Module):
    """Graph convolution module used by the GraphSAGE model with edge weights.

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    """
    def __init__(self, in_feat, out_feat):
        super(WeightedSAGEConv, self).__init__()
        # A linear submodule for projecting the input and neighbor feature to the output.
        self.linear = nn.Linear(in_feat * 2, out_feat)

    def forward(self, g, h, w):
        """Forward computation

        Parameters
        ----------
        g : Graph
            The input graph.
        h : Tensor
            The input node feature.
        w : Tensor
            The edge weight.
        """
        with g.local_scope():
            g.ndata['h'] = h
            g.edata['w'] = w
            g.update_all(message_func=fn.u_mul_e('h', 'w', 'm'), reduce_func=fn.mean('m', 'h_N'))
            h_N = g.ndata['h_N']
            h_total = torch.cat([h, h_N], dim=1)
            return self.linear(h_total)
```

因为这个数据集中的图没有边权重，我们在模型的 `forward() `函数中手动将所有边权重分配为1。 您可以将其替换为您自己的边权重。

```python
class Model(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(Model, self).__init__()
        self.conv1 = WeightedSAGEConv(in_feats, h_feats)
        self.conv2 = WeightedSAGEConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat, torch.ones(g.num_edges(), 1).to(g.device))
        h = F.relu(h)
        h = self.conv2(g, h, torch.ones(g.num_edges(), 1).to(g.device))
        return h

model = Model(g.ndata['feat'].shape[1], 16, dataset.num_classes)
train(g, model)
```

Out:

```
In epoch 0, loss: 1.949, val acc: 0.316 (best 0.316), test acc: 0.317 (best 0.317)
In epoch 5, loss: 1.886, val acc: 0.422 (best 0.422), test acc: 0.451 (best 0.451)
In epoch 10, loss: 1.756, val acc: 0.476 (best 0.532), test acc: 0.470 (best 0.524)
In epoch 15, loss: 1.556, val acc: 0.550 (best 0.550), test acc: 0.529 (best 0.529)
In epoch 20, loss: 1.293, val acc: 0.618 (best 0.618), test acc: 0.576 (best 0.576)
In epoch 25, loss: 0.996, val acc: 0.648 (best 0.648), test acc: 0.626 (best 0.626)
In epoch 30, loss: 0.707, val acc: 0.704 (best 0.704), test acc: 0.671 (best 0.671)
In epoch 35, loss: 0.465, val acc: 0.730 (best 0.730), test acc: 0.714 (best 0.714)
In epoch 40, loss: 0.288, val acc: 0.750 (best 0.750), test acc: 0.749 (best 0.749)
In epoch 45, loss: 0.175, val acc: 0.764 (best 0.764), test acc: 0.756 (best 0.756)
In epoch 50, loss: 0.108, val acc: 0.754 (best 0.764), test acc: 0.755 (best 0.756)
In epoch 55, loss: 0.069, val acc: 0.750 (best 0.764), test acc: 0.756 (best 0.756)
In epoch 60, loss: 0.046, val acc: 0.744 (best 0.764), test acc: 0.757 (best 0.756)
In epoch 65, loss: 0.033, val acc: 0.738 (best 0.764), test acc: 0.759 (best 0.756)
In epoch 70, loss: 0.025, val acc: 0.736 (best 0.764), test acc: 0.758 (best 0.756)
In epoch 75, loss: 0.020, val acc: 0.738 (best 0.764), test acc: 0.759 (best 0.756)
In epoch 80, loss: 0.017, val acc: 0.740 (best 0.764), test acc: 0.761 (best 0.756)
In epoch 85, loss: 0.014, val acc: 0.738 (best 0.764), test acc: 0.762 (best 0.756)
In epoch 90, loss: 0.012, val acc: 0.738 (best 0.764), test acc: 0.762 (best 0.756)
In epoch 95, loss: 0.011, val acc: 0.740 (best 0.764), test acc: 0.761 (best 0.756)
In epoch 100, loss: 0.010, val acc: 0.742 (best 0.764), test acc: 0.761 (best 0.756)
In epoch 105, loss: 0.009, val acc: 0.744 (best 0.764), test acc: 0.761 (best 0.756)
In epoch 110, loss: 0.008, val acc: 0.744 (best 0.764), test acc: 0.762 (best 0.756)
In epoch 115, loss: 0.008, val acc: 0.746 (best 0.764), test acc: 0.763 (best 0.756)
In epoch 120, loss: 0.007, val acc: 0.746 (best 0.764), test acc: 0.763 (best 0.756)
In epoch 125, loss: 0.007, val acc: 0.744 (best 0.764), test acc: 0.763 (best 0.756)
In epoch 130, loss: 0.006, val acc: 0.744 (best 0.764), test acc: 0.762 (best 0.756)
In epoch 135, loss: 0.006, val acc: 0.744 (best 0.764), test acc: 0.763 (best 0.756)
In epoch 140, loss: 0.005, val acc: 0.744 (best 0.764), test acc: 0.763 (best 0.756)
In epoch 145, loss: 0.005, val acc: 0.742 (best 0.764), test acc: 0.763 (best 0.756)
In epoch 150, loss: 0.005, val acc: 0.742 (best 0.764), test acc: 0.764 (best 0.756)
In epoch 155, loss: 0.005, val acc: 0.742 (best 0.764), test acc: 0.765 (best 0.756)
In epoch 160, loss: 0.004, val acc: 0.742 (best 0.764), test acc: 0.765 (best 0.756)
In epoch 165, loss: 0.004, val acc: 0.742 (best 0.764), test acc: 0.765 (best 0.756)
In epoch 170, loss: 0.004, val acc: 0.744 (best 0.764), test acc: 0.766 (best 0.756)
In epoch 175, loss: 0.004, val acc: 0.742 (best 0.764), test acc: 0.767 (best 0.756)
In epoch 180, loss: 0.003, val acc: 0.742 (best 0.764), test acc: 0.767 (best 0.756)
In epoch 185, loss: 0.003, val acc: 0.744 (best 0.764), test acc: 0.767 (best 0.756)
In epoch 190, loss: 0.003, val acc: 0.744 (best 0.764), test acc: 0.767 (best 0.756)
In epoch 195, loss: 0.003, val acc: 0.744 (best 0.764), test acc: 0.767 (best 0.756)
```

## 更多的用户定义函数

DGL 允许用户定义messge和reduce函数以获得最大的表现力。 这是一个用户定义的消息函数，相当于 **fn.u_mul_e('h', 'w', 'm')**。

```python
def u_mul_e_udf(edges):
    return {'m' : edges.src['h'] * edges.data['w']}
```

`edge` 有三个成员：`src`、`data `和`dst`，分别代表所有边的源节点特征、边特征和目的节点特征。

您也可以编写自己的reduce 函数。 例如，以下等效于对传入消息求平均值的内置 **fn.mean('m', 'h_N') 函数**：

```python
def mean_udf(nodes):
    return {'h_N': nodes.mailbox['m'].mean(1)}
```

简而言之，DGL 将根据节点的入度对节点进行分组，并且对于每个组，DGL 沿着第二个维度堆叠传入的消息。 然后，您可以沿第二个维度执行归约以聚合消息。

> 原文，这段翻译的不是很好，In short, DGL will group the nodes by their in-degrees, and for each group DGL stacks the incoming messages along the second dimension. You can then perform a reduction along the second dimension to aggregate messages.

使用自定义函数自定义消息和减少函数的更多详细信息，请参阅 [API 参考](https://docs.dgl.ai/api/python/udf.html#apiudf)。

## 编写自定义 GNN 模块的最佳实践

DGL 推荐以下优先级联系：

+ 使用 **dgl.nn** 模块。
+ 使用包含低级复杂操作的 `dgl.nn.functional `函数，例如计算传入边上每个节点的 softmax。
+ 将 **update_all** 与内置**message**和**reduce**函数一起使用。
+ 使用用户定义的**message**或**reduce**函数。

## What’s next?

- [Writing Efficient Message Passing Code](https://docs.dgl.ai/guide/message-efficient.html#guide-message-passing-efficient).

## 代码

https://docs.dgl.ai/_downloads/f56ecab48bbf4a2401a8e190833eaac7/3_message_passing.py

https://docs.dgl.ai/_downloads/b92a64b54af5c1bb43d3afb760105e4c/3_message_passing.ipynb