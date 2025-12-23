# CompHRDoc

Comp-HRDoc is the first comprehensive benchmark, specifically designed for hierarchical document structure analysis. It encompasses tasks such as page object detection, reading order prediction, table of contents extraction, and hierarchical structure reconstruction. Comp-HRDoc is built upon the [HRDoc-Hard dataset](https://github.com/jfma-USTC/HRDoc), which comprises 1,000 documents for training and 500 documents for testing. We retain all original images without modification and extend the original annotations to accommodate the evaluation of these included tasks. The dataset is for model training and testing. Users can use this dataset to train a model or evaluate the performance for hierarchical document structure analysis.

Comp-HRDoc 是首个专门为层次化文档结构分析设计的综合基准测试。它涵盖了页面对象检测、阅读顺序预测、目录提取和层次结构重建等任务。Comp-HRDoc 基于 [HRDoc-Hard 数据集](https://github.com/jfma-USTC/HRDoc) 构建，
包含 1,000 份训练文档和 500 份测试文档。我们保留所有原始图像不做修改，并扩展原始标注以适应这些任务的评估。该数据集用于模型训练和测试。用户可以使用此数据集训练模型或评估层次化文档结构分析的性能。

## News / 新闻

- **We released the annotations of the Comp-HRDoc benchmark, please refer to [`CompHRDoc.zip`](./CompHRDoc.zip).**
- **我们发布了 Comp-HRDoc 基准测试的标注，请参阅 [`CompHRDoc.zip`](./CompHRDoc.zip)。**
- **We released the evaluation tool of the Comp-HRDoc benchmark, please refer to [`evaluation`](evaluation/) folder.**
- **我们发布了 Comp-HRDoc 基准测试的评估工具，请参阅 [`evaluation`](evaluation/) 文件夹。**
- **We released the original paper, [Detect-Order-Construct: A Tree Construction based Approach for Hierarchical Document Structure Analysis](https://arxiv.org/pdf/2401.11874.pdf), to Arxiv.**
- **我们在 Arxiv 上发布了原始论文：[Detect-Order-Construct: 一种基于树构建的层次化文档结构分析方法](https://arxiv.org/pdf/2401.11874.pdf)。**

## Introduction / 简介

Document Structure Analysis (DSA) is a comprehensive process that identifies the fundamental components within a document, encompassing headings, paragraphs, lists, tables, and figures, and subsequently establishes the logical relationships and structures of these components. This process results in a structured representation of the document's physical layout that accurately mirrors its logical structure, thereby enhancing the effectiveness and accessibility of information retrieval and processing. In a contemporary digital landscape, the majority of mainstream documents are structured creations, crafted using hierarchical-schema authoring software such as LaTeX, Microsoft Word, and HTML. Consequently, Hierarchical Document Structure Analysis (HDSA), which focuses on extracting and reconstructing the inherent hierarchical structures within these document layouts, has gained significant attention. Previous datasets primarily focus on specific sub-tasks of DSA, such as Page Object Detection, Reading Order Prediction, and Table of Contents (TOC) Extraction, among others. Despite the substantial progress achieved in these individual sub-tasks, there remains a gap in the research community for a comprehensive end-to-end system or benchmark that addresses all aspects of document structure analysis concurrently. Leveraging the HRDoc dataset, we establish a comprehensive benchmark, Comp-HRDoc, aimed at evaluating page object detection, reading order prediction, table of contents extraction, and hierarchical structure reconstruction concurrently.

文档结构分析（DSA）是一个综合性过程，用于识别文档中的基本组成部分，包括标题、段落、列表、表格和图片，并随后建立这些组成部分之间的逻辑关系和结构。
此过程生成文档物理布局的结构化表示，准确反映其逻辑结构，从而提高信息检索和处理的效率和可访问性。
在当代数字化环境中，大多数主流文档都是结构化创作，使用 LaTeX、Microsoft Word 和 HTML 等层次化模式编写软件制作。
因此，层次化文档结构分析（HDSA）专注于提取和重建这些文档布局中固有的层次结构，已获得广泛关注。
以往的数据集主要关注 DSA 的特定子任务，如页面对象检测、阅读顺序预测和目录（TOC）提取等。
尽管在这些单独的子任务上取得了实质性进展，但研究界仍缺乏一个能够同时处理文档结构分析所有方面的综合端到端系统或基准测试。
利用 HRDoc 数据集，我们建立了一个综合基准测试 Comp-HRDoc，旨在同时评估页面对象检测、阅读顺序预测、目录提取和层次结构重建。

<!-- ![](assets/example.png) -->
<img src="assets/example.png" height="500" alt="">

### Data Directory Structure / 数据目录结构

```plaintext
Comp-HRDoc/
├── HRDH_MSRA_POD_TRAIN/
│   ├── Images/ # put the document images of HRDoc-Hard training set into this folder / 将 HRDoc-Hard 训练集的文档图像放入此文件夹
│   │   ├── 1401.6399_0.png
│   │   ├── 1401.6399_1.png
│   │   └── ...
│   ├── hdsa_train.json
│   ├── coco_train.json
│   ├── README.md # a detailed explanation of each file and folder / 每个文件和文件夹的详细说明
│   └── ...
└──HRDH_MSRA_POD_TEST/
    ├── Images/ # put the document images of HRDoc-Hard test set into this folder / 将 HRDoc-Hard 测试集的文档图像放入此文件夹
    │   ├── 1401.3699_0.png
    │   ├── 1401.3699_1.png
    │   └── ...
    ├── test_eval/ # hierarchical document structure for evaluation / 用于评估的层次化文档结构
    │   ├── 1401.3699.json
    │   ├── 1402.2741.json
    │   └── ...
    ├── test_eval_toc/ # table of contents structure for evaluation / 用于评估的目录结构
    │   ├── 1401.3699.json
    │   ├── 1402.2741.json
    │   └── ...
    ├── hdsa_test.json
    ├── coco_test.json
    ├── README.md # a detailed explanation of each file and folder / 每个文件和文件夹的详细说明
    └── ...
```

**For a detailed explanation of each file and folder, please refer to `datasets/Comp-HRDoc/HRDH_MSRA_POD_TRAIN/README.md` and `datasets/Comp-HRDoc/HRDH_MSRA_POD_TEST/README.md`.**

**有关每个文件和文件夹的详细说明，请参阅 `datasets/Comp-HRDoc/HRDH_MSRA_POD_TRAIN/README.md` 和 `datasets/Comp-HRDoc/HRDH_MSRA_POD_TEST/README.md`。**

**Due to license restrictions, please go to [HRDoc-Hard dataset](https://github.com/jfma-USTC/HRDoc) to download the images of HRDoc-Hard and put them into the corresponding folders.**

**由于许可证限制，请前往 [HRDoc-Hard 数据集](https://github.com/jfma-USTC/HRDoc) 下载 HRDoc-Hard 的图像并将其放入相应的文件夹中。**

### Evaluation Tool / 评估工具

To utilize the evaluation tool for assessing your model's performance on the Comp-HRDoc dataset, please consult the script located at [`evaluation/unified_layout_evaluation.py`](evaluation/unified_layout_evaluation.py).

要使用评估工具评估您的模型在 Comp-HRDoc 数据集上的性能，请参阅位于 [`evaluation/unified_layout_evaluation.py`](evaluation/unified_layout_evaluation.py) 的脚本。

Below is an example illustrating how to conduct an evaluation for the task of reconstructing the hierarchical document structure:

以下是一个示例，说明如何对层次化文档结构重建任务进行评估：

```python
hds_gt = "datasets/Comp-HRDoc/HRDH_MSRA_POD_TEST/test_eval/"
hds_pred = "path_to_your_predicted_hierarchical_structure/"
python evaluation/hrdoc_tool/teds_eval.py --gt_anno {hds_gt} --pred_folder {hds_pred}
```

We also provide some examples in [`evaluation/examples/`](evaluation/examples/) to demonstrate the format of predicted files required by the evaluation tool.

我们还在 [`evaluation/examples/`](evaluation/examples/) 中提供了一些示例，以演示评估工具所需的预测文件格式。

### Detect-Order-Construct / 检测-排序-构建

We proposed a comprehensive approach to thoroughly analyzing hierarchical document structures using a tree construction based method. This method decomposes tree construction into three distinct stages, namely Detect, Order, and Construct. Initially, given a set of document images, the Detect stage is dedicated to identifying all page objects and assigning a logical role to each object, thereby forming the nodes of the hierarchical document structure tree. Following this, the Order stage establishes the reading order relationships among these nodes, which corresponds to a pre-order traversal of the hierarchical document structure tree. Finally, the Construct stage identifies hierarchical relationships (e.g., Table of Contents) between semantic units to construct an abstract hierarchical document structure tree. By integrating the results of all three stages, we can effectively construct a complete hierarchical document structure tree, facilitating a more comprehensive understanding of complex documents.

我们提出了一种使用基于树构建方法来全面分析层次化文档结构的综合方法。
该方法将树构建分解为三个不同的阶段，即检测（Detect）、排序（Order）和构建（Construct）。首先，给定一组文档图像，检测阶段专门用于识别所有页面对象并为每个对象分配逻辑角色，从而形成层次化文档结构树的节点。随后，排序阶段在这些节点之间建立阅读顺序关系，这对应于层次化文档结构树的前序遍历。最后，构建阶段识别语义单元之间的层次关系（例如目录）以构建抽象的层次化文档结构树。通过整合这三个阶段的结果，我们可以有效地构建完整的层次化文档结构树，从而更全面地理解复杂文档。

<img src="assets/pipeline.png">

## Results / 结果

### Hierarchical Document Structure Reconstruction on HRDoc / HRDoc 上的层次化文档结构重建
<img src="assets/hrdoc_results.png">

### End-to-End Evaluation on Comp-HRDoc / Comp-HRDoc 端到端评估
<img src="assets/results.png">

