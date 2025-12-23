# UniHDSA: A Unified Relation Prediction Approach for Hierarchical Document Structure Analysis

# UniHDSA：一种统一的层次化文档结构分析关系预测方法

## Introduction / 简介

Document structure analysis is essential for understanding both the physical layout and logical structure of documents, aiding in tasks such as information retrieval, document summarization, and knowledge extraction. Hierarchical Document Structure Analysis (HDSA) aims to restore the hierarchical structure of documents created with hierarchical schemas. Traditional approaches either focus on specific subtasks in isolation or use multiple branches to address distinct tasks. In this work, we introduce UniHDSA, a unified relation prediction approach for HDSA that treats various subtasks as relation prediction problems within a consolidated label space. This allows a single module to handle multiple tasks simultaneously, improving efficiency, scalability, and adaptability. Our multimodal Transformer-based system demonstrates state-of-the-art performance on the Comp-HRDoc benchmark and competitive results on the DocLayNet dataset, showcasing the effectiveness of our method across all subtasks.

文档结构分析对于理解文档的物理布局和逻辑结构至关重要，有助于信息检索、文档摘要和知识提取等任务。
层次化文档结构分析（HDSA）旨在恢复使用层次化模式创建的文档的层次结构。传统方法要么孤立地关注特定子任务，要么使用多个分支来处理不同的任务。
在这项工作中，我们引入了 UniHDSA，这是一种用于 HDSA 的统一关系预测方法，它将各种子任务视为统一标签空间内的关系预测问题。
这使得单个模块可以同时处理多个任务，从而提高效率、可扩展性和适应性。
我们基于多模态 Transformer 的系统在 Comp-HRDoc 基准测试上展示了最先进的性能，并在 DocLayNet 数据集上取得了有竞争力的结果，展示了我们的方法在所有子任务上的有效性。

## Reproduction / 复现

This project is built on [detrex](https://github.com/IDEA-Research/detrex/tree/main), a library for computer vision. Due to company policy, we cannot release the code for the model. However, we provide the detailed configuration including the model architecture, training hyperparameters, and data processing methods. We also provide the code for the evaluation of the model.

本项目基于 [detrex](https://github.com/IDEA-Research/detrex/tree/main) 构建，这是一个计算机视觉库。由于公司政策，我们无法发布模型的代码。但是，我们提供了详细的配置，包括模型架构、训练超参数和数据处理方法。我们还提供了模型评估的代码。
