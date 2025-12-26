# Parent Confusion Matrix 输出格式优化

## 修改位置
**文件**: `/root/code/layoutlmft/examples/stage/engines/evaluator.py`

**修改范围**: 第 268-293 行（原始代码）被替换为新的方法调用和新方法实现（第 304-394 行）

## 优化内容

### 原始格式（旧版本）
```
[Evaluator Debug] Parent confusion (child_cls -> gt_parent_cls -> pred_parent_cls):
  [fstline]:
    gt=fstline (587/652): mispredict -> section:54, paraline:11
  [fstline]:
    gt=section (51/55): mispredict -> fstline:3, table:1
```

**问题**:
- 信息密集，不易扫读
- 错误分散在多行中
- 缺少可视的结构

### 新格式（优化后）
```
[Evaluator Debug] Parent Confusion Matrix:
+-------------+-------------+----------+---------------------------------------------+
| Child Class | GT Parent   | Accuracy | Mispredictions                              |
+-------------+-------------+----------+---------------------------------------------+
| fstline     | fstline     | 90% (587/652) | section:54, paraline:11                     |
| fstline     | section     | 93% (51/55)   | fstline:3, table:1                          |
| paraline    | fstline     | 99% (319/322) | paraline:2, section:1                       |
| paraline    | paraline    | 97% (284/293) | fstline:9                                   |
| section     | section     | 94% (74/79)   | fstline:4, table:1                          |
| table       | section     | 85% (11/13)   | table:2                                     |
+-------------+-------------+----------+---------------------------------------------+
```

**优势**:
- 表格式清晰展现，易于对齐对比
- 每行一条记录，结构一致
- 一目了然的列明标头
- ASCII 边框，兼容性强（无 Unicode）

## 核心特性

### 1. **按子类分组**
- 同一个 Child Class 的所有错误相邻显示
- 便于分析某一类别的误判模式

### 2. **只显示有错误的行**
- 条件: `correct < total`
- 避免清单过长，聚焦于需要改进的地方

### 3. **按错误数量排序**
- 从大到小排序：`rows.sort(key=lambda x: -x['error_count'])`
- 优先级高的问题放在前面，便于重点优化

### 4. **准确率百分比 + 绝对值**
- 格式: `90% (587/652)`
- 既直观又精确，便于评估模型性能

### 5. **ASCII 表格兼容性**
- 仅使用 `+`、`-`、`|` 等纯 ASCII 字符
- 无 Unicode 特殊字符（如 ┌ ┐ ├ ┤ 等）
- 确保在所有终端环境下都能正确显示

## 实现细节

### 新增方法
```python
def _print_parent_confusion_matrix(self, stats: List[Dict]) -> None:
```

**核心步骤**:
1. 构建三层字典: `confusion[child_cls][gt_p_cls][pred_p_cls]`
2. 遍历构建行数据，过滤无错误项
3. 计算动态列宽，自适应内容长度
4. 生成 ASCII 表格框架
5. 按错误数从大到小排序输出

**关键代码片段**:
```python
# 只显示有错误的情况
if correct < total:
    error_count = total - correct
    # ... 收集错误详情 ...
    rows.append({...})

# 按错误数量从大到小排序
rows.sort(key=lambda x: -x['error_count'])
```

## 调用关系

原调用点（第 269 行）:
```python
self._print_parent_confusion_matrix(stats)
```

此方法在 `evaluate()` 方法中被调用，用于打印调试信息（当 `debug=True` 或 `verbose=True` 时）。

## 向后兼容性

- 无 API 变更
- 仅改变输出格式，不影响数据处理逻辑
- 现有代码可直接使用，无需修改调用端

## 性能影响

- **计算复杂度**: O(n*m)，其中 n 为 stats 长度，m 为错误类型数
- **内存占用**: 增加行数据列表，占用量可控
- **整体影响**: 可忽略，仅用于调试和评估阶段

## 测试与验证

创建了测试脚本 `/root/code/layoutlmft/test_confusion_matrix.py` 来演示新格式。

示例数据包含：
- fstline -> fstline: 587/652 (90%)，误判为 section:54, paraline:11
- fstline -> section: 51/55 (93%)，误判为 fstline:3, table:1
- paraline -> fstline: 319/322 (99%)，误判为 paraline:2, section:1
- 等共 6 条混淆记录，每条都有不同程度的误判

## 修改总结

| 项目 | 详情 |
|-----|------|
| 修改文件 | `/root/code/layoutlmft/examples/stage/engines/evaluator.py` |
| 删除行数 | 26 行（268-293）|
| 增加行数 | 91 行（304-394）|
| 新增方法 | `_print_parent_confusion_matrix()` |
| 调用点 | 第 269 行 |
| 兼容性 | 完全向后兼容 |
| 依赖项 | 无新增依赖 |

