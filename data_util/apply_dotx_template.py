"""
使用 .dotx 模板安全覆盖 docx 标题样式

功能：
1. 按标题名称（Heading 1-9 / 标题 1-9）匹配，不依赖 styleId
2. 只替换标题样式，保留正文/列表/表格样式不动
3. 安全注入标题编号系统，不覆盖目标文档原有列表
4. 可选复制主题和字体表
5. 自动查找同目录下所有 .dotx 模板文件
6. 自动识别并转换 outlineLvl 大纲级别为标题样式
7. 输出目录：原文件名去掉"标题无编号_"前缀
8. 输出文件名：模板名.docx

用法：
  python apply_dotx_template.py <main_docx_path>

示例：
  python apply_dotx_template.py "E:\\docs\\标题无编号_批注.docx"
  python apply_dotx_template.py main.docx --no-numbering

目录结构示例：
  docs/
  ├── 标题无编号_批注.docx     ← 主文件
  ├── 标书模板.dotx            ← 自动查找
  └── 招投标模板.dotx          ← 自动查找

  输出：
  docs/批注/                   ← 去掉"标题无编号_"前缀
  ├── 标书模板.docx
  └── 招投标模板.docx
"""

import copy
import zipfile
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import sys
import argparse


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}

# 注册命名空间，避免输出时加前缀
ET.register_namespace('w', W_NS)
ET.register_namespace('r', 'http://schemas.openxmlformats.org/officeDocument/2006/relationships')
ET.register_namespace('a', 'http://schemas.openxmlformats.org/drawingml/2006/main')

STYLE_PARTS_ALWAYS_COPY = [
    "word/theme/theme1.xml",
    "word/fontTable.xml",
]


def convert_windows_path_to_wsl(path_str: str) -> str:
    """将 Windows 路径转换为 WSL 路径"""
    if not path_str:
        return path_str
    if len(path_str) >= 2 and path_str[1] == ':':
        drive_letter = path_str[0].lower()
        rest_path = path_str[2:].replace('\\', '/').lstrip('/')
        return f"/mnt/{drive_letter}/{rest_path}"
    return path_str


def is_linux_path(path_str: str) -> bool:
    """判断是否是 Linux 路径"""
    return path_str.startswith('/') or not (len(path_str) >= 2 and path_str[1] == ':')


def _read_zip_part(z: zipfile.ZipFile, name: str) -> Optional[bytes]:
    try:
        return z.read(name)
    except KeyError:
        return None


def _get_style_name(style_el: ET.Element) -> str:
    name_el = style_el.find("w:name", NS)
    if name_el is None:
        return ""
    return name_el.attrib.get(f"{{{W_NS}}}val", "")


def _is_heading_name(name: str) -> bool:
    """判断是否是标题样式名称（兼容英文/中文）"""
    return bool(re.search(r"(heading|标题)\s*([1-9])", name, re.IGNORECASE))


def _heading_level_from_name(name: str) -> Optional[int]:
    """从样式名称提取标题级别"""
    m = re.search(r"(heading|标题)\s*([1-9])", name, re.IGNORECASE)
    if not m:
        return None
    return int(m.group(2))


def _parse_xml(xml_bytes: bytes) -> ET.Element:
    return ET.fromstring(xml_bytes)


def _xml_bytes(root: ET.Element) -> bytes:
    return ET.tostring(root, encoding="UTF-8", xml_declaration=True)


def _index_styles_by_name(styles_root: ET.Element) -> Dict[str, ET.Element]:
    """按样式名称索引"""
    idx = {}
    for st in styles_root.findall("w:style", NS):
        n = _get_style_name(st)
        if n:
            idx[n] = st
    return idx


def _get_num_ids_used_by_heading_styles(styles_root: ET.Element) -> List[str]:
    """获取标题样式使用的 numId"""
    num_ids = []
    for st in styles_root.findall("w:style", NS):
        n = _get_style_name(st)
        if not _is_heading_name(n):
            continue
        numId_el = st.find(".//w:numPr/w:numId", NS)
        if numId_el is not None:
            v = numId_el.attrib.get(f"{{{W_NS}}}val")
            if v and v not in num_ids:
                num_ids.append(v)
    return num_ids


def _max_attr_int(root: ET.Element, tag: str, attr: str) -> int:
    """找到指定标签指定属性的最大整数值"""
    m = 0
    for el in root.findall(tag, NS):
        v = el.attrib.get(attr)
        if v and v.isdigit():
            m = max(m, int(v))
    return m


def _find_num(numbering_root: ET.Element, num_id: str) -> Optional[ET.Element]:
    """查找指定 numId 的 num 节点"""
    for num in numbering_root.findall("w:num", NS):
        if num.attrib.get(f"{{{W_NS}}}numId") == num_id:
            return num
    return None


def _find_abstract_num(numbering_root: ET.Element, abs_id: str) -> Optional[ET.Element]:
    """查找指定 abstractNumId 的 abstractNum 节点"""
    for an in numbering_root.findall("w:abstractNum", NS):
        if an.attrib.get(f"{{{W_NS}}}abstractNumId") == abs_id:
            return an
    return None


def _ensure_numbering_root(target_z: zipfile.ZipFile) -> ET.Element:
    """确保有 numbering.xml 根节点"""
    nb = _read_zip_part(target_z, "word/numbering.xml")
    if nb:
        return _parse_xml(nb)
    # 创建空的 numbering 根节点
    root = ET.Element(f"{{{W_NS}}}numbering")
    root.attrib[f"{{{W_NS}}}xmlns"] = W_NS
    return root


def _convert_outline_level_to_heading_style(
    document_root: ET.Element,
    styles_root: ET.Element,
    verbose: bool = False
) -> Tuple[ET.Element, int]:
    """
    将带有 outlineLvl 但没有 pStyle 的段落转换为标题样式

    outlineLvl 映射：
      0 → Heading 1
      1 → Heading 2
      2 → Heading 3
      3 → Heading 4
      ...
      8 → Heading 9

    Returns:
        (修改后的 document_root, 转换数量)
    """
    # 建立样式名称到styleId的映射
    style_name_to_id = {}
    for style in styles_root.findall("w:style", NS):
        style_id = style.attrib.get(f"{{{W_NS}}}styleId")
        name_el = style.find("w:name", NS)
        if name_el is not None and style_id:
            name = name_el.attrib.get(f"{{{W_NS}}}val", "")
            style_name_to_id[name.lower()] = style_id

    # 大纲级别到标题名称的映射
    outline_to_heading = {
        "0": "heading 1",
        "1": "heading 2",
        "2": "heading 3",
        "3": "heading 4",
        "4": "heading 5",
        "5": "heading 6",
        "6": "heading 7",
        "7": "heading 8",
        "8": "heading 9",
    }

    converted_count = 0

    for para in document_root.findall(".//w:p", NS):
        pPr = para.find("w:pPr", NS)
        if pPr is None:
            continue

        # 检查是否有 outlineLvl
        outline_el = pPr.find("w:outlineLvl", NS)
        if outline_el is None:
            continue

        outline_val = outline_el.attrib.get(f"{{{W_NS}}}val")
        if outline_val not in outline_to_heading:
            continue

        # 检查是否已经有 pStyle
        pStyle_el = pPr.find("w:pStyle", NS)
        if pStyle_el is not None:
            # 已经有样式引用，跳过
            continue

        # 找到对应的标题样式ID
        heading_name = outline_to_heading[outline_val]
        style_id = style_name_to_id.get(heading_name)

        if not style_id:
            if verbose:
                # 获取段落文本用于调试
                text_parts = []
                for t in para.findall(".//w:t", NS):
                    if t.text:
                        text_parts.append(t.text)
                text = ''.join(text_parts)[:50]
                print(f"    [警告] 找不到 {heading_name} 样式，跳过段落: {text}")
            continue

        # 创建 pStyle 元素并插入到 pPr 的开头
        new_pStyle = ET.Element(f"{{{W_NS}}}pStyle")
        new_pStyle.attrib[f"{{{W_NS}}}val"] = style_id
        pPr.insert(0, new_pStyle)

        converted_count += 1

        if verbose:
            # 获取段落文本用于调试
            text_parts = []
            for t in para.findall(".//w:t", NS):
                if t.text:
                    text_parts.append(t.text)
            text = ''.join(text_parts)[:50]
            print(f"    转换: outlineLvl={outline_val} → {heading_name} ({style_id}): {text}")

    return document_root, converted_count


def _inject_heading_numbering(
    target_numbering: ET.Element,
    template_numbering: ET.Element,
    template_styles: ET.Element,
    target_styles: ET.Element,
) -> Tuple[ET.Element, Dict[str, str]]:
    """
    安全注入标题编号系统：
    1. 从模板提取标题使用的 num/abstractNum
    2. 分配新的 numId/abstractNumId，避免冲突
    3. 映射 styleId（模板 -> 目标）
    4. 注入到目标 numbering.xml

    返回: (修改后的 target_numbering, numId 映射字典)
    """
    # 1) 找模板标题样式用到的 numId
    tpl_num_ids = _get_num_ids_used_by_heading_styles(template_styles)
    if not tpl_num_ids:
        return target_numbering, {}

    # 2) 计算目标现有最大 numId / abstractNumId
    max_num = _max_attr_int(target_numbering, "w:num", f"{{{W_NS}}}numId")
    max_abs = _max_attr_int(target_numbering, "w:abstractNum", f"{{{W_NS}}}abstractNumId")

    # 3) 建立标题 styleId 映射（按名称：heading 1..9 / 标题 1..9）
    tpl_by_name = _index_styles_by_name(template_styles)
    tgt_by_name = _index_styles_by_name(target_styles)

    heading_styleid_map: Dict[str, str] = {}  # template_styleId -> target_styleId
    for name, tpl_st in tpl_by_name.items():
        lvl = _heading_level_from_name(name)
        if lvl is None:
            continue
        tgt_st = tgt_by_name.get(name)
        if tgt_st is None:
            continue
        tpl_id = tpl_st.attrib.get(f"{{{W_NS}}}styleId")
        tgt_id = tgt_st.attrib.get(f"{{{W_NS}}}styleId")
        if tpl_id and tgt_id:
            heading_styleid_map[tpl_id] = tgt_id

    numid_map: Dict[str, str] = {}

    # 4) 对每个模板 numId，复制 num + abstractNum，分配新 ID 注入
    for old_numId in tpl_num_ids:
        tpl_num = _find_num(template_numbering, old_numId)
        if tpl_num is None:
            continue

        abs_id_el = tpl_num.find("w:abstractNumId", NS)
        if abs_id_el is None:
            continue
        old_absId = abs_id_el.attrib.get(f"{{{W_NS}}}val")
        if not old_absId:
            continue

        tpl_abs = _find_abstract_num(template_numbering, old_absId)
        if tpl_abs is None:
            continue

        # 分配新 ID
        max_num += 1
        max_abs += 1
        new_numId = str(max_num)
        new_absId = str(max_abs)
        numid_map[old_numId] = new_numId

        # 深拷贝并修改 ID
        num_copy = copy.deepcopy(tpl_num)
        abs_copy = copy.deepcopy(tpl_abs)

        num_copy.attrib[f"{{{W_NS}}}numId"] = new_numId
        num_copy.find("w:abstractNumId", NS).attrib[f"{{{W_NS}}}val"] = new_absId
        abs_copy.attrib[f"{{{W_NS}}}abstractNumId"] = new_absId

        # 把 abstractNum 各级里绑定的 pStyle（模板 styleId）映射成目标 styleId
        for pstyle in abs_copy.findall(".//w:pStyle", NS):
            v = pstyle.attrib.get(f"{{{W_NS}}}val")
            if v and v in heading_styleid_map:
                pstyle.attrib[f"{{{W_NS}}}val"] = heading_styleid_map[v]

        # 注入到目标 numbering
        target_numbering.append(abs_copy)
        target_numbering.append(num_copy)

    return target_numbering, numid_map


def _replace_heading_styles_only(
    target_styles: ET.Element,
    template_styles: ET.Element,
    numid_map: Dict[str, str],
    verbose: bool = False,
) -> Tuple[ET.Element, int, Dict[str, str]]:
    """
    只替换标题 1-9 样式（按名称匹配），保留其他样式不动
    如果有 numid_map，则更新标题样式中的 numId 引用

    Returns:
        (修改后的 target_styles, 替换数量, 详细信息)
    """
    tgt_by_name = _index_styles_by_name(target_styles)
    tpl_by_name = _index_styles_by_name(template_styles)

    replaced_count = 0
    details = {
        "template_headings": [],
        "target_headings": [],
        "matched": [],
        "not_matched": [],
    }

    # 收集模板和目标的标题样式
    for name in tpl_by_name.keys():
        if _is_heading_name(name):
            details["template_headings"].append(name)

    for name in tgt_by_name.keys():
        if _is_heading_name(name):
            details["target_headings"].append(name)

    for name, tpl_style in tpl_by_name.items():
        if not _is_heading_name(name):
            continue

        tgt_style = tgt_by_name.get(name)
        if tgt_style is None:
            # 目标没有该标题样式：直接新增（少见）
            target_styles.append(copy.deepcopy(tpl_style))
            replaced_count += 1
            details["not_matched"].append(f"{name} (模板有，目标无，已新增)")
            continue

        # 保持目标 styleId 不变（防止引用错位）
        tgt_styleId = tgt_style.attrib.get(f"{{{W_NS}}}styleId")
        tpl_copy = copy.deepcopy(tpl_style)
        tpl_copy.attrib[f"{{{W_NS}}}styleId"] = tgt_styleId

        # 修正 basedOn 引用：如果模板样式基于另一个标题，需要映射到目标的对应标题
        basedOn_el = tpl_copy.find('w:basedOn', NS)
        if basedOn_el is not None:
            tpl_basedOn_styleId = basedOn_el.attrib.get(f"{{{W_NS}}}val")
            # 找到模板中这个 basedOn 对应的样式名称
            for tpl_st in template_styles.findall('w:style', NS):
                if tpl_st.attrib.get(f"{{{W_NS}}}styleId") == tpl_basedOn_styleId:
                    tpl_basedOn_name = _get_style_name(tpl_st)
                    # 如果是标题样式，找到目标中对应的 styleId
                    if _is_heading_name(tpl_basedOn_name):
                        tgt_basedOn_style = tgt_by_name.get(tpl_basedOn_name)
                        if tgt_basedOn_style is not None:
                            tgt_basedOn_styleId = tgt_basedOn_style.attrib.get(f"{{{W_NS}}}styleId")
                            basedOn_el.attrib[f"{{{W_NS}}}val"] = tgt_basedOn_styleId
                    break

        # 更新 numId（如果有映射）
        if numid_map:
            for numId_el in tpl_copy.findall(".//w:numPr/w:numId", NS):
                old = numId_el.attrib.get(f"{{{W_NS}}}val")
                if old in numid_map:
                    numId_el.attrib[f"{{{W_NS}}}val"] = numid_map[old]

        # 原位替换
        parent = target_styles
        children = list(parent)
        idx = children.index(tgt_style)
        parent.remove(tgt_style)
        parent.insert(idx, tpl_copy)
        replaced_count += 1
        details["matched"].append(name)

    # 找出目标有但模板没有的标题
    for name in details["target_headings"]:
        if name not in details["template_headings"]:
            details["not_matched"].append(f"{name} (目标有，模板无)")

    return target_styles, replaced_count, details


def apply_dotx_template(
    target_docx: Path,
    template_dotx: Path,
    output_docx: Path,
    apply_heading_numbering: bool = True,
    copy_theme_and_fonts: bool = True,
    enable_all_numbering: bool = False,
    verbose: bool = False,
) -> Tuple[bool, str, Optional[Dict]]:
    """
    用 dotx 模板安全覆盖目标 docx 的标题样式

    Returns:
        (是否成功, 信息, 详细信息字典)
    """
    try:
        with zipfile.ZipFile(template_dotx, "r") as ztpl, \
             zipfile.ZipFile(target_docx, "r") as ztgt:

            # 读取 styles.xml
            tpl_styles_b = _read_zip_part(ztpl, "word/styles.xml")
            tgt_styles_b = _read_zip_part(ztgt, "word/styles.xml")

            if not tpl_styles_b:
                return False, "模板缺少 styles.xml", None
            if not tgt_styles_b:
                return False, "目标文件缺少 styles.xml", None

            tpl_styles = _parse_xml(tpl_styles_b)
            tgt_styles = _parse_xml(tgt_styles_b)

            # 处理编号系统
            numid_map: Dict[str, str] = {}
            tgt_numbering = None

            if apply_heading_numbering:
                tpl_numbering_b = _read_zip_part(ztpl, "word/numbering.xml")
                if tpl_numbering_b:
                    tpl_numbering = _parse_xml(tpl_numbering_b)
                    tgt_numbering = _ensure_numbering_root(ztgt)
                    tgt_numbering, numid_map = _inject_heading_numbering(
                        target_numbering=tgt_numbering,
                        template_numbering=tpl_numbering,
                        template_styles=tpl_styles,
                        target_styles=tgt_styles,
                    )

            # 替换标题样式
            tgt_styles, replaced_count, details = _replace_heading_styles_only(
                target_styles=tgt_styles,
                template_styles=tpl_styles,
                numid_map=numid_map,
                verbose=verbose,
            )

            if replaced_count == 0:
                return False, "未找到可替换的标题样式", details

            # 读取 document.xml 并转换 outlineLvl 为标题样式
            tgt_doc_data = _read_zip_part(ztgt, 'word/document.xml')
            tgt_doc_root = None
            outline_converted_count = 0

            if tgt_doc_data:
                tgt_doc_root = _parse_xml(tgt_doc_data)
                # 转换 outlineLvl → pStyle
                tgt_doc_root, outline_converted_count = _convert_outline_level_to_heading_style(
                    tgt_doc_root,
                    tgt_styles,
                    verbose=verbose
                )

            # 如果启用了清除段落级编号覆盖
            removed_numpr_count = 0

            if enable_all_numbering and tgt_doc_root is not None:
                # 获取标题样式的 styleId
                heading_style_ids = set()
                for style in tgt_styles.findall('w:style', NS):
                    name_el = style.find('w:name', NS)
                    if name_el is not None:
                        name = name_el.attrib.get(f"{{{W_NS}}}val", "").lower()
                        import re
                        if re.search(r'(heading|标题)\s*[1-9]', name):
                            style_id = style.attrib.get(f"{{{W_NS}}}styleId")
                            if style_id:
                                heading_style_ids.add(style_id)

                # 清除标题段落的段落级 numPr
                for para in tgt_doc_root.findall('.//w:p', NS):
                    pPr = para.find('w:pPr', NS)
                    if pPr is None:
                        continue

                    pStyle_el = pPr.find('w:pStyle', NS)
                    if pStyle_el is None:
                        continue

                    style_id = pStyle_el.attrib.get(f"{{{W_NS}}}val")
                    if style_id in heading_style_ids:
                        numPr_el = pPr.find('w:numPr', NS)
                        if numPr_el is not None:
                            pPr.remove(numPr_el)
                            removed_numpr_count += 1

            # 重建输出 docx
            with zipfile.ZipFile(output_docx, "w", zipfile.ZIP_DEFLATED) as zout:
                for info in ztgt.infolist():
                    data = ztgt.read(info.filename)

                    # 替换修改过的部件
                    if info.filename == "word/styles.xml":
                        data = _xml_bytes(tgt_styles)
                    elif apply_heading_numbering and info.filename == "word/numbering.xml" and tgt_numbering is not None:
                        data = _xml_bytes(tgt_numbering)
                    elif info.filename == "word/document.xml" and tgt_doc_root is not None:
                        # 如果转换了 outlineLvl 或清除了 numPr，保存修改后的 document.xml
                        if outline_converted_count > 0 or removed_numpr_count > 0:
                            data = _xml_bytes(tgt_doc_root)
                    elif copy_theme_and_fonts and info.filename in STYLE_PARTS_ALWAYS_COPY:
                        tpl_part = _read_zip_part(ztpl, info.filename)
                        if tpl_part:
                            data = tpl_part

                    zout.writestr(info, data)

                # 补充缺失的部件
                if apply_heading_numbering and tgt_numbering is not None:
                    if "word/numbering.xml" not in ztgt.namelist():
                        zout.writestr("word/numbering.xml", _xml_bytes(tgt_numbering))

                if copy_theme_and_fonts:
                    for part in STYLE_PARTS_ALWAYS_COPY:
                        if part not in ztgt.namelist():
                            tpl_part = _read_zip_part(ztpl, part)
                            if tpl_part:
                                zout.writestr(part, tpl_part)

            msg = f"成功替换 {replaced_count} 个标题样式"
            if outline_converted_count > 0:
                msg += f"，转换 {outline_converted_count} 个大纲级别段落"
            if numid_map:
                msg += f"，注入 {len(numid_map)} 套编号"
            if removed_numpr_count > 0:
                msg += f"，清除 {removed_numpr_count} 个段落级编号覆盖"
            return True, msg, details

    except Exception as e:
        if output_docx.exists():
            output_docx.unlink()
        return False, f"处理失败: {e}", None


def find_template_files(directory: Path) -> List[Path]:
    """查找模板文件 (*.dotx)"""
    return list(directory.glob("*.dotx"))


def batch_apply_templates(
    target_docx: Path,
    template_files: List[Path],
    out_dir: Path,
    apply_numbering: bool = True,
    copy_theme: bool = True,
    enable_all_numbering: bool = False,
    overwrite: bool = True,
    verbose: bool = False,
) -> List[Path]:
    """批量应用模板"""
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    print(f"\n开始应用模板样式...")
    print(f"主文件: {target_docx.name}")
    print(f"模板文件数: {len(template_files)}")
    print(f"应用编号: {'是' if apply_numbering else '否'}")
    print(f"复制主题/字体: {'是' if copy_theme else '否'}")
    print(f"启用所有标题编号: {'是' if enable_all_numbering else '否'}")
    print()

    for template_file in template_files:
        # 输出文件名：模板名_原文件名.docx
        out_path = out_dir / f"{template_file.stem}_{target_docx.stem}{target_docx.suffix}"

        print(f"  [{template_file.stem}] ", end="", flush=True)

        if out_path.exists():
            if overwrite:
                print(f"[覆盖] ", end="", flush=True)
            else:
                print(f"[跳过] 文件已存在")
                continue

        success, msg, details = apply_dotx_template(
            target_docx,
            template_file,
            out_path,
            apply_heading_numbering=apply_numbering,
            copy_theme_and_fonts=copy_theme,
            enable_all_numbering=enable_all_numbering,
            verbose=verbose,
        )

        if success:
            print(f"✓ {msg}")
            if verbose and details:
                print(f"      模板标题: {', '.join(details['template_headings'])}")
                print(f"      目标标题: {', '.join(details['target_headings'])}")
                print(f"      已匹配: {', '.join(details['matched'])}")
                if details['not_matched']:
                    print(f"      未匹配: {', '.join(details['not_matched'])}")
            outputs.append(out_path)
        else:
            print(f"✗ {msg}")
            if verbose and details:
                print(f"      模板标题: {', '.join(details.get('template_headings', []))}")
                print(f"      目标标题: {', '.join(details.get('target_headings', []))}")

    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="使用 .dotx 模板安全覆盖 docx 标题样式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 默认模式（应用标题样式 + 编号）
  python apply_dotx_template.py "E:\\docs\\main.docx"

  # 只应用标题外观，不改编号
  python apply_dotx_template.py main.docx --no-numbering

  # 不复制主题和字体
  python apply_dotx_template.py main.docx --no-theme

说明：
  - 自动在同目录查找所有 .dotx 文件作为模板
  - 只替换 Heading1-9 / 标题1-9，不影响正文和其他样式
  - 安全注入编号系统，不会覆盖目标文档原有列表
  - 自动识别并转换 outlineLvl 大纲级别为标题样式
  - 输出目录：原文件名去掉"标题无编号_"前缀
  - 输出文件名：模板名.docx
        """
    )
    parser.add_argument(
        "main_file",
        help="主 docx 文件路径"
    )
    parser.add_argument(
        "--no-numbering",
        action="store_true",
        help="不应用标题编号（只改外观）"
    )
    parser.add_argument(
        "--no-theme",
        action="store_true",
        help="不复制主题和字体表"
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="如果输出文件已存在则跳过"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细的匹配信息（调试用）"
    )
    parser.add_argument(
        "--enable-all-numbering",
        action="store_true",
        help="清除标题段落的段落级编号覆盖（让所有标题都显示编号）"
    )

    args = parser.parse_args()

    # 转换路径
    original_path = args.main_file
    file_path = convert_windows_path_to_wsl(args.main_file)
    main_file = Path(file_path)

    if not is_linux_path(original_path):
        print(f"[提示] 检测到 Windows 路径")
        print(f"  原始: {original_path}")
        print(f"  转换: {file_path}\n")

    # 检查文件
    if not main_file.exists():
        print(f"[错误] 文件不存在: {main_file}")
        sys.exit(1)

    if not main_file.is_file() or main_file.suffix.lower() != ".docx":
        print(f"[错误] 不是 docx 文件: {main_file}")
        sys.exit(1)

    directory = main_file.parent

    print("=" * 80)
    print("使用 .dotx 模板安全覆盖标题样式")
    print("=" * 80)
    print(f"主文件: {main_file}")
    print(f"工作目录: {directory}")
    print()

    # 查找模板文件
    print("[1/3] 扫描模板文件 (*.dotx)...")
    template_files = find_template_files(directory)

    if not template_files:
        print("[错误] 未找到模板文件 (*.dotx)")
        sys.exit(1)

    print(f"  找到 {len(template_files)} 个模板文件:")
    for tf in template_files:
        print(f"    - {tf.name}")

    # 输出目录：使用原文件名去掉"标题无编号_"前缀作为目录名
    output_dir_name = main_file.stem
    # 去掉"标题无编号_"前缀
    if output_dir_name.startswith("标题无编号_"):
        output_dir_name = output_dir_name[len("标题无编号_"):]

    out_dir = directory / output_dir_name
    print(f"\n[2/3] 输出目录: {out_dir}")

    # 批量应用
    print(f"\n[3/3] 应用模板...")
    outputs = batch_apply_templates(
        main_file,
        template_files,
        out_dir,
        apply_numbering=not args.no_numbering,
        copy_theme=not args.no_theme,
        enable_all_numbering=args.enable_all_numbering,
        overwrite=not args.no_overwrite,
        verbose=args.verbose,
    )

    # 总结
    print("\n" + "=" * 80)
    print(f"完成！成功生成 {len(outputs)}/{len(template_files)} 个文件")
    print("=" * 80)

    if outputs:
        print("\n生成的文件:")
        for out in outputs:
            print(f"  - {out.name}")


if __name__ == "__main__":
    main()
