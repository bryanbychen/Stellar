# ZJNU GPA Calculator / 浙江师范大学 GPA 计算器

**Language / 语言：** [English](#english) · [中文](#中文)

---

## English

### Overview

A local command-line tool for **Zhejiang Normal University (ZJNU)**. It converts percentile scores to a **5-point GPA**, computes weighted and arithmetic means, analyzes graduation targets, and runs **hypothetical retake scenarios**. An optional **score histogram** is available (requires a GUI-capable environment).

### Features

- Map percentile scores to **5-point GPA**; compute weighted and arithmetic averages.
- Separate **completed** courses from **predicted (in-progress)** courses; interval analysis and credits needed to reach your target GPA.
- **Retake scenario**: for scores ≤79, assume retakes to **90** and see GPA impact and remaining credit pressure.
- Optional **histogram** (`matplotlib`).

### Requirements

- Python **3.8+**
- Dependencies: see `requirements.txt`

### Installation and usage

```bash
cd gpa计算器
pip install -r requirements.txt
python GPA.py
```

You will be prompted for:

- Target graduation GPA (press Enter for the default).
- Total graduation credits required (Enter for default).
- Whether to show the histogram: `y` or `n` (default `y`).

### Report format

Core statistics use one consistent line (same pattern as in “idealized retake analysis”):

```text
GPA: x.xxxx | 算术平均: xx.xx | 加权平均: xx.xx
```

**GPA** is the credit-weighted average GPA. **算术平均** and **加权平均** are the arithmetic mean and credit-weighted mean of **percentile scores**.

#### Report sections (summary)

| Block | Description |
|--------|-------------|
| **【已规划】** | One line: total credits (completed + predicted) and required graduation credits, separated by `\|`. |
| **【已修】** | Non-predicted courses only: `GPA \| 算术平均 \| 加权平均`. |
| **【所有课程】** | Completed + predicted; same format; GPA is the overall GPA for the planned part. |
| **【区间分析】** | Two scenarios (predicted at 90 / at 85), each with a short note and one data line. |
| **【重修理想化分析】** | Assume all completed courses with score ≤79 are retaken to 90. |
| **【重修启用后，尚未规划学分】** | `remaining credits \| target GPA \| required average GPA` (4 decimal places). |
| **【毕业目标，当前规划下剩余学分】** | Without retake assumption: average GPA needed on remaining credits. |
| **【毕业规划】** | When planned credits already meet the graduation total; no “remaining credits” analysis. |

Difficulty hints (e.g. “extremely hard”) are **separate sentences** and are **not** part of the `|` data lines.

### Course data

Edit the **`COURSES_DATA`** list in the source file. No external CSV is required. Each entry is a 4-tuple:

```text
("course name", credits, percentile score, is_predicted)
```

| Field | Meaning |
|--------|---------|
| Course name | String; Chinese names are supported. |
| Credits | Float. |
| Percentile score | Float. |
| `is_predicted` | `False`: completed / graded. `True`: estimated or in progress. |

Save the file and run the program again.

### Project layout

```text
gpa计算器/
├── README.md
├── requirements.txt
└── GPA.py              # Entry point; contains COURSES_DATA
```

### Changelog

#### 2026-03-29

- Unified report lines across summary, interval analysis, retake-related blocks, and graduation targets (`GPA: … | 算术平均: … | 加权平均: …`); credits and “required average GPA” on one aligned line.
- Interval analysis: split into “predicted at 90” / “predicted at 85” blocks with arithmetic percentile mean.
- Shorter intro, prompts, and histogram confirmation text.

#### 2026-03-28

- Course data moved into embedded **`COURSES_DATA`** (avoids CSV path and Excel encoding issues).
- Removed CSV loader and unused `os` / `sys` dependencies; former CSV rows migrated into `COURSES_DATA`.

### License

If you open-source this project, add a license here (e.g. MIT, GPL-3.0) and author credits.

---

## 中文

### 概述

面向**浙江师范大学（ZJNU）**的本地命令行工具：按校内规则将百分制成绩换算为 **5 分制绩点**，计算加权 / 算术平均，输出毕业目标与**重修情景分析**；可选**成绩柱状图**（需图形界面环境）。

### 功能概览

- 按校内规则将百分制分数换算为 **5 分制绩点**，并计算加权 / 算术平均。
- 区分 **已修课程** 与 **预测（在读）课程**，输出区间分析与毕业目标所需绩点。
- **重修情景分析**：对低于 79 分的课程假设重修至 **90** 分后的 GPA 与后续学分压力。
- 可选 **成绩柱状图**（`matplotlib`）。

### 环境要求

- Python **3.8+**
- 依赖见 `requirements.txt`

### 安装与运行

```bash
cd gpa计算器
pip install -r requirements.txt
python GPA.py
```

运行后按提示输入：

- 目标毕业平均绩点（可回车使用默认值）。
- 毕业要求总学分（可回车使用默认值）。
- 是否显示柱状图：`y` / `n`（默认 `y`）。

### 报告输出格式说明

终端报告中的**核心统计行**已统一为同一套版式（与「重修理想化分析」中的数据行一致），便于对照阅读：

```text
GPA: x.xxxx | 算术平均: xx.xx | 加权平均: xx.xx
```

其中 **GPA** 为按学分加权的平均绩点；**算术平均**、**加权平均**均为**百分制分数**的算术平均与学分加权平均。

#### 报告各块含义（节选）

| 区块 | 说明 |
|------|------|
| **【已规划】** | 一行：`总学分 (已修+预测)` 与 `毕业要求总学分`，用 `\|` 分隔。 |
| **【已修】** | 仅统计非预测课程：`GPA \| 算术平均 \| 加权平均`。 |
| **【所有课程】** | 已修 + 预测课合计；其中 GPA 即「已规划部分整体平均绩点」。 |
| **【区间分析】** | 「预测课全按 90 / 全按 85」各一段说明 + 一行数据。 |
| **【重修理想化分析】** | 假设所有 ≤79 分已修课重修至 90 分后的整体表现。 |
| **【重修启用后，尚未规划学分】** | `剩余学分 \| 目标 GPA \| 所需平均绩点`（4 位小数）。 |
| **【毕业目标，当前规划下剩余学分】** | 未启用重修假设时，剩余学分达成目标所需平均绩点。 |
| **【毕业规划】** | 已规划学分已达毕业总学分时提示，无「剩余学分」分析。 |

难度提示（如「难度极高」等）仍为独立说明句，不参与上述 `|` 数据行。

### 课程数据如何维护

课程列表在源码中的 **`COURSES_DATA`** 里编辑，无需外部 CSV。每项为四元组：

```text
("课程名称", 学分, 百分制成绩, 是否预测课)
```

| 字段 | 说明 |
|------|------|
| 课程名称 | 字符串，支持中文。 |
| 学分 | 浮点数。 |
| 百分制成绩 | 浮点数。 |
| 是否预测课 | `False`：已修/已出分；`True`：预估或在读。 |

修改后保存文件，重新运行程序即可。

### 项目结构

```text
gpa计算器/
├── README.md
├── requirements.txt
└── GPA.py              # 主程序（含 COURSES_DATA）
```

### 更新记录

#### 2026-03-29

- **报告输出格式统一**：汇总区、区间分析、重修相关与毕业目标等模块的数据行，统一为 `GPA: x.xxxx | 算术平均: xx.xx | 加权平均: xx.xx`；学分与「所需平均绩点」等采用 `键: 值 | …` 单行对齐。
- **区间分析**：由单行混排改为「预测课全按 90 / 全按 85」各一段说明 + 一行数据，并补全百分制算术平均。
- **交互精简**：去掉与源码注释重复的开场长提示；输入项去掉与默认值重复的「如 4.1」类字样；柱状图确认文案缩短。

#### 2026-03-28

- **课程数据改为内置列表**：从「读取 `courses_data.csv` + 交互输入路径」改为在 **`COURSES_DATA`** 中直接维护课程，避免 CSV 编码（如 Excel 打开乱码）与路径输入带来的不便。
- **代码清理**：移除 `load_courses_from_csv`、`_parse_predicted` 及对 `os` / `sys` 的依赖；将原 CSV 中的课程迁移进 `COURSES_DATA`。
- **说明**：若日后仍用 CSV，建议 UTF-8（含 BOM）或通过 Excel「从文本/CSV」指定 UTF-8。

### 许可证

如需开源，请在此补充许可证（例如 MIT、GPL-3.0）及作者信息。
