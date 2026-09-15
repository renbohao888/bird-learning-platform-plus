# 学习行为数据分析系统（Learning Behavior Analytics）

> 一个面向**在线学习平台**的数据分析项目：从原始行为数据出发，完成 **数据质量校验 → SQL 数仓指标层 → 人群分层与集中度分析 → 时段规律挖掘 → 风险预警建模 → 自动生成分析报告** 的全链路闭环。
>
> 核心产出：11 类指标看板 + 4 份可交付文档 + 1 份可直接执行的人群干预名单（10,095 名学生 / 1,249 万次学习交互）。

---

## 一、业务问题

1. 学习投入在人群中的分布是否均衡？平台原有的「学习强度分层」能否支撑资源投放？
2. 学生什么时候在学？督学推送、答疑排班应该投在什么时段？
3. 如何识别低投入学生并给出可执行的干预名单，而不是只出一张"人数分布图"？
4. 报表口径是否可靠、可复现？跨部门对数时能否解释差异来源？

## 二、核心结论（均在 `docs/分析报告.md` 中可复现）

| # | 结论 | 关键数字 |
| --- | --- | --- |
| 1 | 学习投入**极度不均衡**，"人均学习时长"会系统性误导决策 | 均值 1,086.8 分钟 vs 中位数 256.7 分钟（**4.23 倍**）；基尼系数 **0.7274**；**Top 10% 学生贡献 53.7%** 的学习时长 |
| 2 | 分层**两极分化**，存在高性价比的"可唤醒池" | 沉浸层 47.3%（人均 2,206 分钟） vs 轻度层 14.3%（人均 2.66 分钟）；轻度+中度合计 **2,821 人（27.9%）** |
| 3 | 学习行为高度**夜间化**，服务时段需跟着行为走 | 峰值 **8 点**（占 8.12%）；Top4 小时（8/7/12/2 点）合计 **30.6%**；**凌晨 0–6 点仍占 31.3%** |
| 4 | 发现并修复**口径漂移**：分层边界 3 人差异 | 官方「0-10 分钟」挡含端点（≤10），常规实现按 <10 会误分恰好 3 名时长=10 分钟的学生；已固化口径并新增自动校验，现 **4 层 + 合计全部 PASS** |
| 5 | 诊断出原风险模型的**标签泄漏**，并给出更可靠的替代方案 | 原模型用 total_time/study_count 预测由其自身定义的低投入标签 → RF AUC=1.0（同义反复）；改用"**单次学习质量**"作无泄漏目标后：逻辑回归 **AUC 0.9889 / 召回 0.9859**，随机森林 AUC 0.9366 / 召回 0.5694 → **预警场景应选高召回模型** |
| 6 | 数据质量疑点：**1.13% 学生有交互但时长为 0** | 114 人 → 计时埋点缺失，建议加校验规则 |

## 三、指标体系与实现

指标分两层：**描述型指标**（分层、占比、集中度、时段）与**诊断型指标**（共线性、标签泄漏、口径一致性）。

核心 SQL 全部使用**窗口函数**（详见 `sql/02_ads_metrics.sql`），例如集中度（帕累托）分析：

```sql
-- 头部 N% 学生的学习时长贡献占比（用于帕累托 / 洛伦兹分析）
SELECT 'Top 10%' AS segment, ROUND(MAX(cum_time_share), 4) AS time_share
FROM ads_student_profile WHERE time_pct_rank <= 0.10;
```

分档口径与自动校验：

```sql
-- 口径：上界含端点（与业务方「0-10 分钟」表述一致）
CASE WHEN total_time <= 10 THEN '轻度' WHEN total_time <= 60 THEN '中度' ... END

-- 校验：重算结果 vs 官方汇总，输出 PASS / FAIL + 差异条数
SELECT s.stage_name, s.student_count AS official_count, p.student_count AS computed_count, ...
FROM ods_stage_summary s LEFT JOIN ads_stage_structure p USING (stage_name);
```

完整口径（含端点归属、指标复用建议）见 **`docs/指标字典.md`**。

## 四、数据与数仓分层

| 层 | 内容 | 位置 |
| --- | --- | --- |
| ODS | student_learning_summary（10,095 行）、hour_activity_summary（24 行）、stage_summary（4 行）、class_ranking / exam_wrong（可选） | `sql/01_ods.sql` |
| ADS | `ads_student_profile`（分层+排名+分位+累计占比）、`ads_stage_structure`、`ads_concentration`、`ads_hour_distribution`、`ads_stage_validation` | `sql/02_ads_metrics.sql` |
| 数仓 | SQLite（零依赖、开箱即跑；SQL 兼容 MySQL 语法，可平滑迁移） | `data/warehouse.db` |

## 五、目录结构

```
.
├─ app.py                     # Streamlit 可视化看板（分层/时段/模型/风险名单/AI 问答）
├─ sql/
│  ├─ 01_ods.sql              # ODS 建表
│  └─ 02_ads_metrics.sql      # ADS 指标层（窗口函数 + 口径校验）
├─ pipeline/
│  ├─ build_warehouse.py      # CSV → SQLite 数仓，执行 SQL，产出数据质量报告
│  ├─ analysis_insights.py    # 集中度/时段/相关性/风险模型，产出 metrics.json
│  └─ make_report.py          # 自动生成分析报告
├─ docs/
│  ├─ 指标字典.md             # 口径定义与治理规则
│  ├─ 分析报告.md             # 业务结论 + 可落地建议（自动生成）
│  └─ 数据质量报告.md         # 质量校验与口径一致性（自动生成）
└─ outputs/                   # metrics.json、risk_students.csv 等产物
```

## 六、快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 一键跑完分析链路（约 10 秒）
python pipeline/build_warehouse.py     # CSV → SQLite 数仓 → 执行 SQL 指标层 + 数据质量报告
python pipeline/analysis_insights.py   # 集中度 / 时段 / 相关性 / 风险模型 → metrics.json
python pipeline/make_report.py         # 生成 docs/分析报告.md

# 3. 启动可视化看板
streamlit run app.py
```

> AI 问答模块需要智谱 API Key：复制 `.env.example` 为 `.env` 并填写 `ZHIPU_API_KEY`
> （**密钥只从环境变量 / Streamlit secrets 读取，代码中不再硬编码**；线上在 Streamlit Cloud Secrets 中配置）。

产物清单：

| 产物 | 说明 |
| --- | --- |
| `data/warehouse.db` | SQLite 数仓（ODS + ADS 表） |
| `outputs/metrics.json` | 全部分析指标结果（可被其他系统消费） |
| `outputs/risk_students.csv` | 高风险学生名单（含风险概率，按概率降序） |
| `docs/数据质量报告.md` | 缺失/重复/异常 + 口径一致性校验结果 |
| `docs/分析报告.md` | 业务问题 → 关键发现 → 可落地建议 → 局限 |

## 七、可视化看板（Streamlit）

- 学习强度分层结构与人群体量
- 24 小时活跃分布与峰值时段
- 模型评测：逻辑回归 vs 随机森林（准确率 / 精确率 / 召回率 / F1 + 特征重要性）
- 风险模拟沙盘：输入学习时长与交互次数，实时输出风险概率
- AI 数据问答（智谱 GLM）：自然语言查询分析结论

## 八、局限与后续规划

1. **无时间维度**：现有数据为累计快照，无法计算留存 / 活跃趋势；下一步接入「日期 + 会话」粒度行为明细，补齐**留存分析、漏斗分析与真实预测模型**。
2. **特征过少且高度共线**（相关系数 0.9757）：计划补充时段偏好、学习间隔、章节覆盖率、错题分布等特征。
3. **样本代表性**：沉浸层占比 47.3% 偏乐观，需确认是否为"活跃用户子集"，否则结论需限定适用范围。
4. **因果性**：当前为观察性分析，运营动作效果需用 A/B 实验或准实验（DID / PSM）验证。
5. **工程化**：计划增加 GitHub Actions 每日调度 + 指标异常告警（阈值监测）。

## 九、技术栈

`Python` · `Pandas` · `NumPy` · `SQL（窗口函数 / 数仓分层）` · `SQLite` · `scikit-learn`（逻辑回归 / 随机森林 / AUC 评估） · `Streamlit` + `PyECharts`（可视化） · `智谱 GLM`（AI 问答）

---

## 十、项目价值小结（面试 / 评审可讲的点）

- **从"出图"到"出结论"**：每条分析都落到业务动作与验证方式，而不是只展示图表。
- **口径意识**：发现"3 人级"分档差异并定位到端点包含关系，固化口径 + 自动校验。
- **方法论素养**：主动诊断标签泄漏与特征共线性，说明"模型指标好看"不等于"有预测能力"。
- **可复现交付**：一键重跑全链路，报告与指标全部由代码生成，杜绝手工改数。
