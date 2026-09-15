# -*- coding: utf-8 -*-
"""make_report.py —— 读取指标结果，自动生成分析报告（docs/分析报告.md）。

报告结构：业务问题 → 数据说明 → 关键发现（含数字与业务含义）→ 可落地建议 → 局限 → 复现方式
用法：python pipeline/make_report.py
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
M = json.loads((ROOT / "outputs" / "metrics.json").read_text(encoding="utf-8"))
D, P, H, C = M["distribution"], M["pareto"], M["hour"], M["correlation"]
STAGE = {r["stage_name"]: r for r in M["stage_structure"]}
QA = M["quality_prediction"]
RULE = M["rule_replication"]
RISK = M["risk_list"]


def stage_of(key: str) -> dict:
    for name, row in STAGE.items():
        if key in name:
            return row
    return {}


s_immerse, s_deep = stage_of("沉浸"), stage_of("深度")
s_mid, s_light = stage_of("中度"), stage_of("轻度")
awaken = s_mid["student_count"] + s_light["student_count"]
peaks = "、".join(f"{h} 点" for h in H["peak_hours"])

report = f"""# 学习行为数据分析报告（自动生成）

> 数据口径：{D['student_count']:,} 名学生的累计学习时长与交互次数、全员 24 小时活跃分布、官方学习强度分层汇总
> 生成方式：`python pipeline/build_warehouse.py && python pipeline/analysis_insights.py && python pipeline/make_report.py`

---

## 一、业务问题

1. 学习投入在人群中的分布是否均衡？现有「学习强度分层」能否支撑资源投放？
2. 学生的学习时间集中在哪些时段？督学与服务资源应该投在什么时候？
3. 如何提前识别低投入学生，并给出可执行的干预名单？
4. 现有报表口径是否可靠、可复现？

## 二、数据说明

| 数据表 | 行数 | 字段 | 用途 |
| --- | --- | --- | --- |
| student_learning_summary | {D['student_count']:,} | userId / total_time / study_count | 学生投入明细 |
| hour_activity_summary | 24 | study_hour / active_count | 时段活跃分布 |
| stage_summary | 4 | stage / student_count | 官方学习强度分层汇总（口径校验用） |

数据质量：缺失 0、重复 0、负值 0；**{D['zero_time_students']} 人（{D['zero_time_ratio']:.2%}）存在交互但学习时长为 0**（计时埋点疑点）。
口径校验：分档重算结果与官方汇总 **4 项全部 PASS**（详见 `docs/数据质量报告.md`）。

## 三、关键发现

### 发现 1｜学习投入极度不均衡，**均值会误导决策**
- 平均学习时长 **{D['mean_minutes']:,} 分钟**，中位数仅 **{D['median_minutes']} 分钟**，均值是 **{D['mean_over_median']} 倍**；P90 为 {D['p90_minutes']:,} 分钟，最大 {D['max_minutes']:,} 分钟。
- 学习时长**基尼系数 {D['gini']}**；**Top 1% 学生贡献 {P['top_1%']:.1%}** 的学习时长，Top 10% 贡献 **{P['top_10%']:.1%}**，Top 20% 贡献 **{P['top_20%']:.1%}**。
- **业务含义**：用"人均学习时长"做考核或对外口径会系统性高估典型学生的投入；应改用**中位数 / 分位数 / 分层占比**表述，资源投放也应针对分层而非平均值。

### 发现 2｜分层两极分化，存在 **{awaken:,} 人（{awaken / D['student_count']:.1%}）的"可唤醒池"**
| 分层 | 人数 | 占比 | 人均时长(分钟) | 人均交互次数 |
| --- | --- | --- | --- | --- |
| {s_immerse['stage_name']} | {s_immerse['student_count']:,} | {s_immerse['student_ratio']:.1%} | {s_immerse['avg_time']:,} | {s_immerse['avg_actions']:,} |
| {s_deep['stage_name']} | {s_deep['student_count']:,} | {s_deep['student_ratio']:.1%} | {s_deep['avg_time']} | {s_deep['avg_actions']} |
| {s_mid['stage_name']} | {s_mid['student_count']:,} | {s_mid['student_ratio']:.1%} | {s_mid['avg_time']} | {s_mid['avg_actions']} |
| {s_light['stage_name']} | {s_light['student_count']:,} | {s_light['student_ratio']:.1%} | {s_light['avg_time']} | {s_light['avg_actions']} |

- 沉浸层人均 {s_immerse['avg_time']:,} 分钟（≈ {s_immerse['avg_time'] / 60:.1f} 小时），轻度层人均仅 {s_light['avg_time']} 分钟，**相差约 {s_immerse['avg_time'] / max(s_light['avg_time'], 0.01):.0f} 倍**。
- **业务含义**：轻度 + 中度合计 {awaken:,} 人，是投入产出比最高的运营对象——把"分钟级"用户提升到"小时级"，比让沉浸层再多学 100 小时更容易见效。

### 发现 3｜学习行为高度"夜间化"，服务时段要跟着行为走
- 活跃峰值出现在 **{H['peak_hours'][0]} 点**（占全天 {H['peak_ratio']:.2%}），Top4 小时（{peaks}）合计占 **{H['top4_hours_cum_ratio']:.1%}**。
- **凌晨 0–6 点仍占全天活跃的 {H['night_ratio_0_6']:.1%}**。
- **业务含义**：督学触达、答疑排班、系统巡检窗口应按真实活跃曲线排布；夜间高活跃意味着夜间系统稳定性直接影响学习体验。

### 发现 4｜口径治理：分层边界 3 人差异，已定位并修复
- 官方分层汇总与按明细重算曾出现 **3 人差异**（轻度 1447 vs 1444）：根因是「0-10 分钟」档位**未明确是否包含 10 分钟端点**，而恰好有 3 名学生时长正好 = 10 分钟。
- 处理：在 `sql/02_ads_metrics.sql` 中固化端点包含关系，并新增 `ads_stage_validation` 校验表；当前 4 个分层 + 合计项 **全部 PASS**。
- **业务含义**：这类"3 人级"差异在周报里常被忽略，却会在跨部门对账、考核排名时反复出现；**口径必须写进字典并由 SQL 自动校验**。

### 发现 5｜模型诊断：原"风险模型"存在**标签泄漏**，改口径后结论才可靠
- 原实现用 `total_time`、`study_count` 预测"是否低投入"，而标签正是由这两个字段定义 → 随机森林 AUC = {RULE['random_forest']['auc']}（**同义反复，不能作为预测能力证据**）。本项目保留该模型，但降级为**规则一致性校验**（逻辑回归 AUC {RULE['logistic_regression']['auc']}、召回 {RULE['logistic_regression']['recall']}），说明规则可被稳定复现。
- 另设无泄漏任务：预测「**单次学习质量**（每次交互平均时长）是否低于中位数」——逻辑回归 **AUC {QA['logistic_regression']['auc']}** / 召回 **{QA['logistic_regression']['recall']}**，随机森林 AUC {QA['random_forest']['auc']} / 召回 {QA['random_forest']['recall']}。
- **业务含义**：① 教育预警属于"漏判成本 ≫ 误判成本"的场景，应**优先选召回更高的逻辑回归**而非默认用随机森林；② 两个核心指标相关系数高达 {C['pearson_time_actions']}（Spearman {C['spearman_time_actions']}），**特征严重共线**，靠这两个字段无法做出真实预测，必须补充新特征。

## 四、可落地建议

| # | 建议 | 依据 | 预期效果 | 验证方式 |
| --- | --- | --- | --- | --- |
| 1 | 报表口径由"人均学习时长"改为"中位数 + 分层占比" | 均值/中位数差 {D['mean_over_median']} 倍、基尼 {D['gini']} | 对外口径更真实，不被头部学生拉动 | 口径替换后对比历史表述 |
| 2 | 启动"轻度 + 中度"唤醒计划（{awaken:,} 人），按分层差异化动作 | 轻度人均仅 {s_light['avg_time']} 分钟 | 分层迁移率（中度→深度）提升 | 周/月追踪分层迁移率 |
| 3 | 督学推送与答疑排班对齐活跃曲线（{peaks} 前后），并保障 0–2 点系统稳定 | 峰值 {H['peak_hours'][0]} 点、夜间占 {H['night_ratio_0_6']:.1%} | 触达打开率与夜间可用性提升 | 推送打开率、夜间接口成功率 |
| 4 | 修正计时埋点并新增校验规则（时长=0 但有交互） | {D['zero_time_students']} 人（{D['zero_time_ratio']:.2%}）异常 | 时长口径可信度提升 | 每日异常人数趋势归零 |
| 5 | 风险名单按"高召回 + 人工复核"输出 | 逻辑回归召回 {QA['logistic_regression']['recall']} vs 随机森林 {QA['random_forest']['recall']} | 尽量不漏掉需干预学生 | 干预后分层迁移率 |

风险名单已导出：`outputs/risk_students.csv`（规则口径下高风险 **{RISK['rule_high_risk_count']:,} 人，占 {RISK['rule_high_risk_ratio']:.1%}**）。

## 五、局限与后续

1. **无时间维度**：数据为累计快照，无法计算留存、活跃趋势与真实预测；建议补「日期 + 会话」粒度行为明细（从"描述分析"走向"增长分析"的前提）。
2. **特征过少且共线**（相关系数 {C['pearson_time_actions']}）：建议补充时段分布、学习间隔、课程/章节覆盖率、错题分布等特征。
3. **样本代表性**：沉浸层占 {s_immerse['student_ratio']:.1%} 偏乐观，需确认数据是否为"活跃用户子集"，否则结论要限定适用范围。
4. **因果性**：本报告为观察性分析，运营动作效果需用 A/B 实验或准实验（DID / PSM）验证。

## 六、复现方式

```bash
pip install -r requirements.txt
python pipeline/build_warehouse.py     # CSV → SQLite 数仓 → 执行 SQL 指标层
python pipeline/analysis_insights.py   # 集中度 / 时段 / 相关性 / 风险模型
python pipeline/make_report.py         # 生成本报告
```

产物：`data/warehouse.db`、`outputs/metrics.json`、`outputs/risk_students.csv`、`docs/数据质量报告.md`、`docs/分析报告.md`
"""

(ROOT / "docs" / "分析报告.md").write_text(report, encoding="utf-8")
print("已生成 docs/分析报告.md，字数：", len(report))
sys.exit(0)
