# -*- coding: utf-8 -*-
"""analysis_insights.py —— 学习行为分析与风险预警模型。

产出：
1. 学习时长集中度（基尼系数、帕累托 Top N% 贡献）
2. 均值 vs 中位数（长尾分布结论）
3. 学习时段分布（峰值小时 / 黄金时段）
4. 核心指标相关性（共线性诊断）
5. 分层画像交叉表
6. 挂科风险模型：逻辑回归 vs 随机森林（AUC / Recall / 阈值 + 名单导出）
产物：outputs/metrics.json、outputs/risk_students.csv
用法：python pipeline/analysis_insights.py
"""
import json
import pathlib
import sqlite3
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                            recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "warehouse.db"
OUT = ROOT / "outputs"
RISK_TIME, RISK_ACTION = 40, 15          # 风险定义（与业务口径保持一致）


def gini(x: np.ndarray) -> float:
    """学习时长分配的不均衡程度（0=完全平均，1=完全集中）。"""
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def main() -> int:
    OUT.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB)
    prof = pd.read_sql("SELECT * FROM ads_student_profile", conn)
    hours = pd.read_sql("SELECT * FROM ads_hour_distribution", conn)
    stage = pd.read_sql("SELECT * FROM ads_stage_structure", conn)
    valid = pd.read_sql("SELECT * FROM ads_stage_validation", conn)
    conn.close()

    metrics = {}

    # ---------- 1. 分布与集中度 ----------
    t = prof["total_time"]
    metrics["distribution"] = {
        "student_count": int(len(prof)),
        "mean_minutes": round(float(t.mean()), 1),
        "median_minutes": round(float(t.median()), 1),
        "p90_minutes": round(float(t.quantile(0.9)), 1),
        "max_minutes": round(float(t.max()), 1),
        "mean_over_median": round(float(t.mean() / t.median()), 2),
        "gini": round(gini(t.values), 4),
        "zero_time_students": int((t == 0).sum()),
        "zero_time_ratio": round(float((t == 0).mean()), 4),
    }
    metrics["pareto"] = {
        f"top_{int(p)}%": round(float(prof.loc[prof["time_pct_rank"] <= p / 100, "cum_time_share"].max()), 4)
        for p in (1, 5, 10, 20)
    }

    # ---------- 2. 时段分布 ----------
    peak = hours.sort_values("active_count", ascending=False).head(4)
    metrics["hour"] = {
        "peak_hours": peak["study_hour"].tolist(),
        "peak_active_count": int(peak["active_count"].iloc[0]),
        "peak_ratio": float(peak["active_ratio"].iloc[0]),
        "top4_hours_cum_ratio": round(float(peak["active_ratio"].sum()), 4),
        "night_ratio_0_6": round(float(hours.loc[hours["study_hour"] < 6, "active_ratio"].sum()), 4),
    }

    # ---------- 3. 相关性（共线性诊断） ----------
    metrics["correlation"] = {
        "pearson_time_actions": round(float(prof["total_time"].corr(prof["study_count"])), 4),
        "spearman_time_actions": round(float(prof["total_time"].corr(prof["study_count"], method="spearman")), 4),
    }

    # ---------- 4. 分层结构 ----------
    metrics["stage_structure"] = stage.to_dict("records")
    metrics["stage_validation"] = valid.to_dict("records")

    # ---------- 5. 风险模型 ----------
    # 5.1 规则一致性校验：标签由 total_time / study_count 直接定义，
    #     因此 AUC 接近 1 只能说明"模型可复现规则"，不能作为预测能力证据（标签泄漏诊断）
    X = prof[["total_time", "study_count"]].astype(float)
    y_rule = ((prof["total_time"] < RISK_TIME) | (prof["study_count"] < RISK_ACTION)).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_rule, test_size=0.2, random_state=42, stratify=y_rule)
    scaler = StandardScaler().fit(X_tr)

    def score(model, y_train, y_test, use_scaled=False):
        x_tr = scaler.transform(X_tr) if use_scaled else X_tr
        x_te = scaler.transform(X_te) if use_scaled else X_te
        model.fit(x_tr, y_train)
        pred = model.predict(x_te)
        prob = model.predict_proba(x_te)[:, 1]
        return {
            "auc": round(float(roc_auc_score(y_test, prob)), 4),
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        }

    lr = LogisticRegression(random_state=42, max_iter=500)
    rf = RandomForestClassifier(n_estimators=150, max_depth=5, random_state=42)
    metrics["rule_replication"] = {
        "note": "标签由 total_time / study_count 直接定义 → AUC≈1 属于标签泄漏，仅用于校验规则可复现性",
        "risk_definition": f"total_time < {RISK_TIME} 分钟 或 study_count < {RISK_ACTION} 次",
        "positive_ratio": round(float(y_rule.mean()), 4),
        "logistic_regression": score(lr, y_tr, y_te, use_scaled=True),
        "random_forest": score(rf, y_tr, y_te),
    }

    # 5.2 无泄漏的预测任务：预测"单次学习质量"是否低于全体中位数
    #     （目标变量为派生指标 avg_minutes_per_action，与两个特征不构成同义反复）
    prof["quality_label"] = (prof["avg_minutes_per_action"] < prof["avg_minutes_per_action"].median()).astype(int)
    y_q = prof["quality_label"]
    Xq_tr, Xq_te, yq_tr, yq_te = train_test_split(X, y_q, test_size=0.2, random_state=42, stratify=y_q)
    X_tr, X_te = Xq_tr, Xq_te          # 复用划分
    metrics["quality_prediction"] = {
        "target": "avg_minutes_per_action < 全体中位数（单次学习质量偏低）",
        "positive_ratio": round(float(y_q.mean()), 4),
        "logistic_regression": score(LogisticRegression(random_state=42, max_iter=500), yq_tr, yq_te, use_scaled=True),
        "random_forest": score(RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42), yq_tr, yq_te),
    }

    # 5.3 全量名单：按规则口径输出高风险学生（用于运营干预）
    rf_rule = RandomForestClassifier(n_estimators=150, max_depth=5, random_state=42).fit(X, y_rule)
    prof["risk_probability"] = rf_rule.predict_proba(X)[:, 1]
    risk_list = prof.sort_values("risk_probability", ascending=False)
    risk_list.to_csv(OUT / "risk_students.csv", index=False, encoding="utf-8-sig")
    metrics["risk_list"] = {
        "rule_high_risk_count": int(y_rule.sum()),
        "rule_high_risk_ratio": round(float(y_rule.mean()), 4),
        "top10_user_ids": risk_list["user_id"].head(10).tolist(),
    }

    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
