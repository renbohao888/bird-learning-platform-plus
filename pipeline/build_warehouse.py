# -*- coding: utf-8 -*-
"""build_warehouse.py —— 把 CSV 原始数据落进 SQLite 数仓，并执行 SQL 指标层。

分层：ODS（原始落库）→ ADS（指标体系，全部由 SQL 窗口函数计算）
用法：python pipeline/build_warehouse.py
产物：data/warehouse.db、docs/数据质量报告.md
"""
import pathlib
import sqlite3
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SQL = ROOT / "sql"
DOCS = ROOT / "docs"
DB = DATA / "warehouse.db"


def read_csv(name: str) -> pd.DataFrame:
    """按 utf-8 → gbk 顺序尝试读取，避免中文乱码；数据文件位于仓库根目录（或 data/）。"""
    for base in (ROOT, ROOT / "data"):
        path = base / name
        if not path.exists():
            continue
        for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
            try:
                return pd.read_csv(path, encoding=enc)
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        return pd.read_csv(path, encoding="latin-1")
    return pd.DataFrame()


def run_sql_file(conn: sqlite3.Connection, path: pathlib.Path) -> None:
    sql = path.read_text(encoding="utf-8")
    for stmt in (s.strip() for s in sql.split(";")):
        if stmt and not stmt.startswith("--"):
            conn.execute(stmt)


def load_ods(conn: sqlite3.Connection) -> dict:
    """写入 ODS 表，并返回数据质量校验结果。"""
    students = read_csv("student_learning_summary.csv").rename(
        columns={"userId": "user_id", "total_time": "total_time", "study_count": "study_count"}
    )
    hours = read_csv("hour_activity_summary.csv")
    stages = read_csv("stage_summary.csv").rename(columns={"stage": "stage_name"})
    classes = read_csv("class_ranking_summary.csv")
    exams = read_csv("exam_wrong_questions.csv")

    quality = []
    # 基本信息
    quality.append(("student_learning_summary", len(students), students.columns.tolist()))
    quality.append(("hour_activity_summary", len(hours), hours.columns.tolist()))
    quality.append(("stage_summary", len(stages), stages.columns.tolist()))

    # 缺失 / 重复 / 异常
    dup = int(students.duplicated(subset=["user_id"]).sum())
    missing = int(students[["user_id", "total_time", "study_count"]].isna().sum().sum())
    neg = int(((students["total_time"] < 0) | (students["study_count"] < 0)).sum())
    zero_time = int((students["total_time"] == 0).sum())
    quality.extend([
        ("缺失值总数（三个核心字段）", missing, ""),
        ("重复 userId 条数", dup, ""),
        ("异常值（负时长/负次数）", neg, ""),
        ("零学习时长学生数", zero_time, ""),
        ("时段表覆盖小时数", len(hours), f"{hours['study_hour'].min()} - {hours['study_hour'].max()}"),
        ("时段活跃总量（= 交互次数总和）", int(hours["active_count"].sum()), ""),
        ("明细表交互次数总和", int(students["study_count"].sum()), "两者应一致，用于口径校验"),
        ("分层表学生数合计", int(stages["student_count"].sum()), "应等于明细表行数"),
    ])

    students.to_sql("ods_student_learning", conn, if_exists="replace", index=False)
    hours.to_sql("ods_hour_activity", conn, if_exists="replace", index=False)
    stages.to_sql("ods_stage_summary", conn, if_exists="replace", index=False)
    if not classes.empty:
        classes.to_sql("ods_class_ranking", conn, if_exists="replace", index=False)
    if not exams.empty:
        exams.to_sql("ods_exam_wrong", conn, if_exists="replace", index=False)

    return {"quality": quality, "students": students, "hours": hours, "stages": stages}


def main() -> int:
    DATA.mkdir(exist_ok=True)
    DOCS.mkdir(exist_ok=True)
    if DB.exists():
        DB.unlink()
    conn = sqlite3.connect(DB)
    try:
        run_sql_file(conn, SQL / "01_ods.sql")
        info = load_ods(conn)
        run_sql_file(conn, SQL / "02_ads_metrics.sql")
        conn.commit()

        stage = pd.read_sql("SELECT * FROM ads_stage_structure", conn)
        conc = pd.read_sql("SELECT * FROM ads_concentration", conn)
        valid = pd.read_sql("SELECT * FROM ads_stage_validation", conn)

        # 数据质量报告
        lines = ["# 数据质量报告（自动生成）", "", "## 1. 数据概况", "",
                 "| 表 | 行数 | 字段 |", "| --- | --- | --- |"]
        for name, rows, cols in info["quality"][:3]:
            lines.append(f"| {name} | {rows} | {', '.join(cols)} |")
        lines += ["", "## 2. 质量校验项", "", "| 校验项 | 结果 | 说明 |", "| --- | --- | --- |"]
        for name, val, note in info["quality"][3:]:
            lines.append(f"| {name} | {val} | {note} |")
        lines += ["", "## 3. 口径一致性校验（分档结果 vs 官方汇总）", "",
                  "| 分层 | 官方人数 | 计算人数 | 差值 | 结论 |", "| --- | --- | --- | --- | --- |"]
        for _, r in valid.iterrows():
            lines.append(f"| {r['stage_name']} | {r['official_count']} | {r['computed_count']} | "
                         f"{r['diff']} | {r['check_result']} |")
        (DOCS / "数据质量报告.md").write_text("\n".join(lines), encoding="utf-8")

        print("分层结构：")
        print(stage.to_string(index=False))
        print("\n集中度（头部学生时长贡献）：")
        print(conc.to_string(index=False))
        print("\n口径校验：")
        print(valid.to_string(index=False))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
