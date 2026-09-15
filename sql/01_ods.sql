-- ============================================================================
-- 01_ods.sql  ODS 层：原始数据落库（SQLite / MySQL 兼容）
-- 数据来源：student_learning_summary.csv（学生明细）、hour_activity_summary.csv（时段活跃）、
--           stage_summary.csv（学习强度分层汇总）
-- ============================================================================

DROP TABLE IF EXISTS ods_student_learning;
CREATE TABLE ods_student_learning (
    user_id     INTEGER,          -- 学生 ID
    total_time  REAL,             -- 累计学习时长（分钟）
    study_count INTEGER           -- 累计学习交互次数
);

DROP TABLE IF EXISTS ods_hour_activity;
CREATE TABLE ods_hour_activity (
    study_hour   INTEGER,         -- 一天中的小时（0-23）
    active_count INTEGER          -- 该小时的学习交互次数
);

DROP TABLE IF EXISTS ods_stage_summary;
CREATE TABLE ods_stage_summary (
    stage_name    TEXT,           -- 学习强度分层（轻度/中度/深度/沉浸）
    student_count INTEGER        -- 该层学生数
);

DROP TABLE IF EXISTS ods_class_ranking;
CREATE TABLE ods_class_ranking (
    class_name   TEXT,
    avg_time     REAL,
    student_num  INTEGER
);

DROP TABLE IF EXISTS ods_exam_wrong;
CREATE TABLE ods_exam_wrong (
    question_id TEXT,
    wrong_count INTEGER
);
