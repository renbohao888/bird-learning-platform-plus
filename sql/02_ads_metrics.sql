-- ============================================================================
-- 02_ads_metrics.sql  ADS 层：指标体系（全部用窗口函数实现）
-- 这是本项目的核心 SQL 资产，也是数据分析岗面试最常被追问的部分
-- ============================================================================

-- ----------------------------------------------------------------------------
-- M1 学生明细 + 分层标签（口径：与 stage_summary 官方汇总一致 → 上界含端点）
--    轻度 ≤10 分钟 / 中度 ≤60 分钟 / 深度 ≤300 分钟 / 沉浸 >300 分钟
--    注：业务方原始口径为「0-10 分钟」，含 10 分钟端点；若按 <10 实现，
--        恰好 3 名时长=10 分钟的学生会被误分到中度档（口径漂移，见校验 SQL M5）
-- ----------------------------------------------------------------------------
DROP TABLE IF EXISTS ads_student_profile;
CREATE TABLE ads_student_profile AS
SELECT
    user_id,
    total_time,
    study_count,
    CASE
        WHEN total_time <= 10        THEN '0-10分钟(轻度)'
        WHEN total_time <= 60        THEN '10-60分钟(中度)'
        WHEN total_time <= 300       THEN '1-5小时(深度)'
        ELSE '5小时以上(沉浸)'
    END                                                AS stage_name,
    ROUND(total_time / NULLIF(study_count, 0), 2)       AS avg_minutes_per_action,
    -- 学习时长排名与分位（窗口函数）
    -- 学习时长排名与分位（窗口函数）
    ROW_NUMBER() OVER (ORDER BY total_time DESC)        AS time_rank,
    ROUND(PERCENT_RANK() OVER (ORDER BY total_time DESC), 4) AS time_pct_rank,
    -- 时长在全体中的占比与累计占比（用于帕累托 / 洛伦兹曲线）
    ROUND(total_time / (SELECT SUM(total_time) FROM ods_student_learning), 8) AS time_share,
    ROUND(SUM(total_time) OVER (ORDER BY total_time DESC ROWS UNBOUNDED PRECEDING)
          / (SELECT SUM(total_time) FROM ods_student_learning), 6)             AS cum_time_share
FROM ods_student_learning;

-- ----------------------------------------------------------------------------
-- M2 分层结构：人数、占比、分层内的人均/中位学习时长（用窗口函数算中位数）
-- ----------------------------------------------------------------------------
DROP TABLE IF EXISTS ads_stage_structure;
CREATE TABLE ads_stage_structure AS
WITH ranked AS (
    SELECT stage_name, total_time, study_count,
           ROW_NUMBER() OVER (PARTITION BY stage_name ORDER BY total_time) AS rn,
           COUNT(*)    OVER (PARTITION BY stage_name)                      AS cnt
    FROM ads_student_profile
)
SELECT
    stage_name,
    MAX(cnt)                                                     AS student_count,
    ROUND(MAX(cnt) * 1.0 / (SELECT COUNT(*) FROM ads_student_profile), 4) AS student_ratio,
    ROUND(AVG(total_time), 2)                                    AS avg_time,
    ROUND(AVG(CASE WHEN rn IN ((cnt + 1) / 2, (cnt + 2) / 2) THEN total_time END), 2) AS median_time,
    ROUND(AVG(study_count), 1)                                   AS avg_actions,
    ROUND(AVG(total_time / NULLIF(study_count, 0)), 2)           AS avg_minutes_per_action
FROM ranked
GROUP BY stage_name
ORDER BY avg_time DESC;

-- ----------------------------------------------------------------------------
-- M3 学习行为集中度：头部 1% / 5% / 10% 学生的时长贡献占比（帕累托分析）
-- ----------------------------------------------------------------------------
DROP TABLE IF EXISTS ads_concentration;
CREATE TABLE ads_concentration AS
SELECT 'Top 1%' AS segment, ROUND(MAX(cum_time_share), 4) AS time_share
FROM ads_student_profile WHERE time_pct_rank <= 0.01
UNION ALL
SELECT 'Top 5%', ROUND(MAX(cum_time_share), 4)
FROM ads_student_profile WHERE time_pct_rank <= 0.05
UNION ALL
SELECT 'Top 10%', ROUND(MAX(cum_time_share), 4)
FROM ads_student_profile WHERE time_pct_rank <= 0.10
UNION ALL
SELECT 'Top 20%', ROUND(MAX(cum_time_share), 4)
FROM ads_student_profile WHERE time_pct_rank <= 0.20;

-- ----------------------------------------------------------------------------
-- M4 学习时段分布：各小时活跃占比、峰值小时、以及"黄金学习时段"(前 4 小时)
-- ----------------------------------------------------------------------------
DROP TABLE IF EXISTS ads_hour_distribution;
CREATE TABLE ads_hour_distribution AS
SELECT
    study_hour,
    active_count,
    ROUND(active_count * 1.0 / (SELECT SUM(active_count) FROM ods_hour_activity), 4) AS active_ratio,
    RANK() OVER (ORDER BY active_count DESC)                                          AS activity_rank,
    -- 按小时看累计占比（判断时段是否集中）
    ROUND(SUM(active_count) OVER (ORDER BY active_count DESC ROWS UNBOUNDED PRECEDING)
          / (SELECT SUM(active_count) FROM ods_hour_activity), 4)                     AS cum_ratio,
    -- 相邻小时对比（环比波动，用于识别"异常时段"）
    ROUND((active_count - LAG(active_count) OVER (ORDER BY study_hour)) * 1.0
          / NULLIF(LAG(active_count) OVER (ORDER BY study_hour), 0), 4)               AS hour_over_hour
FROM ods_hour_activity;

-- ----------------------------------------------------------------------------
-- M5 口径一致性校验：分档聚合结果 vs stage_summary 上传的官方汇总
-- ----------------------------------------------------------------------------
DROP TABLE IF EXISTS ads_stage_validation;
CREATE TABLE ads_stage_validation AS
SELECT
    s.stage_name,
    s.student_count                       AS official_count,
    p.student_count                       AS computed_count,
    p.student_count - s.student_count     AS diff,
    CASE WHEN p.student_count = s.student_count THEN 'PASS' ELSE 'FAIL' END AS check_result
FROM ods_stage_summary s
LEFT JOIN ads_stage_structure p ON p.stage_name = s.stage_name
UNION ALL
SELECT '合计',
       (SELECT SUM(student_count) FROM ods_stage_summary),
       (SELECT COUNT(*) FROM ods_student_learning),
       (SELECT COUNT(*) FROM ods_student_learning) - (SELECT SUM(student_count) FROM ods_stage_summary),
       CASE WHEN (SELECT COUNT(*) FROM ods_student_learning) = (SELECT SUM(student_count) FROM ods_stage_summary)
            THEN 'PASS' ELSE 'FAIL' END;
