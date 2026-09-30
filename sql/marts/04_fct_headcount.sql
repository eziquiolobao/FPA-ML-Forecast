-- Headcount actual vs plan by department (plan = AOP of the matching fiscal year).
CREATE OR REPLACE TABLE fct_headcount AS
WITH actual AS (
    SELECT period, department, SUM(headcount) AS headcount, SUM(hires) AS hires,
           SUM(terminations) AS terminations
    FROM stg_headcount GROUP BY ALL
),
plan AS (
    SELECT period, department, SUM(planned_headcount) AS planned_headcount
    FROM stg_headcount_plan GROUP BY ALL
)
SELECT COALESCE(a.period, p.period) AS period, COALESCE(a.department, p.department) AS department,
       a.headcount, a.hires, a.terminations, p.planned_headcount
FROM actual a FULL OUTER JOIN plan p USING (period, department)
ORDER BY period, department;
