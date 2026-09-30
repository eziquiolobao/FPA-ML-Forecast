-- Operational drivers from HRIS and CRM.
CREATE OR REPLACE TABLE stg_headcount AS
SELECT h.period, h.cost_center, c.department, h.headcount, h.hires, h.terminations
FROM raw_hris h JOIN raw_cost_centers c USING (cost_center);

CREATE OR REPLACE TABLE stg_headcount_plan AS
SELECT p.plan_version, CAST(p.released_on AS DATE) AS released_on, p.period, p.cost_center,
       c.department, p.planned_headcount
FROM raw_hc_plan p JOIN raw_cost_centers c USING (cost_center);

CREATE OR REPLACE TABLE stg_arr AS SELECT * FROM raw_arr;
