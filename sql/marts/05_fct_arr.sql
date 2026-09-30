-- Company ARR bridge (all entities, USD).
CREATE OR REPLACE TABLE fct_arr AS
SELECT period,
       SUM(beginning_arr) AS beginning_arr, SUM(new_arr) AS new_arr,
       SUM(expansion_arr) AS expansion_arr, SUM(contraction_arr) AS contraction_arr,
       SUM(churn_arr) AS churn_arr, SUM(ending_arr) AS ending_arr, SUM(customers) AS customers
FROM stg_arr GROUP BY period ORDER BY period;
