-- Load data/falcon9_labeled.csv into SQLite as table spacex.
-- Class = 1: successful first-stage landing label; Class = 0: all other coded outcomes.

-- 1) Overall sample and target balance
SELECT COUNT(*) AS launches,
       SUM(CASE WHEN "Class" = 1 THEN 1 ELSE 0 END) AS successful_landings,
       ROUND(100.0 * AVG("Class"), 1) AS success_rate_pct
FROM spacex;
-- 90 | 60 | 66.7%

-- 2) Launch counts and landing-success rates by site
SELECT LaunchSite, COUNT(*) AS launches, SUM("Class") AS successes,
       ROUND(100.0 * AVG("Class"), 1) AS success_rate_pct
FROM spacex
GROUP BY LaunchSite
ORDER BY success_rate_pct DESC;
-- KSC LC 39A: 22, 17, 77.3% | VAFB SLC 4E: 13, 10, 76.9% | CCAFS SLC 40: 55, 33, 60.0%

-- 3) Success rate by calendar year (small yearly denominators matter)
SELECT CAST(substr(Date, 1, 4) AS INTEGER) AS launch_year,
       COUNT(*) AS launches, SUM("Class") AS successes,
       ROUND(100.0 * AVG("Class"), 1) AS success_rate_pct
FROM spacex
GROUP BY launch_year
ORDER BY launch_year;

-- 4) Payload distribution by landing label
SELECT "Class", COUNT(*) AS launches,
       ROUND(AVG(PayloadMass), 1) AS mean_payload_kg,
       MIN(PayloadMass) AS min_payload_kg,
       MAX(PayloadMass) AS max_payload_kg
FROM spacex
GROUP BY "Class";

-- 5) Landing rate by orbit; inspect denominators before interpreting extremes
SELECT Orbit, COUNT(*) AS launches, SUM("Class") AS successes,
       ROUND(100.0 * AVG("Class"), 1) AS success_rate_pct
FROM spacex
GROUP BY Orbit
ORDER BY launches DESC;

-- 6) Counts by booster block and reuse count
SELECT Block, ReusedCount, COUNT(*) AS launches, ROUND(100.0*AVG("Class"),1) AS success_rate_pct
FROM spacex
GROUP BY Block, ReusedCount
ORDER BY Block, ReusedCount;
