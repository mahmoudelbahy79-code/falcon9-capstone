# Applied Data Science Capstone: Predicting Falcon 9 First-Stage Landing Success

## Submission contents

- Final English presentation PDF: included as `presentation.pdf` in the submission archive.
- `notebooks/01_data_collection_api.ipynb` — inspect the historical SpaceX API snapshot.
- `notebooks/02_data_collection_web_scraping.ipynb` — parse archived Wikipedia launch tables.
- `notebooks/03_data_wrangling.ipynb` — inspect missingness, target labels, and model features.
- `notebooks/04_eda_visualization.ipynb` — annual/site/payload visual exploration.
- `notebooks/05_eda_sql.ipynb` — reproducible SQLite aggregations.
- `notebooks/06_folium_dash.ipynb` — map and dashboard entry points.
- `notebooks/07_predictive_analysis_classification.ipynb` — stratified model comparison and cross-validation.
- `build_capstone.py` — download public course artifacts, rebuild metrics, train classifiers, and generate the Folium map.
- `dashboard.py` — interactive Dash app for site filtering, yearly rate trends, and payload analysis.
- `folium_launch_sites.html` — interactive clustered map of 90 launch records and site summaries.
- `sql/eda_queries.sql` — SQLite queries used for EDA.
- `data/` — frozen source data and curated labeled table.
- `results/` — JSON and CSV EDA/model outputs, including `site_proximity_km.csv`.

## Main findings

The labeled sample has 90 launches: 60 positive landing labels (66.7%) and 30 other coded outcomes. The observed site rates are 60.0% for CCAFS SLC 40, 77.3% for KSC LC 39A, and 76.9% for VAFB SLC 4E. The best holdout result in the fixed stratified split was Random Forest: 83.3% accuracy, F1=0.889, and ROC-AUC=0.875 on 18 test records. These results are exploratory due to the small dataset and test size.

The Folium map includes a clustered marker for every modeled launch record. Clicking a marker reveals the flight number, date, launch site, payload, orbit, and class. Site-summary markers report sample counts and label rates. `site_proximity_km.csv` calculates nearest other launch-site distance using the haversine great-circle formula; for example, KSC LC 39A and CCAFS SLC 40 are about 5.8 km apart. This is pad-to-pad proximity, not a measurement to coastlines, roads, or railways.

## Reproduce

From this directory:

```bash
python3 -m pip install -r requirements.txt
python3 build_capstone.py
python3 dashboard.py
```

Open `http://127.0.0.1:8050` for the Dash app. Open `folium_launch_sites.html` in a browser for the map. Run the notebooks in order from `notebooks/` in a Python 3 Jupyter environment.

## Target definition and limitations

`Class=1` denotes a successful first-stage landing label in the capstone dataset. `Class=0` combines other coded outcomes, which may include no landing attempt or an unspecified outcome as well as unsuccessful recovery. It is therefore not the probability of success conditional on an attempted landing. `LandingPad` has 26 missing values in the Part 1 data and was excluded from model predictors. `Outcome`, `LandingPad`, booster serial, date, flight number, and coordinates were also excluded from the predictors to reduce leakage.

The direct SpaceX API endpoint returned gateway/SSL errors from the execution environment during this run. To avoid representing unavailable data as live, the project uses the historical API-response snapshot published with IBM Skills Network materials. The archived Wikipedia scrape covers the Falcon family through the end of 2020, while the labeled modeling table has a narrower cutoff of 2020-11-05; these sources have different coverage and outcome definitions and are not combined.

The GitHub URLs included in the presentation point to a **public reference repository containing the original capstone lab notebooks**. The deliverable files authored for this submission are provided in the accompanying ZIP archive; they have not been uploaded to the user's own GitHub account.

## Public source and lab links

Reference repository: https://github.com/evgenyzorin/IBM-Applied-Data-Science-Capstone

The repository includes notebook files for API collection, Wikipedia web scraping, wrangling, visualization EDA, SQL EDA, Folium, and machine learning. The presentation includes a full direct URL for each lab notebook.

## Data sources

1. [IBM Skills Network — API response snapshot](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/API_call_spacex_api.json)
2. [IBM Skills Network — dataset_part_1.csv](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_1.csv)
3. [IBM Skills Network — dataset_part_2.csv](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv)
4. [SpaceX API v4 — all launches documentation](https://github.com/r-spacex/SpaceX-API/blob/master/docs/launches/v4/all.md)
5. [Wikipedia — Falcon 9 and Falcon Heavy launches, archived revision](https://en.wikipedia.org/w/index.php?title=List_of_Falcon_9_and_Falcon_Heavy_launches&oldid=1027686922)
