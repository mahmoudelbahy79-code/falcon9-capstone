#!/usr/bin/env python3
"""Rebuild the SpaceX Falcon 9 capstone artifacts from public course sources."""
from pathlib import Path
from datetime import datetime
import io, json, re, warnings
import requests
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "data").is_dir() else HERE / "falcon9_capstone"
DATA = ROOT / "data"
OUT = ROOT / "results"
for p in [DATA, OUT, ROOT / "sql"]: p.mkdir(parents=True, exist_ok=True)
BASE = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/"
SOURCES = {
  "api_snapshot": BASE + "API_call_spacex_api.json",
  "part1": BASE + "dataset_part_1.csv",
  "part2": BASE + "dataset_part_2.csv",
  "wikipedia_revision": "https://en.wikipedia.org/w/index.php?title=List_of_Falcon_9_and_Falcon_Heavy_launches&oldid=1027686922",
  "api_docs": "https://github.com/r-spacex/SpaceX-API/blob/master/docs/launches/v4/all.md",
}

def download(url, path):
    r = requests.get(url, headers={"User-Agent":"Mozilla/5.0 (educational capstone; contact: public dataset use)"}, timeout=90)
    r.raise_for_status(); path.write_bytes(r.content); return r

# Save the source snapshot and the paired, course-curated tables.
api_path = DATA / "API_call_spacex_api.json"
part1_path = DATA / "dataset_part_1.csv"
part2_path = DATA / "dataset_part_2.csv"
for u,p in [(SOURCES["api_snapshot"],api_path),(SOURCES["part1"],part1_path),(SOURCES["part2"],part2_path)]:
    if not p.exists(): download(u,p)
raw = json.loads(api_path.read_text())
api_raw_rows = len(raw) if isinstance(raw,list) else None
base = pd.read_csv(part1_path)
df = pd.read_csv(part2_path)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Year"] = df["Date"].dt.year
if not set(df["Class"].unique()).issubset({0,1}): raise ValueError("Unexpected target labels")
df.to_csv(DATA / "falcon9_labeled.csv", index=False)

# Frozen Wikipedia version used by the course's web-scraping lab. Retain the full
# family table as a separate validation/context file; do not mix its different
# coverage or outcome coding into the labeled modeling table.
wiki_path = DATA / "wiki_falcon_family_2021_snapshot.html"
if not wiki_path.exists(): download(SOURCES["wikipedia_revision"], wiki_path)
tables = pd.read_html(io.StringIO(wiki_path.read_text(errors="replace")))
frames = []
for ti, tab in enumerate(tables):
    if "Date and time (UTC)" not in tab.columns: continue
    x = tab.copy()
    versions = [c for c in x.columns if str(c).startswith("Version, Booster")]
    if versions:
        x["Version / booster"] = x[versions].bfill(axis=1).iloc[:,0]
    else: x["Version / booster"] = ""
    x["source_table"] = ti
    frames.append(x)
if frames:
    wiki = pd.concat(frames, ignore_index=True)
    date_col = wiki["Date and time (UTC)"].astype(str).str.extract(r"^(\d{1,2} [A-Z][a-z]+ \d{4})", expand=False)
    wiki["date"] = pd.to_datetime(date_col, format="%d %B %Y", errors="coerce")
    wiki = wiki[wiki["date"].between("2010-01-01", "2020-12-31")].copy()
    cols = [c for c in ["Flight No.","date","Version / booster","Launch site","Payload[c]","Payload mass","Orbit","Customer","Launch outcome","Booster landing","source_table"] if c in wiki.columns]
    wiki[cols].to_csv(DATA / "wikipedia_falcon_family_2010_2020.csv", index=False)
    wiki_rows = len(wiki)
else:
    wiki_rows = 0

# EDA aggregates (exact raw denominators are retained).
site = df.groupby("LaunchSite")["Class"].agg(["count","sum","mean"]).reset_index()
site["success_rate_pct"] = site["mean"]*100
site.to_csv(OUT / "success_by_site.csv", index=False)
year = df.groupby("Year")["Class"].agg(["count","sum","mean"]).reset_index()
year["success_rate_pct"] = year["mean"]*100
year.to_csv(OUT / "success_by_year.csv", index=False)
orbit = df.groupby("Orbit")["Class"].agg(["count","sum","mean"]).reset_index()
orbit["success_rate_pct"] = orbit["mean"]*100
orbit.to_csv(OUT / "success_by_orbit.csv", index=False)
payload = df.groupby("Class")["PayloadMass"].agg(["count","mean","median","min","max"]).reset_index()
payload.to_csv(OUT / "payload_by_class.csv", index=False)
missing = base.isna().sum().to_dict()

# Leakage-conscious preprocessing and model comparison. The outcome, landing pad,
# booster serial, launch date/number and coordinates are excluded from predictors.
num = ["PayloadMass","Flights","Block","ReusedCount"]
cat = ["Orbit","LaunchSite","GridFins","Reused","Legs"]
X = df[num+cat]; y = df["Class"]
pre = ColumnTransformer([
    ("numeric", Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]), num),
    ("categorical", Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]), cat),
])
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)
models = {
    "Logistic regression": (LogisticRegression(max_iter=3000), {"model__C":[0.1,1,10]}),
    "SVM": (SVC(probability=True,random_state=42), {"model__C":[0.1,1,10],"model__kernel":["linear","rbf"]}),
    "Decision tree": (DecisionTreeClassifier(random_state=42), {"model__max_depth":[2,4,6,None],"model__min_samples_leaf":[1,2,4]}),
    "KNN": (KNeighborsClassifier(), {"model__n_neighbors":[3,5,7],"model__weights":["uniform","distance"]}),
    "Random forest": (RandomForestClassifier(random_state=42), {"model__n_estimators":[200],"model__max_depth":[3,5,None],"model__min_samples_leaf":[1,2]}),
}
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
rows=[]; fitted={}
for name,(est,grid) in models.items():
    pipe=Pipeline([("preprocess",pre),("model",est)])
    search=GridSearchCV(pipe,grid,cv=cv,scoring="accuracy",n_jobs=-1,refit=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); search.fit(X_train,y_train)
    pred=search.predict(X_test); proba=search.predict_proba(X_test)[:,1]
    rows.append({"model":name,"cv_accuracy":search.best_score_,"test_accuracy":accuracy_score(y_test,pred),
                 "precision":precision_score(y_test,pred,zero_division=0),"recall":recall_score(y_test,pred,zero_division=0),
                 "f1":f1_score(y_test,pred,zero_division=0),"roc_auc":roc_auc_score(y_test,proba),
                 "best_params":json.dumps(search.best_params_,ensure_ascii=False)})
    fitted[name]=(search,pred,proba)
metrics=pd.DataFrame(rows).sort_values(["test_accuracy","f1","roc_auc"],ascending=False)
metrics.to_csv(OUT / "model_comparison.csv",index=False)
best_name=metrics.iloc[0]["model"]
best_search,best_pred,best_proba=fitted[best_name]
cm=confusion_matrix(y_test,best_pred,labels=[0,1]).tolist()

# Random forest feature importances for an interpretable exploratory display.
rf_search=fitted["Random forest"][0]
rf=rf_search.best_estimator_.named_steps["model"]
try:
    names=rf_search.best_estimator_.named_steps["preprocess"].get_feature_names_out()
    importances=pd.DataFrame({"feature":names,"importance":rf.feature_importances_}).sort_values("importance",ascending=False)
    importances.to_csv(OUT / "random_forest_importance.csv",index=False)
except Exception: importances=pd.DataFrame()

summary={
 "dataset_rows":int(len(df)),"positive":int((df.Class==1).sum()),"negative":int((df.Class==0).sum()),
 "date_min":str(df.Date.min().date()),"date_max":str(df.Date.max().date()),"api_snapshot_rows":api_raw_rows,
 "wikipedia_scraped_family_rows_2010_2020":int(wiki_rows),"class_success_rate_pct":float(df.Class.mean()*100),
 "missing_base":{k:int(v) for k,v in missing.items()},"sites":site.round(6).to_dict("records"),
 "years":year.round(6).to_dict("records"),"orbits":orbit.round(6).to_dict("records"),
 "payload_by_class":payload.round(3).to_dict("records"),"model_comparison":metrics.to_dict("records"),
 "best_test_model":best_name,"best_confusion_matrix_labels_0_1":cm,
 "test_size":int(len(y_test)),"split":"stratified 80/20; random_state=42","cv":"5-fold stratified CV on training split",
 "feature_importance_top10":importances.head(10).to_dict("records") if len(importances) else [],
 "sources":SOURCES,
 "note":"Class=1 denotes the curated dataset's successful first-stage landing label; Class=0 includes all other coded outcomes and is not limited to attempted landings."
}
(OUT / "summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False))

# Interactive Folium map: aggregate launch-site markers plus every labeled
# launch record in a cluster layer. Proximity is explicitly site-to-site,
# computed geodesically from the coordinates supplied in the course dataset.
import folium
from folium.plugins import MarkerCluster
from math import radians, sin, cos, asin, sqrt
coords={"CCAFS SLC 40":(28.561857,-80.577366),"KSC LC 39A":(28.608058,-80.603956),"VAFB SLC 4E":(34.632093,-120.610829)}
def haversine_km(a,b):
    lat1,lon1=map(radians,a); lat2,lon2=map(radians,b)
    dlat=lat2-lat1; dlon=lon2-lon1
    h=sin(dlat/2)**2+cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 6371.0088*2*asin(sqrt(h))
proximity=[]
for name,a in coords.items():
    dists={other:haversine_km(a,b) for other,b in coords.items() if other!=name}
    nearest=min(dists,key=dists.get)
    proximity.append({"LaunchSite":name,"NearestOtherSite":nearest,"NearestDistanceKm":dists[nearest],"Latitude":a[0],"Longitude":a[1]})
prox=pd.DataFrame(proximity)
prox.to_csv(OUT / "site_proximity_km.csv",index=False)
center=(sum(v[0] for v in coords.values())/len(coords),sum(v[1] for v in coords.values())/len(coords))
m=folium.Map(location=center,zoom_start=3,tiles="CartoDB positron")
launch_layer=folium.FeatureGroup(name="Individual launch records (clustered)")
cluster=MarkerCluster(name="Falcon 9 launch records").add_to(launch_layer)
for _,r in df.iterrows():
    latlon=coords[r["LaunchSite"]]
    color="#00a8b5" if int(r["Class"])==1 else "#7650a8"
    popup=(f"Flight {int(r['FlightNumber'])} | {r['Date'].date()}<br>"+
           f"Site: {r['LaunchSite']}<br>Payload: {r['PayloadMass']:.0f} kg<br>Orbit: {r['Orbit']}<br>Class: {int(r['Class'])}")
    folium.CircleMarker(location=latlon,radius=5,color=color,fill=True,fill_color=color,fill_opacity=.78,
        popup=folium.Popup(popup,max_width=280),tooltip=f"Flight {int(r['FlightNumber'])} | Class {int(r['Class'])}").add_to(cluster)
launch_layer.add_to(m)
for r in site.to_dict("records"):
    latlon=coords[r["LaunchSite"]]
    nearest=prox.loc[prox.LaunchSite==r["LaunchSite"]].iloc[0]
    folium.CircleMarker(location=latlon,radius=10+int(r["count"])//5,color="#00E5FF",fill=True,fill_color="#00E5FF",fill_opacity=.45,
        popup=(f"{r['LaunchSite']}<br>Launches: {int(r['count'])}<br>Positive labels: {int(r['sum'])}<br>Rate: {r['success_rate_pct']:.1f}%<br>"+
               f"Nearest other site: {nearest.NearestOtherSite} ({nearest.NearestDistanceKm:.1f} km)"),
        tooltip=f"{r['LaunchSite']}: N={int(r['count'])}, rate={r['success_rate_pct']:.1f}%").add_to(m)
folium.LayerControl(collapsed=False).add_to(m)
m.fit_bounds([[min(v[0] for v in coords.values())-1,min(v[1] for v in coords.values())-2],[max(v[0] for v in coords.values())+1,max(v[1] for v in coords.values())+2]])
m.save(str(ROOT / "folium_launch_sites.html"))
print(json.dumps({"project":str(ROOT),"summary":summary},ensure_ascii=False,indent=2))
