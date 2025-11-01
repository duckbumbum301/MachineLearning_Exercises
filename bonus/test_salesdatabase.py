from ML_Excercises.project_retail.connectors.connector import Connector
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# CLI: default is web-only; show plots only when --plots is provided
parser = argparse.ArgumentParser(description="Retail clustering and export")
parser.add_argument("--plots", action="store_true", help="Show plots (histogram, elbow, 2D/3D)")
args = parser.parse_args()
SHOW_PLOTS = args.plots

conn=Connector(database="salesdatabase")
conn.connect()
sql="select * from customer"
df=conn.queryDataset(sql)
print(df)

sql2=("select distinct customer.CustomerId, Age, Annual_Income,Spending_Score from customer, customer_spend_score "
      "where customer.CustomerId=customer_spend_score.CustomerID")
df2=conn.queryDataset(sql2)
print(df2)

df2.columns = ['CustomerId', 'Age', 'Annual Income', 'Spending Score']

print(df2)
print(df2.head())
print(df2.describe())

def showHistogram(df, columns):
    plt.figure(1, figsize=(7, 8))
    n = 0
    for column in columns:
        n += 1
        plt.subplot(3, 1, n)
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        sns.histplot(df[column], bins=32)  # Lưu ý: distplot đã deprecated; có thể đổi sang histplot
        plt.title(f'Histogram of {column}')
    plt.show()

# Bỏ qua CustomerId
if SHOW_PLOTS:
    showHistogram(df2, df2.columns[1:])

def elbowMethod(df, columnsForElbow):
    X = df.loc[:, columnsForElbow].values
    inertia = []
    for n in range(1, 11):
        model = KMeans(n_clusters=n, init='k-means++', max_iter=500, random_state=42)
        model.fit(X)
        inertia.append(model.inertia_)

    plt.figure(figsize=(15, 6))          # đừng tái dùng figure(1)
    plt.plot(np.arange(1, 11), inertia, 'o')
    plt.plot(np.arange(1, 11), inertia, '-.', alpha=0.5)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Cluster sum of squared distances')
    plt.title('Elbow Method')
    plt.show()

# GỌI HÀM Ở NGOÀI:
if SHOW_PLOTS:
    elbowMethod(df2, ['Age', 'Spending Score'])

def runKMeans(X, cluster):
    model = KMeans(
        n_clusters=cluster,
        init='k-means++',
        max_iter=500,
        random_state=42
    )
    model.fit(X)
    labels = model.labels_
    centroids = model.cluster_centers_
    y_kmeans = model.fit_predict(X)
    return y_kmeans, centroids, labels

# Chọn 2 trục để vẽ: Age & Spending Score
columns = ['Age', 'Spending Score']
X = df2.loc[:, columns].values

cluster = 4
colors = ["red", "green", "blue", "purple", "black", "pink", "orange"]

y_kmeans, centroids, labels = runKMeans(X, cluster)
print(y_kmeans)
print(centroids)
print(labels)

df2['cluster'] = labels

def visualizeKMeans(X, y_kmeans, cluster, title, xlabel, ylabel, colors):
    plt.figure(figsize=(10, 10))
    for i in range(cluster):
        plt.scatter(
            X[y_kmeans == i, 0],
            X[y_kmeans == i, 1],
            s=100,
            c=colors[i],
            label='Cluster %i' % (i + 1)
        )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

if SHOW_PLOTS:
    visualizeKMeans(
        X,
        y_kmeans,
        cluster,
        "Clusters of Customers - Age X Spending Score",
        "Age",
        "Spending Score",
        colors
    )

import plotly.express as px
from sklearn.preprocessing import StandardScaler

# 1) Chọn 3 biến để vẽ 3D
columns = ['Age', 'Annual Income', 'Spending Score']

# 2) (khuyến nghị) scale trước khi KMeans
X_raw = df2.loc[:, columns].values
X = StandardScaler().fit_transform(X_raw)

# 3) Set k = 5 và train lại
cluster = 5
y_kmeans, centroids, labels = runKMeans(X, cluster)
df2['cluster'] = labels.astype(int)

# 4) Vẽ 3D với Plotly
def visualize3DKmeans(df, columns, hover_data, cluster):
    fig = px.scatter_3d(
        df, x=columns[0], y=columns[1], z=columns[2],
        color='cluster',
        hover_data=list(hover_data),
        category_orders={'cluster': list(range(cluster))}
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()

if SHOW_PLOTS:
    visualize3DKmeans(df2, columns, df2.columns, cluster)

# ====== ADD BELOW THIS LINE (after df2['cluster'] is created) ======
from pathlib import Path

# (A) Lấy ALL customers từ MySQL
def fetch_all_customers(conn) -> pd.DataFrame:
    sql = "SELECT * FROM customer"
    return conn.queryDataset(sql)

# (B) Merge nhãn cluster vào danh sách customer
def merge_customers_with_cluster(df_customers: pd.DataFrame,
                                 df_clusters: pd.DataFrame,
                                 customerid_left: str = "CustomerID",
                                 customerid_right: str = "CustomerId",
                                 cluster_col: str = "cluster") -> pd.DataFrame:
    need_cols = [customerid_right, cluster_col]
    missing = [c for c in need_cols if c not in df_clusters.columns]
    if missing:
        raise ValueError(f"df_clusters missing columns: {missing}")
    merged = df_customers.merge(
        df_clusters[need_cols],
        left_on=customerid_left,
        right_on=customerid_right,
        how="inner"
    )
    return merged

# (1) In ra console theo từng cluster
def print_customers_by_cluster(merged: pd.DataFrame, cluster_col: str = "cluster") -> None:
    for k, group in merged.sort_values(cluster_col).groupby(cluster_col):
        print(f"\n===== Cluster {k} | {len(group)} customers =====")
        print(group.drop(columns=[cluster_col]).to_string(index=False))

# (2) Xuất Excel: sheet tổng + mỗi cluster một sheet
def export_clusters_to_excel(merged: pd.DataFrame,
                             out_path: str = "customers_by_cluster.xlsx",
                             cluster_col: str = "cluster") -> str:
    out_path = str(Path(out_path).resolve())
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        merged.sort_values(cluster_col).to_excel(writer, sheet_name="All_Customers", index=False)
        for k, group in merged.groupby(cluster_col):
            sheet_name = f"Cluster_{k}"
            group.drop(columns=[cluster_col]).to_excel(writer, sheet_name=sheet_name, index=False)

        # Định dạng nhẹ cho đẹp
        workbook = writer.book
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#F1F5F9", "border": 1})
        cell_fmt = workbook.add_format({"border": 1})
        for name, ws in writer.sheets.items():
            df_preview = merged if name == "All_Customers" else merged.drop(columns=[cluster_col])
            for col_idx, col in enumerate(df_preview.columns):
                width = max(10, min(35, int(df_preview[col].astype(str).str.len().quantile(0.9)) + 2))
                ws.set_column(col_idx, col_idx, width, cell_fmt)
            ws.set_row(0, 20, header_fmt)
    return out_path

# (2') Tạo file HTML Bootstrap đẹp, có tab theo cluster + search
def write_customers_html(merged: pd.DataFrame,
                         out_path: str = "customers_by_cluster.html",
                         cluster_col: str = "cluster") -> str:
    out_path = str(Path(out_path).resolve())
    cols_no_cluster = [c for c in merged.columns if c != cluster_col]
    clusters = sorted(merged[cluster_col].unique())

    # CSS + HTML (Bootstrap 5, giao diện dark)
    head = """
<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Customers by Cluster</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
 body { background:#0b1220; color:#e5e7eb; }
 .navbar{ background:#0f172a; }
 .card{ background:#111827; border:1px solid #1f2937; }
 .table thead th{ background:#111827; color:#93c5fd; position:sticky; top:0; z-index:1; }
 .table tbody tr:hover{ background:#0f172a; }
 .badge-cluster{ background:#2563eb; }
 .search-input{ background:#0f172a; color:#e5e7eb; border:1px solid #374151; }
 .tab-btn{ color:#93c5fd; }
 .tab-btn.active{ background:#1f2937; color:#fff; }
</style></head><body>
<nav class="navbar navbar-dark px-3">
  <span class="navbar-brand">Customer Clusters</span>
  <a class="btn btn-outline-info" href="customers_by_cluster.xlsx">⬇ Download Excel</a>
</nav>
<div class="container py-4">
"""
    # Tabs
    tabs = ['<ul class="nav nav-pills mb-3">']
    for i, k in enumerate(clusters):
        active = "active" if i == 0 else ""
        count = int((merged[cluster_col] == k).sum())
        tabs.append(
            f'<li class="nav-item me-2">'
            f'<button class="btn tab-btn {active}" data-bs-toggle="pill" data-bs-target="#tab-{k}">'
            f'Cluster {k} <span class="badge badge-cluster ms-1">{count}</span>'
            f'</button></li>'
        )
    tabs.append('</ul>')
    tabs_html = "\n".join(tabs)

    # Search box
    search_html = """
<div class="row mb-3"><div class="col-md-6"></div>
<div class="col-md-6"><input id="search" class="form-control search-input" placeholder="Search current cluster..."></div></div>
"""

    # Tab panes with tables
    panes = ['<div class="tab-content">']
    for i, k in enumerate(clusters):
        active = "show active" if i == 0 else ""
        rows = merged.loc[merged[cluster_col] == k, cols_no_cluster]
        table_head = "<tr>" + "".join([f"<th>{c}</th>" for c in rows.columns]) + "</tr>"
        body_rows = []
        for _, r in rows.iterrows():
            tds = "".join([f"<td>{r[c]}</td>" for c in rows.columns])
            body_rows.append(f"<tr>{tds}</tr>")
        table_body = "\n".join(body_rows)
        panes.append(f"""
<div class="tab-pane fade {active}" id="tab-{k}">
  <div class="card"><div class="card-body">
    <div class="table-responsive" style="max-height:70vh;">
      <table class="table table-sm table-hover align-middle">
        <thead>{table_head}</thead>
        <tbody>{table_body}</tbody>
      </table>
    </div>
  </div></div>
</div>""")
    panes.append("</div>")  # end tab-content

    # Footer + JS
    tail = """
</div>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
<script>
const searchInput = document.getElementById('search');
searchInput.addEventListener('input', function(){
  const q = this.value.toLowerCase();
  const activePane = document.querySelector('.tab-pane.active');
  const rows = activePane.querySelectorAll('tbody tr');
  rows.forEach(tr => tr.style.display = tr.innerText.toLowerCase().includes(q) ? '' : 'none');
});
</script>
</body></html>
"""
    html = head + tabs_html + search_html + "\n".join(panes) + tail
    Path(out_path).write_text(html, encoding="utf-8")
    return out_path

# ===== Run: lấy dữ liệu, merge, in console, xuất excel & html =====
df_customers = fetch_all_customers(conn)
df_clusters  = df2[['CustomerId', 'cluster']].copy()
merged = merge_customers_with_cluster(df_customers, df_clusters)

# (1) Console
print_customers_by_cluster(merged)

# (2) Excel
excel_path = export_clusters_to_excel(merged, "customers_by_cluster.xlsx")
print("✅ Excel saved to:", excel_path)

# (2') Web tĩnh (HTML)
html_path = write_customers_html(merged, "customers_by_cluster.html")
print("✅ HTML saved to:", html_path, "\n👉 Open this file in your browser.")
# ====== END ADD ======

import webbrowser, os, socket
# Ưu tiên mở qua server nếu sẵn có, fallback sang file:// nếu không có server
web_url = f"http://127.0.0.1:8000/{Path(html_path).name}"
try:
    with socket.create_connection(("127.0.0.1", 8000), timeout=0.5):
        webbrowser.open(web_url)
        print("🌐 Opened in browser:", web_url)
except OSError:
    webbrowser.open('file://' + html_path.replace('\\','/'))
    print("🌐 Opened local file:", html_path)


