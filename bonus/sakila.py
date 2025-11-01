import json
import os
import webbrowser
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from ML_Excercises.project_retail.connectors.connector import Connector


def fetch_customers_by_film(conn: Connector) -> pd.DataFrame:
    """Return distinct customer-film pairs for rentals in sakila.

    Columns: FilmID, FilmTitle, CustomerID, Name, Email, Active
    """
    sql = """
        SELECT DISTINCT
            f.film_id AS FilmID,
            f.title AS FilmTitle,
            c.customer_id AS CustomerID,
            CONCAT(c.first_name, ' ', c.last_name) AS Name,
            c.email AS Email,
            c.active AS Active
        FROM rental r
        INNER JOIN inventory i ON r.inventory_id = i.inventory_id
        INNER JOIN film f ON i.film_id = f.film_id
        INNER JOIN customer c ON r.customer_id = c.customer_id
        ORDER BY f.title, c.customer_id;
    """
    return conn.queryDataset(sql)


def fetch_customers_by_category(conn: Connector) -> pd.DataFrame:
    """Return distinct customers per film category.

    Columns: CategoryID, Category, CustomerID, Name, Email, Active
    """
    sql = """
        SELECT DISTINCT
            cat.category_id AS CategoryID,
            cat.name AS Category,
            c.customer_id AS CustomerID,
            CONCAT(c.first_name, ' ', c.last_name) AS Name,
            c.email AS Email,
            c.active AS Active
        FROM rental r
        INNER JOIN inventory i ON r.inventory_id = i.inventory_id
        INNER JOIN film f ON i.film_id = f.film_id
        INNER JOIN film_category fc ON f.film_id = fc.film_id
        INNER JOIN category cat ON fc.category_id = cat.category_id
        INNER JOIN customer c ON r.customer_id = c.customer_id
        ORDER BY cat.name, c.customer_id;
    """
    return conn.queryDataset(sql)


def fetch_interest_features(conn: Connector) -> pd.DataFrame:
    """Build per-customer interest metrics for clustering.

    Returns columns: CustomerID, Name, Rentals, DistinctFilms, DistinctCategories
    """
    sql = """
        SELECT
            c.customer_id AS CustomerID,
            CONCAT(c.first_name, ' ', c.last_name) AS Name,
            COUNT(r.rental_id) AS Rentals,
            COUNT(DISTINCT f.film_id) AS DistinctFilms,
            COUNT(DISTINCT cat.category_id) AS DistinctCategories
        FROM customer c
        LEFT JOIN rental r ON r.customer_id = c.customer_id
        LEFT JOIN inventory i ON r.inventory_id = i.inventory_id
        LEFT JOIN film f ON i.film_id = f.film_id
        LEFT JOIN film_category fc ON f.film_id = fc.film_id
        LEFT JOIN category cat ON fc.category_id = cat.category_id
        GROUP BY c.customer_id, c.first_name, c.last_name
        ORDER BY Rentals DESC;
    """
    return conn.queryDataset(sql)


def print_grouped(df: pd.DataFrame, group_col: str, key_cols: list[str]):
    groups = df.groupby(group_col)
    for key, g in groups:
        print(f"\n=== {group_col}: {key} (customers: {g.shape[0]}) ===")
        print(g[key_cols].to_string(index=False))


def write_customers_by_film_html(df: pd.DataFrame, out_path: Path):
    # Prepare JSON rows
    rows = df.to_dict(orient="records")
    films = (
        df.groupby("FilmTitle")
        .size()
        .sort_values(ascending=False)
        .reset_index(name="Count")
        .to_dict(orient="records")
    )

    html = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Sakila - Customers by Film</title>
  <link href=\"https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css\" rel=\"stylesheet\" />
  <style>
    body {{ background-color: #121212; color: #e0e0e0; }}
    .table thead th {{ color: #9bd; }}
    .card {{ background-color: #1e1e1e; }}
    .form-select, .form-control {{ background-color: #1a1a1a; color: #e0e0e0; }}
  </style>
</head>
<body class=\"p-3\">
  <div class=\"container\">
    <h3 class=\"mb-3\">Sakila – Customers by Film</h3>
    <div class=\"row g-2 mb-3\">
      <div class=\"col-md-6\">
        <label class=\"form-label\">Select film</label>
        <select id=\"filmSelect\" class=\"form-select\"></select>
      </div>
      <div class=\"col-md-6\">
        <label class=\"form-label\">Search customers</label>
        <input id=\"searchInput\" type=\"text\" class=\"form-control\" placeholder=\"Name or Email\" />
      </div>
    </div>
    <div class=\"mb-2\" id=\"countInfo\"></div>
    <div class=\"card\">
      <div class=\"card-body\">
        <div class=\"table-responsive\">
          <table class=\"table table-dark table-striped table-hover\" id=\"customersTable\">
            <thead>
              <tr>
                <th>Film</th><th>CustomerID</th><th>Name</th><th>Email</th><th>Active</th>
              </tr>
            </thead>
            <tbody></tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

  <script>
    const rows = {json.dumps(rows)};
    const films = {json.dumps(films)};

    const filmSelect = document.getElementById('filmSelect');
    const searchInput = document.getElementById('searchInput');
    const tbody = document.querySelector('#customersTable tbody');
    const countInfo = document.getElementById('countInfo');

    function populateFilms() {{
      filmSelect.innerHTML = films.map(f => `<option value="${{f.FilmTitle}}">${{f.FilmTitle}} (customers: ${{f.Count}})</option>`).join('');
    }}

    function render() {{
      const film = filmSelect.value || (films[0] ? films[0].FilmTitle : '');
      const term = searchInput.value.toLowerCase();
      const filtered = rows.filter(r => r.FilmTitle === film && (
        r.Name.toLowerCase().includes(term) || (r.Email || '').toLowerCase().includes(term)
      ));
      tbody.innerHTML = filtered.map(r => `<tr><td>${{r.FilmTitle}}</td><td>${{r.CustomerID}}</td><td>${{r.Name}}</td><td>${{r.Email || ''}}</td><td>${{r.Active}}</td></tr>`).join('');
      const totalForFilm = rows.filter(r => r.FilmTitle === film).length;
      countInfo.textContent = `Film: ${{film}} — showing ${{filtered.length}} of ${{totalForFilm}} customers`;
    }}

    populateFilms();
    render();
    filmSelect.addEventListener('change', render);
    searchInput.addEventListener('input', render);
  </script>
</body>
</html>
"""

    out_path.write_text(html, encoding="utf-8")


def write_customers_by_category_html(df: pd.DataFrame, out_path: Path):
    categories = (
        df.groupby(["CategoryID", "Category"]).size().reset_index(name="Count").sort_values("Category")
    )
    # Build per-category rows
    rows_by_cat: dict[int, list[dict]] = {}
    for _, r in df.iterrows():
        rows_by_cat.setdefault(int(r["CategoryID"]), []).append(
            {
                "CustomerID": int(r["CustomerID"]),
                "Name": r["Name"],
                "Email": r.get("Email", ""),
                "Active": int(r["Active"]),
            }
        )

    html = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Sakila - Customers by Category</title>
  <link href=\"https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css\" rel=\"stylesheet\" />
  <style>
    body {{ background-color: #121212; color: #e0e0e0; }}
    .table thead th {{ color: #9bd; }}
    .card {{ background-color: #1e1e1e; }}
    .nav-link {{ color: #e0e0e0; }}
  </style>
</head>
<body class=\"p-3\">
  <div class=\"container\">
    <h3 class=\"mb-3\">Sakila – Customers by Category</h3>
    <ul class=\"nav nav-pills mb-3\" id=\"catTabs\" role=\"tablist\">
      {''.join([f'<li class="nav-item" role="presentation"><button class="nav-link {'active' if i==0 else ''}" data-bs-toggle="tab" data-bs-target="#cat-{row.CategoryID}" type="button" role="tab">{row.Category} ({row.Count})</button></li>' for i, row in categories.iterrows()])}
    </ul>
    <div class=\"tab-content\">
      {''.join([f'<div class="tab-pane fade {'show active' if i==0 else ''}" id="cat-{row.CategoryID}" role="tabpanel">\n  <div class="card"><div class="card-body">\n    <div class="table-responsive">\n      <table class="table table-dark table-striped table-hover">\n        <thead><tr><th>CustomerID</th><th>Name</th><th>Email</th><th>Active</th></tr></thead>\n        <tbody>' + ''.join([f'<tr><td>{r['CustomerID']}</td><td>{r['Name']}</td><td>{r['Email']}</td><td>{r['Active']}</td></tr>' for r in rows_by_cat.get(int(row.CategoryID), [])]) + '</tbody>\n      </table>\n    </div>\n  </div></div>\n</div>' for i, row in categories.iterrows()])}
    </div>
  </div>
  <script src=\"https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js\"></script>
</body>
</html>
"""

    out_path.write_text(html, encoding="utf-8")


def write_clusters_html(df: pd.DataFrame, out_path: Path):
    clusters = (
        df.groupby("Cluster").size().reset_index(name="Count").sort_values("Cluster")
    )
    rows_by_cluster: dict[int, list[dict]] = {}
    for _, r in df.iterrows():
        rows_by_cluster.setdefault(int(r["Cluster"]), []).append(
            {
                "CustomerID": int(r["CustomerID"]),
                "Name": r["Name"],
                "Rentals": int(r["Rentals"]),
                "DistinctFilms": int(r["DistinctFilms"]),
                "DistinctCategories": int(r["DistinctCategories"]),
            }
        )

    html = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Sakila - Customers by Interest Clusters</title>
  <link href=\"https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css\" rel=\"stylesheet\" />
  <style>
    body {{ background-color: #121212; color: #e0e0e0; }}
    .table thead th {{ color: #9bd; }}
    .card {{ background-color: #1e1e1e; }}
    .nav-link {{ color: #e0e0e0; }}
  </style>
</head>
<body class=\"p-3\">
  <div class=\"container\">
    <h3 class=\"mb-3\">Sakila – Customers by Interest Clusters</h3>
    <ul class=\"nav nav-pills mb-3\" id=\"clusterTabs\" role=\"tablist\">
      {''.join([f'<li class="nav-item" role="presentation"><button class="nav-link {'active' if i==0 else ''}" data-bs-toggle="tab" data-bs-target="#cluster-{row.Cluster}" type="button" role="tab">Cluster {row.Cluster} ({row.Count})</button></li>' for i, row in clusters.iterrows()])}
    </ul>
    <div class=\"tab-content\">
      {''.join([f'<div class="tab-pane fade {'show active' if i==0 else ''}" id="cluster-{row.Cluster}" role="tabpanel">\n  <div class="card"><div class="card-body">\n    <div class="table-responsive">\n      <table class="table table-dark table-striped table-hover">\n        <thead><tr><th>CustomerID</th><th>Name</th><th>Rentals</th><th>DistinctFilms</th><th>DistinctCategories</th></tr></thead>\n        <tbody>' + ''.join([f'<tr><td>{r['CustomerID']}</td><td>{r['Name']}</td><td>{r['Rentals']}</td><td>{r['DistinctFilms']}</td><td>{r['DistinctCategories']}</td></tr>' for r in rows_by_cluster.get(int(row.Cluster), [])]) + '</tbody>\n      </table>\n    </div>\n  </div></div>\n</div>' for i, row in clusters.iterrows()])}
    </div>
  </div>
  <script src=\"https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js\"></script>
</body>
</html>
"""

    out_path.write_text(html, encoding="utf-8")


def export_excel(df: pd.DataFrame, path: Path, sheet_name: str):
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)


def cluster_customers(features_df: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    X = features_df[["Rentals", "DistinctFilms", "DistinctCategories"]].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    out = features_df.copy()
    out["Cluster"] = labels
    return out


def open_on_server(relative_filename: str):
    base = "http://127.0.0.1:8000/"
    url = base + relative_filename.replace("\\", "/")
    try:
        webbrowser.open(url, new=2)
    except Exception:
        # Fallback to file path
        webbrowser.open(str(Path(__file__).parent / relative_filename), new=2)


def main():
    # Always connect to sakila (use credentials from test_connection.py)
    conn = Connector(server="localhost", port=3306, database="sakila", username="root", password="123456")
    if conn.connect() is None:
        print("[ERROR] Không thể kết nối tới MySQL 'sakila'. Kiểm tra server/username/password trong connectors/connector.py")
        return

    tests_dir = Path(__file__).parent
    film_html = tests_dir / "sakila_customers_by_film.html"
    cat_html = tests_dir / "sakila_customers_by_category.html"
    cluster_html = tests_dir / "sakila_customers_by_interest_clusters.html"

    print("Fetching customers by film ...")
    df_film = fetch_customers_by_film(conn)
    if df_film is None:
        print("[ERROR] Truy vấn khách theo phim thất bại.")
        return
    print_grouped(df_film, "FilmTitle", ["CustomerID", "Name", "Email", "Active"])
    print(f"\nWriting HTML: {film_html}")
    write_customers_by_film_html(df_film, film_html)
    export_excel(df_film, tests_dir / "sakila_customers_by_film.xlsx", "CustomersByFilm")

    print("\nFetching customers by category ...")
    df_cat = fetch_customers_by_category(conn)
    if df_cat is None:
        print("[ERROR] Truy vấn khách theo category thất bại.")
        return
    print_grouped(df_cat, "Category", ["CustomerID", "Name", "Email", "Active"])
    print(f"\nWriting HTML: {cat_html}")
    write_customers_by_category_html(df_cat, cat_html)
    export_excel(df_cat, tests_dir / "sakila_customers_by_category.xlsx", "CustomersByCategory")

    print("\nComputing interest features and clustering ...")
    features = fetch_interest_features(conn)
    if features is None:
        print("[ERROR] Truy vấn đặc trưng quan tâm thất bại.")
        return
    clustered = cluster_customers(features, k=4)
    print("Cluster sizes:")
    print(clustered.groupby("Cluster").size().to_string())
    print(f"\nWriting HTML: {cluster_html}")
    write_clusters_html(clustered, cluster_html)
    export_excel(clustered, tests_dir / "sakila_customers_clusters.xlsx", "Clusters")

    # Open category page by default for easier overview
    open_on_server("sakila_customers_by_category.html")


if __name__ == "__main__":
    main()
import json
import os
import webbrowser
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from ML_Excercises.project_retail.connectors.connector import Connector


def fetch_customers_by_film(conn: Connector) -> pd.DataFrame:
    sql = """
        SELECT DISTINCT
            f.film_id AS FilmID,
            f.title AS FilmTitle,
            c.customer_id AS CustomerID,
            CONCAT(c.first_name, ' ', c.last_name) AS Name,
            c.email AS Email,
            c.active AS Active
        FROM rental r
        INNER JOIN inventory i ON r.inventory_id = i.inventory_id
        INNER JOIN film f ON i.film_id = f.film_id
        INNER JOIN customer c ON r.customer_id = c.customer_id
        ORDER BY f.title, c.customer_id;
    """
    return conn.queryDataset(sql)


def fetch_customers_by_category(conn: Connector) -> pd.DataFrame:
    sql = """
        SELECT DISTINCT
            cat.category_id AS CategoryID,
            cat.name AS Category,
            c.customer_id AS CustomerID,
            CONCAT(c.first_name, ' ', c.last_name) AS Name,
            c.email AS Email,
            c.active AS Active
        FROM rental r
        INNER JOIN inventory i ON r.inventory_id = i.inventory_id
        INNER JOIN film f ON i.film_id = f.film_id
        INNER JOIN film_category fc ON f.film_id = fc.film_id
        INNER JOIN category cat ON fc.category_id = cat.category_id
        INNER JOIN customer c ON r.customer_id = c.customer_id
        ORDER BY cat.name, c.customer_id;
    """
    return conn.queryDataset(sql)


def fetch_interest_features(conn: Connector) -> pd.DataFrame:
    sql = """
        SELECT
            c.customer_id AS CustomerID,
            CONCAT(c.first_name, ' ', c.last_name) AS Name,
            COUNT(r.rental_id) AS Rentals,
            COUNT(DISTINCT f.film_id) AS DistinctFilms,
            COUNT(DISTINCT cat.category_id) AS DistinctCategories
        FROM customer c
        LEFT JOIN rental r ON r.customer_id = c.customer_id
        LEFT JOIN inventory i ON r.inventory_id = i.inventory_id
        LEFT JOIN film f ON i.film_id = f.film_id
        LEFT JOIN film_category fc ON f.film_id = fc.film_id
        LEFT JOIN category cat ON fc.category_id = cat.category_id
        GROUP BY c.customer_id, c.first_name, c.last_name
        ORDER BY Rentals DESC;
    """
    return conn.queryDataset(sql)


def print_grouped(df: pd.DataFrame, group_col: str, key_cols: list[str]):
    groups = df.groupby(group_col)
    for key, g in groups:
        print(f"\n=== {group_col}: {key} (customers: {g.shape[0]}) ===")
        print(g[key_cols].to_string(index=False))


def write_customers_by_film_html(df: pd.DataFrame, out_path: Path):
    rows = df.to_dict(orient="records")
    films = (
        df.groupby("FilmTitle").size().sort_values(ascending=False).reset_index(name="Count").to_dict(orient="records")
    )

    html = f"""
<!DOCTYPE html><html lang=\"en\"><head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Sakila - Customers by Film</title>
  <link href=\"https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css\" rel=\"stylesheet\" />
  <style>body{{background:#121212;color:#e0e0e0}}.card{{background:#1e1e1e}}</style>
</head><body class=\"p-3\"><div class=\"container\">
  <h3 class=\"mb-3\">Sakila – Customers by Film</h3>
  <div class=\"row g-2 mb-3\">
    <div class=\"col-md-6\"><label class=\"form-label\">Select film</label><select id=\"filmSelect\" class=\"form-select\"></select></div>
    <div class=\"col-md-6\"><label class=\"form-label\">Search customers</label><input id=\"searchInput\" type=\"text\" class=\"form-control\" placeholder=\"Name or Email\" /></div>
  </div>
  <div class=\"mb-2\" id=\"countInfo\"></div>
  <div class=\"card\"><div class=\"card-body\"><div class=\"table-responsive\">
    <table class=\"table table-dark table-striped table-hover\" id=\"customersTable\"><thead><tr>
      <th>Film</th><th>CustomerID</th><th>Name</th><th>Email</th><th>Active</th></tr></thead><tbody></tbody></table>
  </div></div></div>
</div>
<script>
const rows={json.dumps(rows)};const films={json.dumps(films)};
const filmSelect=document.getElementById('filmSelect');const searchInput=document.getElementById('searchInput');const tbody=document.querySelector('#customersTable tbody');const countInfo=document.getElementById('countInfo');
function populateFilms(){{filmSelect.innerHTML=films.map(f=>`<option value="${{f.FilmTitle}}">${{f.FilmTitle}} (customers: ${{f.Count}})</option>`).join('')}}
function render(){{const film=filmSelect.value||(films[0]?films[0].FilmTitle:'');const term=searchInput.value.toLowerCase();const filtered=rows.filter(r=>r.FilmTitle===film&&(r.Name.toLowerCase().includes(term)||String(r.Email||'').toLowerCase().includes(term)));tbody.innerHTML=filtered.map(r=>`<tr><td>${{r.FilmTitle}}</td><td>${{r.CustomerID}}</td><td>${{r.Name}}</td><td>${{r.Email || ''}}</td><td>${{r.Active}}</td></tr>`).join('');const totalForFilm=rows.filter(r=>r.FilmTitle===film).length;countInfo.textContent=`Film: ${{film}} — showing ${{filtered.length}} of ${{totalForFilm}} customers`}}
populateFilms();render();filmSelect.addEventListener('change',render);searchInput.addEventListener('input',render);
</script></body></html>
"""
    out_path.write_text(html, encoding="utf-8")


def write_customers_by_category_html(df: pd.DataFrame, out_path: Path):
    categories = df.groupby(["CategoryID", "Category"]).size().reset_index(name="Count").sort_values("Category")
    rows_by_cat: dict[int, list[dict]] = {}
    for _, r in df.iterrows():
        rows_by_cat.setdefault(int(r["CategoryID"]), []).append({
            "CustomerID": int(r["CustomerID"]),
            "Name": r["Name"],
            "Email": r.get("Email", ""),
            "Active": int(r["Active"]),
        })

    def table_for(cat_id: int) -> str:
        rows = rows_by_cat.get(cat_id, [])
        body = "".join([f"<tr><td>{r['CustomerID']}</td><td>{r['Name']}</td><td>{r['Email']}</td><td>{r['Active']}</td></tr>" for r in rows])
        return f"<table class=\"table table-dark table-striped table-hover\"><thead><tr><th>CustomerID</th><th>Name</th><th>Email</th><th>Active</th></tr></thead><tbody>{body}</tbody></table>"

    tabs = []
    panes = []
    for i, row in enumerate(categories.itertuples(index=False)):
        active = "active" if i == 0 else ""
        show = "show active" if i == 0 else ""
        tabs.append(f"<li class=\"nav-item\" role=\"presentation\"><button class=\"nav-link {active}\" data-bs-toggle=\"tab\" data-bs-target=\"#cat-{row.CategoryID}\" type=\"button\" role=\"tab\">{row.Category} ({row.Count})</button></li>")
        panes.append(f"<div class=\"tab-pane fade {show}\" id=\"cat-{row.CategoryID}\" role=\"tabpanel\"><div class=\"card\"><div class=\"card-body\">{table_for(int(row.CategoryID))}</div></div></div>")

    html = f"""
<!DOCTYPE html><html lang=\"en\"><head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Sakila - Customers by Category</title>
  <link href=\"https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css\" rel=\"stylesheet\" />
  <style>body{{background:#121212;color:#e0e0e0}}.card{{background:#1e1e1e}}</style>
</head><body class=\"p-3\"><div class=\"container\">
  <h3 class=\"mb-3\">Sakila – Customers by Category</h3>
  <ul class=\"nav nav-pills mb-3\" id=\"catTabs\" role=\"tablist\">{''.join(tabs)}</ul>
  <div class=\"tab-content\">{''.join(panes)}</div>
</div>
<script src=\"https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js\"></script>
</body></html>
"""
    out_path.write_text(html, encoding="utf-8")


def write_clusters_html(df: pd.DataFrame, out_path: Path):
    clusters = df.groupby("Cluster").size().reset_index(name="Count").sort_values("Cluster")
    rows_by_cluster: dict[int, list[dict]] = {}
    for _, r in df.iterrows():
        rows_by_cluster.setdefault(int(r["Cluster"]), []).append({
            "CustomerID": int(r["CustomerID"]),
            "Name": r["Name"],
            "Rentals": int(r["Rentals"]),
            "DistinctFilms": int(r["DistinctFilms"]),
            "DistinctCategories": int(r["DistinctCategories"]),
        })

    def table_for(cid: int) -> str:
        rows = rows_by_cluster.get(cid, [])
        body = "".join([f"<tr><td>{r['CustomerID']}</td><td>{r['Name']}</td><td>{r['Rentals']}</td><td>{r['DistinctFilms']}</td><td>{r['DistinctCategories']}</td></tr>" for r in rows])
        return f"<table class=\"table table-dark table-striped table-hover\"><thead><tr><th>CustomerID</th><th>Name</th><th>Rentals</th><th>DistinctFilms</th><th>DistinctCategories</th></tr></thead><tbody>{body}</tbody></table>"

    tabs = []
    panes = []
    for i, row in enumerate(clusters.itertuples(index=False)):
        active = "active" if i == 0 else ""
        show = "show active" if i == 0 else ""
        tabs.append(f"<li class=\"nav-item\" role=\"presentation\"><button class=\"nav-link {active}\" data-bs-toggle=\"tab\" data-bs-target=\"#cluster-{int(row.Cluster)}\" type=\"button\" role=\"tab\">Cluster {int(row.Cluster)} ({int(row.Count)})</button></li>")
        panes.append(f"<div class=\"tab-pane fade {show}\" id=\"cluster-{int(row.Cluster)}\" role=\"tabpanel\"><div class=\"card\"><div class=\"card-body\">{table_for(int(row.Cluster))}</div></div></div>")

    html = f"""
<!DOCTYPE html><html lang=\"en\"><head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Sakila - Customers by Interest Clusters</title>
  <link href=\"https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css\" rel=\"stylesheet\" />
  <style>body{{background:#121212;color:#e0e0e0}}.card{{background:#1e1e1e}}</style>
</head><body class=\"p-3\"><div class=\"container\">
  <h3 class=\"mb-3\">Sakila – Customers by Interest Clusters</h3>
  <ul class=\"nav nav-pills mb-3\" id=\"clusterTabs\" role=\"tablist\">{''.join(tabs)}</ul>
  <div class=\"tab-content\">{''.join(panes)}</div>
</div>
<script src=\"https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js\"></script>
</body></html>
"""
    out_path.write_text(html, encoding="utf-8")


def export_excel(df: pd.DataFrame, path: Path, sheet_name: str):
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)


def cluster_customers(features_df: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    X = features_df[["Rentals", "DistinctFilms", "DistinctCategories"]].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    out = features_df.copy()
    out["Cluster"] = labels
    return out


def open_on_server(relative_filename: str):
    base = "http://127.0.0.1:8001/"  # use 8001 preview server
    url = base + relative_filename.replace("\\", "/")
    try:
        webbrowser.open(url, new=2)
    except Exception:
        webbrowser.open(str(Path(__file__).parent / relative_filename), new=2)


def main():
    conn = Connector(database="sakila")
    if conn.connect() is None:
        print("[ERROR] Không thể kết nối tới MySQL 'sakila'. Kiểm tra connectors/connector.py")
        return

    tests_dir = Path(__file__).parent
    film_html = tests_dir / "sakila_customers_by_film.html"
    cat_html = tests_dir / "sakila_customers_by_category.html"
    cluster_html = tests_dir / "sakila_customers_by_interest_clusters.html"

    print("Fetching customers by film ...")
    df_film = fetch_customers_by_film(conn)
    if df_film is None:
        print("[ERROR] Truy vấn khách theo phim thất bại.")
        return
    print_grouped(df_film, "FilmTitle", ["CustomerID", "Name", "Email", "Active"])
    print(f"\nWriting HTML: {film_html}")
    write_customers_by_film_html(df_film, film_html)
    export_excel(df_film, tests_dir / "sakila_customers_by_film.xlsx", "CustomersByFilm")

    print("\nFetching customers by category ...")
    df_cat = fetch_customers_by_category(conn)
    if df_cat is None:
        print("[ERROR] Truy vấn khách theo category thất bại.")
        return
    print_grouped(df_cat, "Category", ["CustomerID", "Name", "Email", "Active"])
    print(f"\nWriting HTML: {cat_html}")
    write_customers_by_category_html(df_cat, cat_html)
    export_excel(df_cat, tests_dir / "sakila_customers_by_category.xlsx", "CustomersByCategory")

    print("\nComputing interest features and clustering ...")
    features = fetch_interest_features(conn)
    if features is None:
        print("[ERROR] Truy vấn đặc trưng quan tâm thất bại.")
        return
    clustered = cluster_customers(features, k=4)
    print("Cluster sizes:")
    print(clustered.groupby("Cluster").size().to_string())
    print(f"\nWriting HTML: {cluster_html}")
    write_clusters_html(clustered, cluster_html)
    export_excel(clustered, tests_dir / "sakila_customers_clusters.xlsx", "Clusters")

    open_on_server("sakila_customers_by_category.html")


if __name__ == "__main__":
    main()