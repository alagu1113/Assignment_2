# amazon_full_dashboard_final.py
# Full multi-tab Streamlit app for Q1 - Q30
# Update: uses your provided CSV paths.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import plotly.graph_objects as go
import os

# Optional sklearn for linear forecasting (if installed)
try:
    from sklearn.linear_model import LinearRegression
    SKLEARN = True
except Exception:
    SKLEARN = False

st.set_page_config(page_title="Amazon — Full BI Suite (Q1–Q30)", layout="wide")

# -------------------------
# File paths (user-provided)
# -------------------------
CUSTOMER_FILE = r"C:\Users\Alagu\DATASCIENCE\Cleaned_data\amazon_india_complete_2015_2025_with_base_price.csv"
CUSTOMER_FILE_OLD = r"C:\Users\Alagu\DATASCIENCE\Cleaned_data\amazon_india_complete_2015_2025_cleaned_03.csv"
PRODUCT_FILE = r"C:\Users\Alagu\DATASCIENCE\amazon_india_products_catalog.csv"

# -------------------------
# Load data safely
# -------------------------
@st.cache_data(ttl=600)
def load_csv(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

cust = load_csv(CUSTOMER_FILE)
if cust is None:
    st.error(f"Customer file not found: {CUSTOMER_FILE}")
    st.stop()

prod = load_csv(PRODUCT_FILE)  # product catalog optional
# also load old customer as backup
cust_old = load_csv(CUSTOMER_FILE_OLD)

# -------------------------
# Basic preprocessing & heuristics
# -------------------------
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# canonicalize columns in customer file
COL_ORDER_DATE = find_col(cust, ['order_date','OrderDate','order_datetime','date'])
COL_FINAL = find_col(cust, ['final_amount_inr','final_amount','amount','order_value'])
COL_CUSTOMER = find_col(cust, ['customer_id','customer','cust_id'])
COL_PRODUCT = find_col(cust, ['product_id','sku','product_id_str','product'])
COL_CATEGORY = find_col(cust, ['subcategory','product_category','product_cat'])
COL_BRAND = find_col(cust, ['brand','Brand'])
COL_STATE = find_col(cust, ['customer_state','state','customer_state_name'])
COL_CITY = find_col(cust, ['customer_city','city'])
COL_FESTIVAL = find_col(cust, ['festival_name','festival'])
COL_QUANTITY = find_col(cust, ['quantity','qty','units_sold'])
COL_BASE_PRICE = find_col(cust, ['base_price','base_price_2015'])
COL_FINAL = COL_FINAL or find_col(cust, ['final_amount_inr','final_amount','amount'])

# ensure numeric
if COL_FINAL:
    cust[COL_FINAL] = pd.to_numeric(cust[COL_FINAL], errors='coerce')
if COL_BASE_PRICE:
    cust[COL_BASE_PRICE] = pd.to_numeric(cust[COL_BASE_PRICE], errors='coerce')
if COL_QUANTITY:
    cust[COL_QUANTITY] = pd.to_numeric(cust[COL_QUANTITY], errors='coerce')

# date conversions
if COL_ORDER_DATE:
    cust[COL_ORDER_DATE] = pd.to_datetime(cust[COL_ORDER_DATE], errors='coerce')
    cust['order_month'] = cust[COL_ORDER_DATE].dt.to_period('M').dt.to_timestamp()
    cust['order_quarter'] = cust[COL_ORDER_DATE].dt.to_period('Q').dt.to_timestamp()
    cust['order_year'] = cust[COL_ORDER_DATE].dt.year
else:
    cust['order_month'] = pd.NaT
    cust['order_quarter'] = pd.NaT
    cust['order_year'] = np.nan

# prepare product mapping if product file exists
if prod is not None:
    prod_cols = prod.columns.tolist()
    prod_id_col = find_col(prod, ['product_id','sku','product'])
    prod_price_col = find_col(prod, ['base_price','base_price_inr','mrp','price'])
else:
    prod_id_col = prod_price_col = None

# -------------------------
# Formatting helpers
# -------------------------
def fmt_curr(x):
    try:
        if pd.isna(x):
            return ""
        return f"{float(x):,.0f}"
    except Exception:
        return str(x)

def monthly_revenue(df):
    if 'order_month' not in df.columns:
        return pd.DataFrame(columns=['order_month','revenue'])
    return df.groupby('order_month')[COL_FINAL].sum().reset_index().rename(columns={COL_FINAL:'revenue'}).sort_values('order_month')

def yearly_revenue(df):
    if 'order_year' not in df.columns:
        return pd.DataFrame(columns=['order_year','revenue'])
    return df.groupby('order_year')[COL_FINAL].sum().reset_index().rename(columns={COL_FINAL:'revenue'}).sort_values('order_year')

def linear_forecast(series, periods=6):
    s = series.dropna()
    if len(s) < 2 or not SKLEARN:
        # fallback to last value repeated or moving average
        if len(s) == 0:
            return np.array([np.nan]*periods)
        last = float(s.iloc[-1])
        return np.array([last]*periods)
    X = np.arange(len(s)).reshape(-1,1)
    y = s.values
    model = LinearRegression().fit(X,y)
    future = np.arange(len(s), len(s)+periods).reshape(-1,1)
    return model.predict(future)

# -------------------------
# Sidebar global controls
# -------------------------
st.sidebar.header("Global Controls")
lookback_months = st.sidebar.slider("Lookback months", 6, 120, 50)
top_n = st.sidebar.slider("Top N (categories/brands/products)", 5, 100, 50)
forecast_horizon = st.sidebar.slider("Forecast horizon (periods)", 1, 12, 6)

# -------------------------
# Tabs Q1 - Q30
# -------------------------
tab_labels = [
    "Q1 Exec Summary",
    "Q2 Real-time",
    "Q3 Strategic",
    "Q4 Financial",
    "Q5 Growth",
    "Q6 Revenue Trend",
    "Q7 Category",
    "Q8 Geographic",
    "Q9 Festival",
    "Q10 PriceOpt",
    "Q11 RFM",
    "Q12 CustomerJourney",
    "Q13 Prime",
    "Q14 Retention",
    "Q15 Demographics",
    "Q16 ProductPerf",
    "Q17 Brand",
    "Q18 — Inventory Optimization",
    "Q19 — Rating & Review",
    "Q20 NewLaunch",
    "Q21 Delivery",
    "Q22 Payment",
    "Q23 Returns",
    "Q24 CS"
]
tabs = st.tabs(tab_labels)

today = datetime.now()

# -------------------------
# Q1 Executive Summary
# -------------------------
with tabs[0]:
    st.header("Q1 — Executive Summary")
    total_rev = cust[COL_FINAL].sum() if COL_FINAL else 0
    active_cust = cust[COL_CUSTOMER].nunique() if COL_CUSTOMER else np.nan
    aov = cust[COL_FINAL].mean() if COL_FINAL else np.nan
    yrev = yearly_revenue(cust)
    yoy = yrev['revenue'].pct_change().iloc[-1]*100 if len(yrev)>=2 else np.nan

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Revenue (INR)", fmt_curr(total_rev))
    c2.metric("YoY Growth (%)", f"{yoy:.2f}%" if not pd.isna(yoy) else "N/A")
    c3.metric("Active Customers", f"{int(active_cust):,}" if not pd.isna(active_cust) else "N/A")
    c4.metric("Avg Order Value (INR)", fmt_curr(aov))

    st.subheader("Top Performing Categories")
    if COL_CATEGORY:
        top_cats = cust.groupby(COL_CATEGORY)[COL_FINAL].sum().reset_index().sort_values(COL_FINAL, ascending=False).head(top_n)
        st.plotly_chart(px.bar(top_cats, x=COL_CATEGORY, y=COL_FINAL, title="Top Categories by Revenue"), use_container_width=True)
    else:
        st.info("Category column not found.")

# -------------------------
# Q2 Real-time Business Performance Monitor
# -------------------------
with tabs[1]:
    st.header("Q2 — Real-time Business Performance Monitor")
    cur_month = pd.to_datetime(datetime(today.year, today.month, 1))
    mr = monthly_revenue(cust)
    cur_rev = float(mr.loc[mr['order_month']==cur_month,'revenue']) if (not mr.empty and cur_month in list(mr['order_month'])) else 0.0
    target_rev = st.number_input("Target Revenue (current month) INR", value=5_000_000, step=1000)
    target_new_cust = st.number_input("Target New Customers (current month)", value=1000, step=1)
    new_cust = int(cust.loc[cust['order_month']==cur_month, COL_CUSTOMER].nunique()) if COL_CUSTOMER and 'order_month' in cust.columns else 0
    run_rate = cur_rev*12
    c1,c2,c3 = st.columns(3)
    c1.metric(f"This Month ({cur_month.strftime('%b %Y')})", fmt_curr(cur_rev), delta=f"{cur_rev - target_rev:,.0f} vs target")
    c2.metric("Revenue Run-rate (annual)", fmt_curr(run_rate))
    c3.metric("New Customers (this month)", f"{new_cust:,}", delta=f"{new_cust - target_new_cust} vs target")
    if cur_rev < target_rev:
        st.error("Alert: Revenue below target")
    else:
        st.success("Revenue on track")
    if not mr.empty:
        st.plotly_chart(px.bar(mr.tail(lookback_months), x='order_month', y='revenue', title="Recent Monthly Revenue"), use_container_width=True)

# -------------------------
# Q3 Strategic Overview
# -------------------------
with tabs[2]:
    st.header("Q3 — Strategic Overview")
    basis = COL_CATEGORY or COL_BRAND
    if basis:
        share = cust.groupby(basis)[COL_FINAL].sum().reset_index()
        share['pct'] = share[COL_FINAL]/share[COL_FINAL].sum()*100
        st.plotly_chart(px.pie(share.sort_values(COL_FINAL, ascending=False).head(top_n), names=basis, values='pct', title=f"Market Share ({basis})"), use_container_width=True)
    else:
        st.info("category/brand missing")
    if COL_STATE:
        geo = cust.groupby(COL_STATE)[COL_FINAL].sum().reset_index().sort_values(COL_FINAL, ascending=False).head(20)
        st.plotly_chart(px.bar(geo, x=COL_STATE, y=COL_FINAL, title="Top states by revenue"), use_container_width=True)

# -------------------------
# Q4 Financial Performance
# -------------------------
with tabs[3]:
    st.header("Q4 — Financial Performance")
    if COL_CATEGORY:
        rev_by_cat = cust.groupby(COL_CATEGORY)[COL_FINAL].sum().reset_index()
        st.plotly_chart(px.treemap(rev_by_cat, path=[COL_CATEGORY], values=COL_FINAL, title="Revenue by Category"), use_container_width=True)
    if COL_BASE_PRICE and COL_PRODUCT:
        st.subheader("Base Price vs Avg Final Amount")
        avg_by_prod = cust.groupby(COL_PRODUCT)[COL_FINAL].mean().reset_index().rename(columns={COL_FINAL:'avg_final_amount'})
        if COL_BASE_PRICE in cust.columns:
            df_compare = avg_by_prod.merge(cust[[COL_PRODUCT, COL_BASE_PRICE]].drop_duplicates(subset=[COL_PRODUCT]), on=COL_PRODUCT, how='left')
            st.dataframe(df_compare.head(50).style.format({"avg_final_amount":"{:,.0f}", COL_BASE_PRICE:"{:,.0f}"}))

# -------------------------
# Q5 Growth Analytics
# -------------------------
with tabs[4]:
    st.header("Q5 — Growth Analytics")
    if 'order_month' in cust.columns and COL_CUSTOMER:
        cust_growth = cust.groupby('order_month')[COL_CUSTOMER].nunique().reset_index().rename(columns={COL_CUSTOMER:'unique_customers'})
        st.plotly_chart(px.line(cust_growth.tail(lookback_months), x='order_month', y='unique_customers', title="Unique Customers Over Time"), use_container_width=True)
    if COL_PRODUCT:
        sku_by_year = cust.groupby('order_year')[COL_PRODUCT].nunique().reset_index().rename(columns={COL_PRODUCT:'unique_skus'})
        st.plotly_chart(px.bar(sku_by_year, x='order_year', y='unique_skus', title="Unique SKUs by Year"), use_container_width=True)

# -------------------------
# Q6 Revenue Trend
# -------------------------
with tabs[5]:
    st.header("Q6 — Revenue Trend Analysis")
    freq = st.selectbox("Frequency", ["Monthly","Quarterly","Yearly"], index=0)
    if freq == "Monthly":
        rev = monthly_revenue(cust).rename(columns={'revenue':'revenue'})
        xcol = 'order_month'
    elif freq == "Quarterly":
        rev = cust.groupby('order_quarter')[COL_FINAL].sum().reset_index().rename(columns={COL_FINAL:'revenue'})
        xcol = 'order_quarter'
    else:
        rev = yearly_revenue(cust).rename(columns={'revenue':'revenue'})
        xcol = 'order_year'
    if not rev.empty:
        rev['growth_pct'] = rev['revenue'].pct_change()*100
        st.plotly_chart(px.line(rev, x=xcol, y='revenue', markers=True, title=f"{freq} Revenue Trend"), use_container_width=True)
        st.dataframe(rev.tail(60).style.format({"revenue":"{:,.0f}", "growth_pct":"{:.2f}%"}))
        preds = linear_forecast(rev['revenue'], periods=forecast_horizon)
        future_df = pd.DataFrame({f'forecast_{i+1}': [p] for i,p in enumerate(preds)}).T.reset_index()
        future_df.columns = ['period','forecast_revenue']
        st.subheader("Forecast (next periods)")
        st.dataframe(future_df.style.format({"forecast_revenue":"{:,.0f}"}))
    else:
        st.info("Insufficient revenue series")

# -------------------------
# Q7 Category Performance
# -------------------------
with tabs[6]:
    st.header("Q7 — Category Performance")
    if COL_CATEGORY:
        categories = sorted(cust[COL_CATEGORY].dropna().unique().tolist())
        sel = st.selectbox("Select category or All", ["All"] + categories)
        if sel=="All":
            cat_df = cust.groupby(COL_CATEGORY)[COL_FINAL].sum().reset_index().sort_values(COL_FINAL, ascending=False).head(top_n)
            st.plotly_chart(px.bar(cat_df, x=COL_CATEGORY, y=COL_FINAL, title="Top categories"), use_container_width=True)
            st.dataframe(cat_df.style.format({COL_FINAL:"{:,.0f}"}))
        else:
            sub = cust[cust[COL_CATEGORY]==sel]
            st.metric("Revenue", fmt_curr(sub[COL_FINAL].sum()))
            if 'order_month' in sub.columns:
                tr = sub.groupby('order_month')[COL_FINAL].sum().reset_index()
                st.plotly_chart(px.line(tr.tail(lookback_months), x='order_month', y=COL_FINAL, title=f"{sel} revenue trend"), use_container_width=True)
    else:
        st.info("Category column missing")

# -------------------------
# Q8 Geographic Revenue
# -------------------------
with tabs[7]:
    st.header("Q8 — Geographic Revenue Analysis")
    level = st.selectbox("Level", ["State","City"])
    geo_col = COL_STATE if level=="State" else COL_CITY
    if geo_col:
        gdf = cust.groupby(geo_col)[COL_FINAL].sum().reset_index().sort_values(COL_FINAL, ascending=False).head(20)
        st.plotly_chart(px.bar(gdf, x=geo_col, y=COL_FINAL, title=f"Top {level}s by Revenue"), use_container_width=True)
    else:
        st.info("No geographic columns found")

# -------------------------
# Q9 Festival Sales
# -------------------------
with tabs[8]:
    st.header("Q9 — Festival Sales Analytics")
    if COL_FESTIVAL:
        festivals = sorted(cust[COL_FESTIVAL].dropna().unique().tolist())
        sel_f = st.selectbox("Select festival", festivals)
        fest_df = cust[cust[COL_FESTIVAL]==sel_f].groupby('order_year')[COL_FINAL].sum().reset_index()
        st.plotly_chart(px.bar(fest_df, x='order_year', y=COL_FINAL, title=f"{sel_f} Revenue by Year"), use_container_width=True)
    else:
        st.info("No festival data available")

# -------------------------
# Q10 Price Optimization
# -------------------------
with tabs[9]:
    st.header("Q10 — Price Optimization")
    if COL_BASE_PRICE and COL_FINAL:
        # compare base_price vs avg final amount by product
        avg_by_prod = cust.groupby(COL_PRODUCT)[COL_FINAL].mean().reset_index().rename(columns={COL_FINAL:'avg_final_amount'})
        if COL_BASE_PRICE in cust.columns:
            compare = avg_by_prod.merge(cust[[COL_PRODUCT, COL_BASE_PRICE]].drop_duplicates(subset=[COL_PRODUCT]), on=COL_PRODUCT, how='left')
            st.plotly_chart(px.scatter(compare, x=COL_BASE_PRICE, y='avg_final_amount', hover_data=[COL_PRODUCT], title="Base Price vs Avg Final Amount"), use_container_width=True)
            st.dataframe(compare.head(50).style.format({"avg_final_amount":"{:,.0f}", COL_BASE_PRICE:"{:,.0f}"}))
        else:
            st.info("No base_price column present in customer file")
    else:
        st.info("Price/final amount data not sufficient for price optimization")

# -------------------------
# Q11–Q15 Customer Analytics (scaffold)
# -------------------------
with tabs[10]:
    st.header("Q11 — Customer Segmentation (RFM Analysis)")

    if COL_CUSTOMER and COL_ORDER_DATE and COL_FINAL:
        # Snapshot Date
        snapshot = cust[COL_ORDER_DATE].max()

        # Compute RFM components
        recency = cust.groupby(COL_CUSTOMER)[COL_ORDER_DATE].max().reset_index().rename(columns={COL_ORDER_DATE: 'last_purchase'})
        recency['recency_days'] = (snapshot - recency['last_purchase']).dt.days

        freq = cust.groupby(COL_CUSTOMER).size().reset_index(name='frequency')
        monet = cust.groupby(COL_CUSTOMER)[COL_FINAL].sum().reset_index(name='monetary')

        # Merge into RFM table
        rfm = recency.merge(freq, on=COL_CUSTOMER).merge(monet, on=COL_CUSTOMER)

        st.subheader("RFM Summary (Sample 200 Customers)")
        st.dataframe(
            rfm.head(200).style.format({
                "recency_days": "{:.0f}",
                "frequency": "{:.0f}",
                "monetary": "{:,.0f}"
            })
        )

        # -------------------------------
        # 🔹 RFM Distribution Visualizations
        # -------------------------------

        st.subheader("RFM Distributions")

        col1, col2, col3 = st.columns(3)

        # Recency distribution
        with col1:
            fig_r = px.histogram(
                rfm, x="recency_days",
                nbins=30, color_discrete_sequence=["#1f77b4"],
                title="Recency (Days Since Last Purchase)",
                labels={"recency_days": "Days"}
            )
            st.plotly_chart(fig_r, use_container_width=True)

        # Frequency distribution
        with col2:
            fig_f = px.histogram(
                rfm, x="frequency",
                nbins=30, color_discrete_sequence=["#ff7f0e"],
                title="Frequency (Number of Orders)",
                labels={"frequency": "Orders"}
            )
            st.plotly_chart(fig_f, use_container_width=True)

        # Monetary distribution
        with col3:
            fig_m = px.histogram(
                rfm, x="monetary",
                nbins=30, color_discrete_sequence=["#2ca02c"],
                title="Monetary Value (Total Spend)",
                labels={"monetary": "Amount (INR)"}
            )
            st.plotly_chart(fig_m, use_container_width=True)

        # -------------------------------
        # 🔹 2D Scatter: Frequency vs Monetary
        # -------------------------------
        st.subheader("Customer Segmentation — Frequency vs Monetary")
        fig_scatter = px.scatter(
            rfm,
            x="frequency",
            y="monetary",
            color="recency_days",
            color_continuous_scale="Viridis",
            size="monetary",
            hover_data=[COL_CUSTOMER],
            title="Customer Distribution by Frequency, Monetary, and Recency"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # -------------------------------
        # 🔹 Optional RFM Segmentation (Quantile-based)
        # -------------------------------
        st.subheader("RFM Segmentation (Quantile-Based)")

        # Quantile segmentation
        rfm['R_Score'] = pd.qcut(rfm['recency_days'], 4, labels=[4,3,2,1])
        rfm['F_Score'] = pd.qcut(rfm['frequency'].rank(method="first"), 4, labels=[1,2,3,4])
        rfm['M_Score'] = pd.qcut(rfm['monetary'], 4, labels=[1,2,3,4])
        rfm['RFM_Segment'] = rfm[['R_Score','F_Score','M_Score']].astype(str).agg(''.join, axis=1)
        rfm['RFM_Score'] = rfm[['R_Score','F_Score','M_Score']].astype(int).sum(axis=1)

        # Segment visualization
        fig_seg = px.histogram(
            rfm, x="RFM_Score",
            nbins=12,
            color_discrete_sequence=["#9467bd"],
            title="Distribution of RFM Scores",
            labels={"RFM_Score": "Total RFM Score"}
        )
        st.plotly_chart(fig_seg, use_container_width=True)

    else:
        st.info("Need customer, order_date, and final_amount_inr columns for RFM analysis.")

with tabs[11]:
    st.header("Q12 — Customer Journey Analytics (Category Transitions)")

    if COL_CUSTOMER and COL_ORDER_DATE and COL_CATEGORY:
        # Ensure datetime
        cust[COL_ORDER_DATE] = pd.to_datetime(cust[COL_ORDER_DATE], errors='coerce')

        # Sort by customer and date
        cust_sorted = cust.sort_values(by=[COL_CUSTOMER, COL_ORDER_DATE])

        # Get previous and next category to track transitions
        cust_sorted['next_category'] = cust_sorted.groupby(COL_CUSTOMER)[COL_CATEGORY].shift(-1)

        # Drop nulls
        trans = cust_sorted.dropna(subset=[COL_CATEGORY, 'next_category'])

        # Count transitions
        trans_count = (
            trans.groupby([COL_CATEGORY, 'next_category'])
            .size()
            .reset_index(name='count')
            .sort_values('count', ascending=False)
        )

        # Show top transitions
        st.subheader("Top Customer Category Transitions")
        st.dataframe(trans_count.head(20))

        # Sankey diagram for flow visualization
        st.subheader("Customer Journey Flow (Category Transitions)")


        # Build Sankey nodes
        labels = list(pd.unique(trans_count[[COL_CATEGORY, 'next_category']].values.ravel('K')))
        label_to_index = {label: i for i, label in enumerate(labels)}

        # Build links
        source = trans_count[COL_CATEGORY].map(label_to_index)
        target = trans_count['next_category'].map(label_to_index)
        value = trans_count['count']

        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color="teal"
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color="rgba(0,150,136,0.3)"
            )
        )])

        fig.update_layout(
            title_text="Customer Journey Across Categories",
            font_size=12,
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Need customer, order_date, and category columns for journey analysis.")

with tabs[12]:
    st.header("Q13 — Prime Membership Analytics")

    prime_col = find_col(cust, ['is_prime_member', 'is_prime', 'prime'])
    if prime_col:
        # --- Summary statistics ---
        summary = cust.groupby(prime_col)[COL_FINAL].agg(['count', 'sum', 'mean']).reset_index()
        summary.columns = [prime_col, 'Order Count', 'Total Revenue', 'Average Revenue']

        # Display table
        st.dataframe(summary.style.format({
            "Total Revenue": "{:,.0f}",
            "Average Revenue": "{:,.0f}"
        }))

        # --- Visualization Section ---
        st.subheader("Prime vs Non-Prime Performance")

        # 1️⃣ Bar chart — Total Revenue comparison
        fig1 = px.bar(
            summary,
            x=prime_col,
            y='Total Revenue',
            title="Total Revenue by Prime Membership",
            color=prime_col,
            text='Total Revenue'
        )
        fig1.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        fig1.update_layout(yaxis_title="Revenue (INR)", xaxis_title="Prime Membership")
        st.plotly_chart(fig1, use_container_width=True)

        # 2️⃣ Pie chart — Order distribution
        st.subheader("Order Distribution by Prime Membership")
        fig2 = px.pie(
            summary,
            names=prime_col,
            values='Order Count',
            title="Order Count Share (Prime vs Non-Prime)",
            hole=0.4
        )
        st.plotly_chart(fig2, use_container_width=True)

        # 3️⃣ Bar chart — Average Revenue comparison
        st.subheader("Average Revenue per Order")
        fig3 = px.bar(
            summary,
            x=prime_col,
            y='Average Revenue',
            color=prime_col,
            text='Average Revenue',
            title="Average Revenue by Prime Membership"
        )
        fig3.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        st.plotly_chart(fig3, use_container_width=True)

    else:
        st.info("No prime membership column found")

# Q14 — Customer Retention (Cohorts)
# ----------------------------------
with tabs[13]:
    st.header("Q14 — Customer Retention Dashboard (Cohort Analysis)")

    try:
        # Use the main customer dataframe
        df = cust.copy()

        # Identify columns
        COL_CUSTOMER = next((c for c in df.columns if 'customer_id' in c.lower()), None)
        COL_DATE = next((c for c in df.columns if 'order_date' in c.lower()), None)

        if COL_CUSTOMER and COL_DATE:
            df[COL_DATE] = pd.to_datetime(df[COL_DATE])

            # Create monthly order periods
            df['order_month'] = df[COL_DATE].dt.to_period('M')
            first_purchase = df.groupby(COL_CUSTOMER)['order_month'].min().reset_index()
            first_purchase.columns = [COL_CUSTOMER, 'cohort_month']

            df = df.merge(first_purchase, on=COL_CUSTOMER)

            # Calculate period number (months since first purchase)
            df['period_number'] = (df['order_month'].dt.year - df['cohort_month'].dt.year) * 12 + \
                                  (df['order_month'].dt.month - df['cohort_month'].dt.month)

            # Convert periods to strings for JSON serialization
            df['order_month'] = df['order_month'].astype(str)
            df['cohort_month'] = df['cohort_month'].astype(str)

            # Cohort aggregation
            cohort_data = df.groupby(['cohort_month', 'period_number'])[COL_CUSTOMER].nunique().reset_index()
            cohort_pivot = cohort_data.pivot(index='cohort_month', columns='period_number', values=COL_CUSTOMER)

            # Retention calculation
            cohort_size = cohort_pivot.iloc[:, 0]
            retention = cohort_pivot.divide(cohort_size, axis=0).round(3) * 100

            # 🔥 Heatmap
            st.subheader("Customer Retention Heatmap (%)")
            fig = px.imshow(
                retention,
                text_auto=True,
                color_continuous_scale="Tealgrn",
                aspect="auto",
                labels=dict(x="Months Since First Purchase", y="Cohort (Join Month)", color="Retention %"),
                title="Cohort Retention Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)

            # 📈 Average Retention Trend
            avg_retention = retention.mean(axis=1).round(1).reset_index()
            avg_retention.columns = ['Cohort', 'Avg Retention %']

            st.subheader("Average Retention by Cohort")
            fig_line = px.line(
                avg_retention,
                x='Cohort',
                y='Avg Retention %',
                markers=True,
                title="Average Retention Trend by Customer Cohort"
            )
            st.plotly_chart(fig_line, use_container_width=True)

        else:
            st.warning("Required columns (customer_id, order_date) not found in dataset.")

    except Exception as e:
        st.error(f"Error generating cohort analysis: {e}")

#########################
with tabs[14]:
    st.header("Q15 — Demographics & Behavior")
    demo_cols = [c for c in ['customer_age_group','age','gender','customer_tier'] if c in cust.columns]
    if demo_cols:
        st.write("Available demographics:", demo_cols)
        for c in demo_cols:
            st.plotly_chart(px.histogram(cust, x=c, title=f"Distribution: {c}"), use_container_width=True)
    else:
        st.info("No demographic columns detected")

# -------------------------
# Q16–Q20 Product & Inventory (scaffold)
# -------------------------
with tabs[15]:
    st.header("Q16 — Product Performance")
    if COL_PRODUCT:
        prod_perf = cust.groupby(COL_PRODUCT).agg({COL_FINAL:'sum', COL_QUANTITY:'sum'}).reset_index().sort_values(COL_FINAL, ascending=False).head(top_n)
        st.dataframe(prod_perf.style.format({COL_FINAL:"{:,.0f}", COL_QUANTITY:"{:,.0f}"}))
    else:
        st.info("No product id column")

with tabs[16]:
    st.header("Q17 — Brand Analytics")
    if COL_BRAND:
        brand = cust.groupby(COL_BRAND)[COL_FINAL].sum().reset_index().sort_values(COL_FINAL, ascending=False).head(top_n)
        st.plotly_chart(px.bar(brand, x=COL_BRAND, y=COL_FINAL, title="Top Brands"), use_container_width=True)
    else:
        st.info("No brand column")

with tabs[17]:
    st.header("Q18 — Inventory Optimization")
    st.info("Requires inventory / stock / lead-time columns to compute. Placeholder shown.")

with tabs[18]:
    st.header("Q19 — Rating & Review")
    if 'customer_rating' in cust.columns:
        st.plotly_chart(px.histogram(cust, x='customer_rating', nbins=10, title="Rating distribution"), use_container_width=True)
    else:
        st.info("No rating column")

# Q20 — New Product Launch
# -------------------------
with tabs[19]:
    st.header("Q20 — New Product Launch")

    if prod is None or prod.empty:
        st.error("Product file not loaded. Please check PRODUCT_FILE path.")
    else:
        # Identify columns
        COL_PRODUCT = next((c for c in cust.columns if 'product_id' in c.lower()), None)
        COL_CUSTOMER = next((c for c in cust.columns if 'customer_id' in c.lower()), None)
        COL_LAUNCH_YEAR = next((c for c in prod.columns if 'launch' in c.lower() and 'year' in c.lower()), None)
        COL_BRAND = next((c for c in prod.columns if 'brand' in c.lower()), None)

        # Fallback: if launch year not present, derive from first appearance in customer data
        if not COL_LAUNCH_YEAR and COL_PRODUCT and COL_ORDER_DATE in cust.columns:
            st.warning("No 'launch_year' column found in product catalog — deriving launch year from first order date.")
            first_appearance = cust.groupby(COL_PRODUCT)[COL_ORDER_DATE].min().dt.year.reset_index()
            first_appearance.columns = [COL_PRODUCT, 'launch_year']
            prod = prod.merge(first_appearance, on=COL_PRODUCT, how='left')
            COL_LAUNCH_YEAR = 'launch_year'

        if COL_PRODUCT and COL_CUSTOMER and COL_LAUNCH_YEAR:
            merged = (
                cust[[COL_PRODUCT, COL_CUSTOMER]].drop_duplicates()
                .merge(prod[[COL_PRODUCT, COL_LAUNCH_YEAR, COL_BRAND]] if COL_BRAND else prod[[COL_PRODUCT, COL_LAUNCH_YEAR]],
                       on=COL_PRODUCT, how='left')
            )
            merged = merged.dropna(subset=[COL_LAUNCH_YEAR])

            # Compute launch stats
            launch_stats = merged.groupby(COL_LAUNCH_YEAR)[COL_CUSTOMER].nunique().reset_index()
            launch_stats.rename(columns={COL_CUSTOMER: 'unique_customers'}, inplace=True)
            launch_stats = launch_stats.sort_values(COL_LAUNCH_YEAR)

            # Metrics
            total_launches = prod[COL_PRODUCT].nunique()
            total_customers = merged[COL_CUSTOMER].nunique()
            latest_year = int(launch_stats[COL_LAUNCH_YEAR].max())

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Products Launched", f"{total_launches:,}")
            c2.metric("Total Customers Purchased", f"{total_customers:,}")
            c3.metric("Latest Launch Year", latest_year)

            # Chart — Customer Reach by Launch Year
            fig1 = px.bar(
                launch_stats,
                x=COL_LAUNCH_YEAR,
                y='unique_customers',
                text='unique_customers',
                title="Customer Reach by Product Launch Year",
                labels={COL_LAUNCH_YEAR: "Launch Year", 'unique_customers': "Unique Customers"},
                color='unique_customers',
                color_continuous_scale='Blues'
            )
            fig1.update_traces(textposition='outside')
            st.plotly_chart(fig1, use_container_width=True)

            # Top 10 products for a selected launch year
            sel_year = st.selectbox("Select Launch Year", ["All"] + launch_stats[COL_LAUNCH_YEAR].astype(str).tolist())

            if sel_year != "All":
                year_data = merged[merged[COL_LAUNCH_YEAR] == int(sel_year)]
                top_products = (
                    year_data.groupby([COL_PRODUCT, COL_BRAND])[COL_CUSTOMER]
                    .count().reset_index().rename(columns={COL_CUSTOMER: 'purchase_count'})
                    .sort_values('purchase_count', ascending=False).head(10)
                )
                top_products["label"] = top_products.apply(
                    lambda x: f"{x[COL_BRAND]} — {x[COL_PRODUCT]}" if pd.notna(x[COL_BRAND]) else x[COL_PRODUCT],
                    axis=1
                )

                fig2 = px.bar(
                    top_products, x='label', y='purchase_count',
                    text='purchase_count', title=f"Top 10 Products Launched in {sel_year}",
                    color='purchase_count', color_continuous_scale='Tealgrn'
                )
                fig2.update_traces(textposition='outside')
                fig2.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.error("Missing required columns — check product_id, customer_id, launch_year, or brand.")

# -------------------------
# Q21–Q25 Operations & Logistics (scaffold)
# -------------------------
# -------------------------
# Q21 — Delivery Performance
# -------------------------
with tabs[20]:
    st.header("Q21 — Delivery Performance")

    # Check required columns
    if 'delivery_days' not in cust.columns:
        st.info("No 'delivery_days' column found in customer dataset.")
    elif not any('rating' in c.lower() for c in cust.columns):
        st.info("No 'customer_rating' column found in customer dataset.")
    else:
        # Identify rating column automatically
        col_rating = next((c for c in cust.columns if 'rating' in c.lower()), None)

        # Convert data to numeric safely
        cust['delivery_days'] = pd.to_numeric(cust['delivery_days'], errors='coerce')
        cust[col_rating] = pd.to_numeric(cust[col_rating], errors='coerce')

        # --- 1️⃣ Delivery days distribution ---
        st.subheader("Delivery Days Distribution")
        st.plotly_chart(
            px.histogram(
                cust,
                x='delivery_days',
                nbins=30,
                title="Delivery Days Distribution",
                color_discrete_sequence=['#007ACC']
            ),
            use_container_width=True
        )

        # --- 2️⃣ Average rating by delivery duration ---
        st.subheader("Average Rating vs Delivery Days")
        rating_vs_delivery = (
            cust.groupby('delivery_days')[col_rating]
            .mean()
            .reset_index()
            .dropna()
        )

        fig = px.line(
            rating_vs_delivery,
            x='delivery_days',
            y=col_rating,
            markers=True,
            title="Average Customer Rating vs Delivery Days",
            labels={'delivery_days': 'Delivery Days', col_rating: 'Average Rating'},
        )
        fig.add_hline(y=3.5, line_dash="dot", line_color="red", annotation_text="Rating Threshold 3.5")
        st.plotly_chart(fig, use_container_width=True)

        # --- 3️⃣ Performance Evaluation Logic ---
        poor_perf = cust[
            (cust[col_rating] < 3.5) &
            (cust['delivery_days'] > 3)
        ]

        total_orders = len(cust)
        poor_orders = len(poor_perf)
        percent_poor = (poor_orders / total_orders * 100) if total_orders > 0 else 0

        st.subheader("Delivery Performance Evaluation")
        st.metric("Total Orders", f"{total_orders:,}")
        st.metric("Orders with Poor Delivery Experience", f"{poor_orders:,} ({percent_poor:.1f}%)")

        if percent_poor > 20:
            st.error(f"⚠️ Delivery performance is poor — {percent_poor:.1f}% of orders had long delivery times (>3 days) and low ratings (<3.5).")
        elif 10 < percent_poor <= 20:
            st.warning(f"⚠️ Delivery performance is average — {percent_poor:.1f}% of orders impacted.")
        else:
            st.success(f"✅ Delivery performance is good — only {percent_poor:.1f}% of orders affected.")

        # --- 4️⃣ Optional: show detail of poor deliveries ---
        with st.expander("View Sample of Poor Delivery Orders"):
            st.dataframe(
                poor_perf[[col_rating, 'delivery_days', 'customer_id', 'order_date', 'product_id']].head(50)
                if set(['customer_id', 'order_date', 'product_id']).issubset(cust.columns)
                else poor_perf[[col_rating, 'delivery_days']].head(50)
            )

# -------------------------
# Q22 — Payment Analytics
# -------------------------
with tabs[21]:
    st.header("Q22 — Payment Analytics")

    # Check if necessary columns exist
    if 'payment_method' not in cust.columns:
        st.info("No 'payment_method' column found in customer dataset.")
    elif 'order_year' not in cust.columns:
        st.info("No 'order_year' column found for trend analysis.")
    else:
        # --- 1️⃣ Payment Method Share (Pie Chart) ---
        st.subheader("Overall Payment Method Share")
        pm = (
            cust.groupby('payment_method')
            .size()
            .reset_index(name='count')
            .sort_values('count', ascending=False)
        )
        st.plotly_chart(
            px.pie(
                pm,
                names='payment_method',
                values='count',
                title="Overall Payment Method Share",
                hole=0.3,
                color_discrete_sequence=px.colors.qualitative.Pastel
            ),
            use_container_width=True
        )

        # --- 2️⃣ Yearly Trend of Payment Methods ---
        st.subheader("Payment Method Trends (2015–2025)")

        # Prepare data
        cust['order_year'] = pd.to_numeric(cust['order_year'], errors='coerce')
        trend_df = (
            cust.groupby(['order_year', 'payment_method'])
            .size()
            .reset_index(name='count')
        )

        # Compute percentage share each year
        total_per_year = trend_df.groupby('order_year')['count'].transform('sum')
        trend_df['share_percent'] = trend_df['count'] / total_per_year * 100

        # --- 3️⃣ Stacked Area Chart: Market Share over Time ---
        fig_area = px.area(
            trend_df.sort_values(['order_year', 'payment_method']),
            x='order_year',
            y='share_percent',
            color='payment_method',
            title="Evolution of Payment Methods (2015–2025)",
            labels={'order_year': 'Year', 'share_percent': 'Market Share (%)'},
            line_group='payment_method'
        )

        fig_area.update_traces(mode='none')  # smooth filled areas
        fig_area.update_layout(legend_title_text="Payment Method")
        st.plotly_chart(fig_area, use_container_width=True)

        # --- 4️⃣ Insights ---
        st.subheader("Insights Summary")

        latest_year = trend_df['order_year'].max()
        earliest_year = trend_df['order_year'].min()

        pm_2015 = trend_df[trend_df['order_year'] == earliest_year]
        pm_2025 = trend_df[trend_df['order_year'] == latest_year]

        def get_share(df, method):
            val = df.loc[df['payment_method'].str.lower().eq(method.lower()), 'share_percent']
            return round(val.values[0], 1) if not val.empty else 0.0

        upi_2015, upi_2025 = get_share(pm_2015, 'UPI'), get_share(pm_2025, 'UPI')
        cod_2015, cod_2025 = get_share(pm_2015, 'COD'), get_share(pm_2025, 'COD')

        st.write(f"📈 **UPI** usage grew from **{upi_2015}% in {earliest_year}** to **{upi_2025}% in {latest_year}**, showing rapid digital adoption.")
        st.write(f"📉 **Cash on Delivery (COD)** declined from **{cod_2015}%** to **{cod_2025}%**, reflecting increased trust in online payments.")
        st.write("💳 Other methods such as Credit/Debit Cards maintained steady or moderate growth.")

        # --- 5️⃣ Optional: Year-wise detailed table ---
        with st.expander("View Detailed Yearly Payment Data"):
            pivot_df = trend_df.pivot(index='order_year', columns='payment_method', values='share_percent').fillna(0)
            st.dataframe(pivot_df.style.format("{:.1f}%"))

with tabs[22]:
    st.header("Q23 — Returns & Cancellations")
    
    # Check if necessary columns exist
    required_cols = ['return_status', 'product_name', 'customer_id']
    missing_cols = [c for c in required_cols if c not in cust.columns]
    
    if missing_cols:
        st.info(f"Missing columns for analysis: {', '.join(missing_cols)}")
    else:
        # --- 1️⃣ Revenue by return_status (bar chart) ---
        rt = cust.groupby('return_status')[COL_FINAL].sum().reset_index()
        st.plotly_chart(
            px.bar(rt, x='return_status', y=COL_FINAL, title="Revenue by Return Status"),
            use_container_width=True
        )
        
        # --- 2️⃣ Return % and Cancellation % ---
        total_orders = len(cust)
        return_count = len(cust[cust['return_status'].str.lower() == 'returned'])
        cancel_count = len(cust[cust['return_status'].str.lower() == 'cancelled'])
        
        return_pct = round(return_count / total_orders * 100, 2)
        cancel_pct = round(cancel_count / total_orders * 100, 2)
        
        st.subheader("Return & Cancellation Percentages")
        st.write(f"🔹 Return Percentage: {return_pct}%")
        st.write(f"🔹 Cancellation Percentage: {cancel_pct}%")

with tabs[23]:
    st.header("Q24 — Customer Service Dashboard")

    # ----------------------------
    # STEP 1 — Clean up customer ratings first (remove rating = 4.3)
    # ----------------------------
    tmp_col_rating_cust = next((c for c in cust.columns if 'customer_rating' in c.lower()), None)
    if tmp_col_rating_cust:
        cust[tmp_col_rating_cust] = pd.to_numeric(cust[tmp_col_rating_cust], errors='coerce')
        cust = cust[cust[tmp_col_rating_cust] != 4.3]

    # ----------------------------
    # STEP 2 — Identify relevant columns
    # ----------------------------
    col_product_cust = next((c for c in cust.columns if 'product_id' in c.lower()), None)
    col_product_name = next((c for c in cust.columns if 'product_name' in c.lower()), None)
    col_rating_cust = next((c for c in cust.columns if 'customer_rating' in c.lower()), None)
    col_return = next((c for c in cust.columns if 'return_status' in c.lower()), None)
    col_city = next((c for c in cust.columns if 'city' in c.lower()), None)

    col_product_prod = next((c for c in prod.columns if 'product_id' in c.lower()), None)
    col_product_name_prod = next((c for c in prod.columns if 'product_name' in c.lower()), None)
    col_rating_prod = next((c for c in prod.columns if 'rating' in c.lower()), None)
    col_brand = next((c for c in prod.columns if 'brand' in c.lower()), None)
    col_category = next((c for c in prod.columns if 'category' in c.lower()), None)

    # ----------------------------
    # STEP 3 — Customer vs Product Rating Comparison
    # ----------------------------
    if not (col_product_cust and col_rating_cust and col_product_prod and col_rating_prod):
        st.warning("⚠️ Missing required columns for rating comparison.")
    else:
        try:
            cust[col_rating_cust] = pd.to_numeric(cust[col_rating_cust], errors='coerce')
            prod[col_rating_prod] = pd.to_numeric(prod[col_rating_prod], errors='coerce')

            merged = cust[[col_product_cust, col_rating_cust]].merge(
                prod[[col_product_prod, col_product_name_prod, col_rating_prod, col_brand, col_category]],
                left_on=col_product_cust,
                right_on=col_product_prod,
                how='inner'
            )

            merged['rating_difference'] = merged[col_rating_cust] - merged[col_rating_prod]

            product_comparison = (
                merged.groupby([col_product_name_prod, col_brand, col_category])
                .agg({
                    col_rating_cust: 'mean',
                    col_rating_prod: 'mean',
                    'rating_difference': 'mean'
                })
                .reset_index()
            )

            product_comparison.rename(columns={
                col_rating_cust: 'avg_customer_rating',
                col_rating_prod: 'avg_catalog_rating',
                'rating_difference': 'avg_rating_difference'
            }, inplace=True)

            st.subheader("📊 Average Rating Comparison by Product")
            st.dataframe(product_comparison.head(100))

            avg_diff = product_comparison['avg_rating_difference'].mean()
            st.success(f"✅ Average Customer–Catalog Rating Difference: {avg_diff:+.2f}")

        except Exception as e:
            st.error(f"Error generating Customer–Product Rating Comparison: {e}")

    # ----------------------------
    # STEP 4 — City-Wise Rating Analysis
    # ----------------------------
    if col_city and col_rating_cust:
        st.subheader("🏙️ Average Customer Rating by City")
        city_rating = (
            cust.groupby(col_city)[col_rating_cust]
            .mean()
            .reset_index()
            .rename(columns={col_rating_cust: 'avg_customer_rating'})
        )

        def classify_rating(r):
            if r >= 4.5:
                return 'Excellent (4.5+)'
            elif r >= 4.0:
                return 'Good (4.0–4.49)'
            elif r >= 3.5:
                return 'Bad (3.5–3.99)'
            else:
                return 'Poor (<3.5)'

        city_rating['Rating_Level'] = city_rating['avg_customer_rating'].apply(classify_rating)
        st.dataframe(city_rating.head(50))

    # ----------------------------
    # STEP 5 — Return Analysis (Bar Chart + Highlight, using Product Name)
    # ----------------------------
    if col_return and col_rating_cust:
        st.subheader("📦 Poor Quality Products (Returns with Low Ratings)")

        cust[col_return] = cust[col_return].astype(str).str.lower().str.strip()
        cust[col_rating_cust] = pd.to_numeric(cust[col_rating_cust], errors='coerce')

        # Filter returns with rating ≤ 3.5
        poor_returns = cust[
            (cust[col_return].isin(['returned', 'return', 'yes', 'true', '1'])) &
            (cust[col_rating_cust] <= 3.5)
        ]

        if poor_returns.empty:
            st.success("✅ No poor-rated returned products.")
        else:
            st.warning(f"⚠️ Found {len(poor_returns)} poor-rated returned products.")

            # Group by product name if available, else product ID
            group_cols = [col_product_name if col_product_name else col_product_cust]
            poor_summary = (
                poor_returns.groupby(group_cols)
                .agg({
                    col_rating_cust: 'mean',
                    col_return: 'count'
                })
                .reset_index()
                .rename(columns={
                    col_rating_cust: 'avg_customer_rating',
                    col_return: 'return_count'
                })
            )

            # Merge brand/category info for context
            poor_summary = poor_summary.merge(
                prod[[col_product_prod, col_product_name_prod, col_brand, col_category]],
                left_on=col_product_cust if col_product_name is None else col_product_name,
                right_on=col_product_name_prod if col_product_name else col_product_prod,
                how='left'
            )

            # Sort descending by return count
            poor_summary = poor_summary.sort_values('return_count', ascending=False)

            # Display table
            st.dataframe(poor_summary.head(20).style.format({'avg_customer_rating': '{:.2f}'}))

            # --- Visualization: Bar Chart ---
            top_products = poor_summary.head(10).copy()
            max_return_count = top_products['return_count'].max()
            top_products['Highlight'] = top_products['return_count'].apply(
                lambda x: '🔴 Highest Returns' if x == max_return_count else 'Returned'
            )

            # Use product name for X-axis if available
            x_axis_col = col_product_name_prod if col_product_name_prod else col_product_cust

            fig_return = px.bar(
                top_products,
                x=x_axis_col,
                y='return_count',
                color='Highlight',
                color_discrete_map={
                    '🔴 Highest Returns': 'red',
                    'Returned': 'orange'
                },
                hover_data=[col_brand, col_category, 'avg_customer_rating'],
                text='return_count',
                title="Top 10 Products with Poor Ratings & Returns (≤ 3.5)",
                labels={
                    x_axis_col: 'Product Name' if col_product_name_prod else 'Product ID',
                    'return_count': 'Number of Returns'
                }
            )

            fig_return.update_traces(textposition='outside')
            fig_return.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_return, use_container_width=True)

            # --- Summary Message ---
            top_row = top_products.loc[top_products['return_count'].idxmax()]
            top_product_label = (
                top_row[col_product_name_prod]
                if col_product_name_prod and pd.notna(top_row[col_product_name_prod])
                else top_row[col_product_cust]
            )
            top_brand = top_row[col_brand]
            st.info(f"🏆 **Product with Most Returns:** {top_product_label} ({top_brand}) — {int(max_return_count)} returns.")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption(f"Customer file: {CUSTOMER_FILE}  ·  Product file: {PRODUCT_FILE if prod is not None else 'Not loaded'}")
