
# **Inventory and Management Model**
"""

# synthetic_data_generator.py

import random
from datetime import datetime, timedelta
import pandas as pd

def generate_synthetic_data(company_name='DemoCompany', num_products=5, num_suppliers=3):
    """
    Generates synthetic inventory and supplier CSV data for testing.
    Saves files as {company_name}_inventory.csv and {company_name}_suppliers.csv.
    """

    print(f"Generating synthetic data for company '{company_name}' with {num_products} products and {num_suppliers} suppliers.")

    # Generate inventory data
    inv_rows = []
    for i in range(1, num_products + 1):
        pid = f"P{i:03d}"
        pname = f"Product_{i}"
        stock_qty = random.randint(20, 150)
        capacity = stock_qty + random.randint(50, 200)
        inv_rows.append([pid, pname, stock_qty, capacity])
    inventory_df = pd.DataFrame(inv_rows, columns=['product_id', 'product_name', 'stock_qty', 'inventory_capacity'])

    # Generate supplier data
    sup_rows = []
    for s in range(1, num_suppliers + 1):
        sid = f"S{s:02d}"
        # Each supplier supplies 1 or 2 random products
        supplied_products = random.sample(inventory_df['product_id'].tolist(), k=random.randint(1, 2))
        for pid in supplied_products:
            lead_time = random.randint(2, 10)  # lead time in days
            next_order_qty = random.randint(10, 100)
            # next order date within next 7 days
            next_order_date = (datetime.today() + timedelta(days=random.randint(0,7))).date().isoformat()
            sup_rows.append([sid, pid, lead_time, next_order_qty, next_order_date])
    supplier_df = pd.DataFrame(sup_rows, columns=['supplier_id','product_id','lead_time_days','next_order_qty','next_order_date'])

    # Save to CSV
    safe_name = company_name.replace(' ', '_')
    inv_path = f"{safe_name}_inventory.csv"
    sup_path = f"{safe_name}_suppliers.csv"
    inventory_df.to_csv(inv_path, index=False)
    supplier_df.to_csv(sup_path, index=False)

    print(f"Synthetic inventory data saved to: {inv_path}")
    print(f"Synthetic supplier data saved to: {sup_path}")

if __name__ == "__main__":
    cname = input("Enter company name for synthetic data: ").strip() or "DemoCompany"
    try:
        num_products = int(input("Number of products to generate (default 5): ").strip() or "5")
        num_suppliers = int(input("Number of suppliers to generate (default 3): ").strip() or "3")
    except Exception:
        print("Invalid input. Using default values: 5 products, 3 suppliers.")

        num_products = 5
        num_suppliers = 3

    generate_synthetic_data(cname, num_products, num_suppliers)

# inventory_agentic_ai.py

import os
import math
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Try imports for Prophet and pytrends
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except Exception:
    PYTRENDS_AVAILABLE = False

def fetch_google_trends(category, days=90):
    end = datetime.today().date()
    start = end - timedelta(days=days-1)
    dates = pd.date_range(start, end).date

    if PYTRENDS_AVAILABLE:
        try:
            pytrends = TrendReq(timeout=(10,25))
            pytrends.build_payload([category], timeframe=f'{start.isoformat()} {end.isoformat()}')
            df = pytrends.interest_over_time()
            if df.empty:
                raise RuntimeError("pytrends returned empty data")
            df = df.reset_index().rename(columns={'date': 'date', category: 'trend_index'})
            df['date'] = pd.to_datetime(df['date']).dt.date
            return df[['date','trend_index']]
        except Exception as e:
            print("⚠️ Google Trends fetch failed, using synthetic data fallback:", e)

    # Synthetic fallback
    rng = np.arange(days)
    weekly = 10 * (1 + np.sin(2 * np.pi * rng / 7))
    trend = 0.05 * rng
    noise = np.random.normal(0, 2.5, size=days)
    values = np.clip(50 + weekly + trend + noise, 0, None).round().astype(int)
    return pd.DataFrame({'date': dates, 'trend_index': values})

def fetch_platform_mock(category, platform, days=90):
    end = datetime.today().date()
    start = end - timedelta(days=days-1)
    dates = pd.date_range(start, end).date
    product_ids = [f'{category[:3].upper()}-{i}' for i in range(1,6)]

    rows = []
    for d in dates:
        base = {
            'Amazon': 20,
            'Flipkart': 12,
            'Shopify': 6,
            'eBay': 8,
            'Other': 4
        }.get(platform, 5)
        weekday = d.weekday()
        weekday_factor = 1.2 if weekday in (4,5) else 0.9
        qty = max(0, int(np.random.poisson(base * weekday_factor)))
        pid = np.random.choice(product_ids)
        price = round(500 + np.random.normal(0, 30), 2)
        cost = price * np.random.uniform(0.6, 0.85)
        profit = max(0.0, round(price - cost, 2)) * qty
        rows.append([platform, d, pid, qty, price, profit])
    return pd.DataFrame(rows, columns=['platform','date','product_id','quantity','price','profit'])

def aggregate_market_data(category, platforms=None, days=90):
    if platforms is None:
        platforms = ['Amazon','Flipkart','Shopify','eBay','Other']
    trends_df = fetch_google_trends(category, days=days)
    platform_dfs = [fetch_platform_mock(category, p, days=days) for p in platforms]
    platforms_all = pd.concat(platform_dfs, ignore_index=True)
    platforms_all['date'] = pd.to_datetime(platforms_all['date']).dt.date
    platforms_all['quantity'] = platforms_all['quantity'].astype(int)
    platforms_all['profit'] = platforms_all['profit'].astype(float)
    return trends_df, platforms_all

def run_forecast_and_analysis(category, inventory_df, supplier_df, platforms_all, trends_df, prediction_days=14):
    daily_sales = platforms_all.groupby('date', as_index=False)['quantity'].sum().rename(columns={'date':'ds','quantity':'y'})
    daily_sales['ds'] = pd.to_datetime(daily_sales['ds'])
    idx = pd.date_range(daily_sales['ds'].min(), daily_sales['ds'].max())
    daily_sales = daily_sales.set_index('ds').reindex(idx, fill_value=0).rename_axis('ds').reset_index()

    trends_df2 = trends_df.copy()
    trends_df2['ds'] = pd.to_datetime(trends_df2['date'])
    trends_df2 = trends_df2[['ds','trend_index']].set_index('ds').reindex(pd.date_range(trends_df['date'].min(), trends_df['date'].max())).fillna(method='ffill').reset_index().rename(columns={'index':'ds'})
    trends_df2['ds'] = pd.to_datetime(trends_df2['ds'])

    merged = pd.merge(daily_sales, trends_df2, on='ds', how='left')
    merged['trend_index'] = merged['trend_index'].fillna(merged['trend_index'].mean())

    if PROPHET_AVAILABLE:
        m = Prophet(interval_width=0.95)
        m.add_regressor('trend_index')
        m.fit(merged[['ds','y','trend_index']])
        future = m.make_future_dataframe(periods=prediction_days)
        last_trend = merged['trend_index'].iloc[-1]
        future_trend = np.linspace(last_trend, last_trend*(1+0.02), num=len(future))
        future['trend_index'] = future_trend
        forecast = m.predict(future)
    else:
        base_mean = merged['y'].tail(14).mean() if not merged.empty else 1.0
        ds_future = pd.date_range(merged['ds'].max() + pd.Timedelta(days=1), periods=prediction_days)
        trend_factor = merged['trend_index'].iloc[-1] / max(1, merged['trend_index'].mean())
        yhat = [max(0, base_mean * trend_factor * (1 + 0.02*i)) for i in range(prediction_days)]
        lower = [max(0, v * 0.7) for v in yhat]
        upper = [v * 1.3 for v in yhat]
        forecast = pd.DataFrame({'ds': ds_future, 'yhat': yhat, 'yhat_lower': lower, 'yhat_upper': upper})
        m = None

    current_stock = int(inventory_df['stock_qty'].sum()) if not inventory_df.empty else 0
    capacity = int(inventory_df['inventory_capacity'].sum()) if not inventory_df.empty else 0

    repl_rows = []
    if not supplier_df.empty:
        for _, s in supplier_df.iterrows():
            try:
                qty = int(s['next_order_qty'])
                date = pd.to_datetime(s['next_order_date']).date()
                lead = int(s['lead_time_days'])
                arrival = date + timedelta(days=lead)
                if qty > 0 and arrival >= datetime.today().date():
                    repl_rows.append({'arrival_date': arrival, 'qty': qty, 'supplier_id': s['supplier_id']})
            except Exception:
                continue
    repl_df = pd.DataFrame(repl_rows)

    fut = forecast[['ds','yhat','yhat_lower','yhat_upper']] if 'yhat' in forecast.columns else forecast.copy()
    fut['date'] = pd.to_datetime(fut['ds']).dt.date
    fut = fut.head(prediction_days).reset_index(drop=True)

    sim_rows = []
    stock = current_stock
    for _, row in fut.iterrows():
        day = row['date']
        demand = max(0, float(row.get('yhat', 0)))
        arrivals = int(repl_df[repl_df['arrival_date'] == day]['qty'].sum()) if not repl_df.empty else 0
        stock += arrivals
        stock -= math.ceil(demand)
        sim_rows.append({'date': day, 'forecast_demand': demand, 'arrivals': arrivals, 'stock_after': stock})
    sim_df = pd.DataFrame(sim_rows)
    stockout_rows = sim_df[sim_df['stock_after'] <= 0]
    stockout_date = stockout_rows['date'].iloc[0] if not stockout_rows.empty else None
        # Predict order placement date and arrival date based on lead time
    lead_time = int(supplier_df['lead_time_days'].mean()) if not supplier_df.empty else 7
    order_place_date = None
    predicted_arrival_date = None
    if stockout_date:
        order_place_date = stockout_date - timedelta(days=lead_time)
        predicted_arrival_date = order_place_date + timedelta(days=lead_time)


    sum_pred = fut['yhat'].sum() if 'yhat' in fut.columns else 0
    sum_upper = fut['yhat_upper'].sum() if 'yhat_upper' in fut.columns else sum_pred * 1.2
    safety_stock = max(0, float(sum_upper - sum_pred))

    total_arrivals = repl_df['qty'].sum() if not repl_df.empty else 0
    if stockout_date:
        reorder_qty = max(0, math.ceil(sum_pred + safety_stock - (current_stock + total_arrivals)))
    else:
        reorder_qty = max(0, math.ceil(safety_stock - (current_stock + total_arrivals - sum_pred)))

    capacity_warning = (current_stock + total_arrivals) > capacity if capacity > 0 else False

    avg_profit_per_unit = 0.0
    if not platforms_all.empty:
        total_q = platforms_all['quantity'].sum()
        total_p = platforms_all['profit'].sum()
        avg_profit_per_unit = (total_p / total_q) if total_q > 0 else 0.0

    expected_profit = avg_profit_per_unit * sum_pred
    expected_profit_lower = expected_profit * 0.85
    expected_profit_upper = expected_profit * 1.15

    results = {
        'model': m,
        'forecast': forecast,
        'simulated_inventory': sim_df,
        'stockout_date': stockout_date,
        'safety_stock': safety_stock,
        'reorder_qty_suggestion': int(reorder_qty),
        'capacity_warning': capacity_warning,
        'scheduled_replenishments': repl_df,
        'profit_range': (float(expected_profit_lower), float(expected_profit_upper)),
        'demand_range': (float(fut['yhat_lower'].sum()) if 'yhat_lower' in fut.columns else float(sum_pred*0.9),
                         float(fut['yhat_upper'].sum()) if 'yhat_upper' in fut.columns else float(sum_pred*1.1)),
        'current_stock': current_stock,
        'capacity': capacity,
        'platforms_summary': platforms_all.groupby('platform', as_index=False)['profit'].sum().sort_values('profit', ascending=False),
        'order_place_date': order_place_date,
        'predicted_arrival_date': predicted_arrival_date,

    }
    return results

def display_and_decide(category, results, prediction_days=14):
    print(f"\n--- Analysis for category: {category} ---")
    print(f"Current total stock: {results['current_stock']}")
    print(f"Inventory capacity: {results['capacity']}")
    print(f"Forecasted demand (next {prediction_days} days) range: {results['demand_range'][0]:.1f} - {results['demand_range'][1]:.1f}")
    pr_lo, pr_hi = results['profit_range']
    print(f"Estimated profit range (next {prediction_days} days): ${pr_lo:.2f} - ${pr_hi:.2f}")
    print(f"Suggested reorder quantity: {results['reorder_qty_suggestion']}")
    if results['capacity_warning']:
        print("⚠️  Incoming + current stock may exceed inventory capacity.")
    if not results['scheduled_replenishments'].empty:
        print("\nScheduled replenishments (arrival_date, qty):")
        print(results['scheduled_replenishments'])
    if results.get('order_place_date'):
        print(f"Suggested order placement date: {results['order_place_date']}")
    if results.get('predicted_arrival_date'):
        print(f"Predicted arrival date: {results['predicted_arrival_date']}")


    print("\nPlatform profit summary:")
    for _, r in results['platforms_summary'].iterrows():
        print(f"- {r['platform']}: ${r['profit']:.2f}")

    # Plots
    forecast = results['forecast'].copy()
    if 'yhat' in forecast.columns:
        plt.figure(figsize=(12,6))
        plt.plot(pd.to_datetime(forecast['ds']), forecast['yhat'], label='Forecast')
        if 'yhat_lower' in forecast and 'yhat_upper' in forecast:
            plt.fill_between(pd.to_datetime(forecast['ds']), forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2, label='95% CI')
        plt.title(f"Forecast for {category}")
        plt.xlabel('Date'); plt.ylabel('Quantity'); plt.legend(); plt.grid(True)
        plt.show()

    sim = results['simulated_inventory']
    if not sim.empty:
        plt.figure(figsize=(10,4))
        plt.plot(sim['date'], sim['stock_after'], marker='o', label='Projected Stock After Demand')
        plt.axhline(0, color='r', linestyle='--', label='Stockout')
        plt.title('Projected Inventory over Forecast Horizon')
        plt.xlabel('Date'); plt.ylabel('Stock after demand'); plt.legend(); plt.grid(True)
        plt.show()

    ps = results['platforms_summary']
    if not ps.empty:
        plt.figure(figsize=(8,4))
        plt.bar(ps['platform'], ps['profit'])
        plt.title(f"Platform Profit Comparison for {category}")
        plt.ylabel('Total Profit'); plt.xticks(rotation=45); plt.grid(axis='y'); plt.show()

    while True:
        decision = input("\nDo you want to proceed with the suggested reorder? (yes/no/modify): ").strip().lower()
        if decision in ('yes','no','modify'):
            break
        print("Type 'yes', 'no', or 'modify'.")

    final_reorder = results['reorder_qty_suggestion']
    if decision == 'yes':
        print(f"✅ Proceeding with reorder of {final_reorder} units.")
    elif decision == 'no':
        final_reorder = 0
        print("❌ No reorder will be placed.")
    else:
        while True:
            try:
                new_q = int(input("Enter your desired reorder quantity (integer >=0): ").strip())
                if new_q < 0:
                    raise ValueError()
                final_reorder = new_q
                print(f"✅ Updated reorder quantity set to {final_reorder}")
                break
            except Exception:
                print("Please enter a valid integer >= 0.")

    log_row = {
        'timestamp': datetime.now().isoformat(),
        'category': category,
        'suggested': results['reorder_qty_suggestion'],
        'final_reorder': final_reorder,
        'decision': decision
    }
    log_df = pd.DataFrame([log_row])
    log_path = f"{category.replace(' ','_')}_reorder_decisions.csv"
    if os.path.exists(log_path):
        log_df.to_csv(log_path, mode='a', header=False, index=False)
    else:
        log_df.to_csv(log_path, index=False)
    print(f"Decision saved to {log_path}")

def main():
    print("=== Inventory Forecasting & Market-Aware Reorder Assistant ===")
    cname = input("Enter company name (used in saved data CSVs): ").strip()
    safe_name = cname.replace(' ','_')
    inv_path = f"{safe_name}_inventory.csv"
    sup_path = f"{safe_name}_suppliers.csv"

    if not os.path.exists(inv_path) or not os.path.exists(sup_path):
        print(f"❌ Data files not found for '{cname}'. Run synthetic data generator or save company data first.")
        return

    inventory_df = pd.read_csv(inv_path)
    supplier_df = pd.read_csv(sup_path)
    category = input("Enter category/product type to analyze (e.g., 'saree', 'laptop'): ").strip()
    if not category:
        print("Category required.")
        return

    print("\nFetching market data (Google Trends + platform mocks)...")
    trends_df, platforms_all = aggregate_market_data(category)

    print("Running forecast and analysis...")
    results = run_forecast_and_analysis(category, inventory_df, supplier_df, platforms_all, trends_df)

    display_and_decide(category, results)

if __name__ == "__main__":
    main()



"""# **Negotiation**"""

import random
from datetime import datetime, timedelta

def inventory_flow_with_restock():
    print("=== Inventory Forecasting & Reorder Assistant ===")
    company_name = input("Enter company name: ").strip()
    if not company_name:
        print("Company name required.")
        return

    category = input("Enter category/product type to analyze (e.g., 'saree', 'laptop'): ").strip()
    if not category:
        print("Category required.")
        return

    suggested_reorder_qty = random.randint(100, 300)
    print(f"\nSuggested reorder quantity for '{category}': {suggested_reorder_qty}")

    while True:
        restock = input(f"Do you want to restock '{category}'? (yes/no): ").strip().lower()
        if restock in ('yes','no'):
            break
        print("Please type 'yes' or 'no'.")

    if restock == 'yes':
        run_supplier_discovery(category, suggested_reorder_qty)
    else:
        print(f"Restock declined for '{category}'. No supplier search initiated.")

def run_supplier_discovery(category, order_qty):
    print(f"\n=== Supplier Discovery & Negotiation for '{category}' ===")
    print("Searching for cost-effective raw product suppliers...")

    suppliers = []
    for i in range(5):
        suppliers.append({
            'name': f'Supplier_{i+1}',
            'location': random.choice(['Bangalore', 'Chennai', 'Mumbai', 'Delhi', 'Hyderabad']),
            'product_category': category,
            'price_per_unit': round(random.uniform(20.0, 30.0), 2),
            'rating': round(random.uniform(3.5, 5.0), 1),
            'lead_time_days': random.randint(3, 10)
        })

    print(f"\nFound {len(suppliers)} suppliers for '{category}':\n")
    for i, s in enumerate(suppliers, 1):
        print(f"{i}. {s['name']} | Location: {s['location']} | Price/unit: ${s['price_per_unit']} | "
              f"Rating: {s['rating']} stars | Lead Time: {s['lead_time_days']} days")

    for s in suppliers:
        s['score'] = (5 - s['price_per_unit']/10) + s['rating']

    best_supplier = max(suppliers, key=lambda x: x['score'])
    print(f"\nBest supplier based on price, rating, and lead time: {best_supplier['name']}")

    while True:
        choice = input("\nDo you want to negotiate with the best supplier automatically? (yes/manual/skip): ").strip().lower()
        if choice == 'skip':
            print("Supplier negotiation skipped.")
            return
        elif choice == 'yes':
            negotiate_with_supplier(best_supplier, order_qty)
            return
        elif choice == 'manual':
            while True:
                num = input(f"Enter supplier number (1-{len(suppliers)}) or 'cancel': ").strip().lower()
                if num == 'cancel':
                    print("Manual negotiation cancelled.")
                    return
                if num.isdigit() and 1 <= int(num) <= len(suppliers):
                    negotiate_with_supplier(suppliers[int(num)-1], order_qty)
                    return
                print("Invalid choice.")
        else:
            print("Please enter 'yes', 'manual', or 'skip'.")

def negotiate_with_supplier(supplier, order_qty):
    print(f"\nNegotiating with {supplier['name']} ({supplier['location']})...")
    print(f"Requested quantity: {order_qty} units at ${supplier['price_per_unit']} each.")

    discount = round(random.uniform(0.05, 0.15), 2)
    final_price = round(supplier['price_per_unit'] * (1 - discount), 2)
    print(f"Supplier offers a {discount*100:.0f}% discount. Final price per unit: ${final_price}")

    total_cost = round(order_qty * final_price, 2)
    arrival_date = (datetime.today() + timedelta(days=supplier['lead_time_days'])).date()

    print(f"Total cost: ${total_cost}")
    print(f"Predicted arrival date: {arrival_date}")
    print("✅ Purchase order placed successfully!")

if __name__ == "__main__":
    inventory_flow_with_restock()



"""# **Central Control and Order Management**"""

from datetime import datetime, timedelta

def generate_synthetic_pending_pos(company_name):
    # Synthetic product and location data
    products = [
        {'product_id': 'P1001', 'product_name': 'Laptop'},
        {'product_id': 'P1002', 'product_name': 'Mobiles'},
        {'product_id': 'P1003', 'product_name': 'RAM'}
    ]

    locations = ['Bangalore', 'Chennai', 'Mumbai']

    # Use fixed reorder quantities inspired by your inventory model output
    suggested_reorders = [230, 320, 158]  # sum close to 708 reorder qty in your example

    # Generate synthetic pending POs combining product, location, qty
    pending_pos = []
    for i in range(min(3, len(products))):
        po = {
            'product_id': products[i]['product_id'],
            'product_name': products[i]['product_name'],
            'location': locations[i],
            'quantity': suggested_reorders[i],
            'supplier': f"Supplier_{locations[i]}",
            'order_date': datetime.today().date(),
            'expected_arrival': datetime.today().date() + timedelta(days=7),
            'status': 'Pending'
        }
        pending_pos.append(po)
    return pending_pos

def confirm_shipments_flow():
    print("=== Pending Purchase Orders Confirmation ===")
    company_name = input("Enter company name: ").strip()
    print(f"\nFetching synthetic pending purchase orders for '{company_name}'...\n")

    pending_pos = generate_synthetic_pending_pos(company_name)

    for po in pending_pos:
        print(f"Product ID: {po['product_id']} | Product: {po['product_name']} | Location: {po['location']} | Qty: {po['quantity']} | Supplier: {po['supplier']} | Order Date: {po['order_date']} | Expected Arrival: {po['expected_arrival']} | Status: {po['status']}")

        while True:
            resp = input("Confirm shipment for this PO? (yes/no): ").strip().lower()
            if resp in ('yes','no'):
                break
            print("Please type 'yes' or 'no'.")

        if resp == 'yes':
            po['status'] = 'Confirmed'
            print(f"Shipment for {po['product_name']} confirmed.\n")
        else:
            print(f"Shipment for {po['product_name']} remains pending.\n")

if __name__ == "__main__":
    confirm_shipments_flow()

