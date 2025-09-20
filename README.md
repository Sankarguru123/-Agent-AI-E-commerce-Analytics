# Agent AI E-commerce Analytics

An AI-powered analytics system that lets users ask natural language questions about **e-commerce data** and automatically generates SQL queries + code snippets (Pandas, PySpark, Scala, R, JavaScript, Java, C#).  

The project includes:
- **Backend (Flask + Groq API)** → Converts NL queries into SQL/code.
- **Frontend (Streamlit)** → Upload CSV or connect to DB, visualize results with charts, and export reports.

---

## 📂 Project Structure
```
Agent_AI_E-commerce_Analytics/
│── backend/
│   ├── .env                # Environment variables (Groq API key etc.)
│   ├── app.py              # Flask backend
│   ├── Dockerfile          # Backend container
│   ├── llm_cache.db        # SQLite cache for queries
│   └── requirements.txt    # Backend dependencies
│
│── frontend/
│   ├── sample_data/        # Example CSV data
│   ├── Dockerfile          # Frontend container
│   ├── requirements.txt    # Frontend dependencies
│   └── streamlit_app.py    # Streamlit UI
│
└── README.md               # Project documentation
```

---

## ✨ Features
- Upload **CSV dataset** or connect to **SQLLITE/Postgres/MySQL DB**.
- Ask questions like:
  - *"How many customers purchased more than once this year?"*
  - *"Show revenue by product category in Q2."*
  📌 Example Natural Language Questions
👥 Customer Analytics

- *"How many customers purchased more than once this year?"*

- *"List the top 10 customers by total spending."*

- *"Which customers purchased in both Q1 and Q2?"*

- *"Show customers who bought more than 5 different products."*

- *"How many new customers joined in the last 6 months?"*

📦 Product Analytics

- *"Show revenue by product category in Q2."*

- *"Which product had the highest sales last month?"*

"Top 5 most returned products."*

- *"Which products were purchased together most often?"*

- *"Show average price per product category."*

💰 Revenue & Sales

- *"Total revenue this year vs last year."*

- *"Monthly sales trend for 2024."*

- *"Which quarter had the highest revenue in 2023?"*

- *"Show daily revenue for the last 30 days."*

- *"What percentage of sales came from repeat customers?"*

📅 Time-based Insights

- *"Revenue growth by quarter in 2022."*

- *"Which month had the highest number of new orders?"*

- *"Weekly revenue trend for Q1 2024."*

- *"How many orders were placed on weekends vs weekdays?"*

- *"Year-over-year growth for product category Electronics."*

🌍 Geography (if Region/Country data available)

- *"Top 5 countries by revenue in 2023."*

- *"Which region had the highest average order value?"*

- *"Monthly sales trend in Asia-Pacific region."*

- *"Which city had the most customers last year?"*

- *"Compare revenue between rural vs urban regions."*

- Backend generates:
  - ✅ SQLite-safe SQL (auto converts Postgres-style `EXTRACT` → `strftime`)
  - ✅ Equivalent code snippets for Pandas, PySpark, Scala, R, JavaScript, Java, C#.
- Frontend:
  - Interactive grid view
  - Pie/Bar/Line charts
  - Export to Excel, PDF, PowerPoint

---

## ⚙️ Setup

### 1. Clone repo
```bash
git clone <repo-url>
cd Agent_AI_E-commerce_Analytics
```

### 2. Backend Setup
```bash
cd backend
pip install -r requirements.txt
```

Create `.env` inside `backend/`:
```ini
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
LLM_CACHE_DB=llm_cache.db
```

Run backend:
```bash
python app.py
```
👉 Runs on: `http://localhost:5000`

---

### 3. Frontend Setup
```bash
cd ../frontend
pip install -r requirements.txt
```

Run frontend:
```bash
streamlit run streamlit_app.py
```
👉 Opens at: `http://localhost:8501`

---

## 🔍 Testing Backend

### Health Check
```bash
curl http://localhost:5000/health
```

### Generate SQL
```bash
curl -X POST "http://localhost:5000/generate_sql"   -H "Content-Type: application/json"   -d '{
    "nl_query": "Show revenue by quarter",
    "schema": {
      "OrderID":"INTEGER","CustomerName":"TEXT","ProductID":"TEXT",
      "ProductName":"TEXT","Quantity":"INTEGER","Price":"REAL","OrderDate":"DATE"
    }
  }'
```

Response:
```json
{
  "ok": true,
  "sql": "SELECT CASE ... END AS Quarter, SUM(Quantity*Price) AS Revenue FROM orders GROUP BY Quarter LIMIT 1000;",
  "implementations": { "pandas": "...", "pyspark": "...", "scala": "...", ... },
  "cached": false
}
```

---

## 📊 Frontend Usage
1. Upload CSV (`orders.csv` with OrderID, CustomerName, ProductID, ProductName, Quantity, Price, OrderDate).
2. Enter a natural language query.
3. View:
   - Generated SQL
   - Query results (grid view)
   - Auto charts (pie, bar, line)
4. Export reports:
   - Excel
   - PDF
   - PowerPoint

---

## 🧰 SQLite Auto-Fix
The backend includes `_convert_sql_to_sqlite()` that rewrites invalid SQL (like `EXTRACT(MONTH FROM OrderDate)`) into valid SQLite (`strftime('%m', OrderDate)`).

---
