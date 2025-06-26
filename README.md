# Customer Order Data Extraction and Transformation

This project provides a robust Python class \`CustomerDataExtractor\` to load, clean, and flatten nested customer order data from a pickled file and a VIP customers list, producing a clean Pandas DataFrame ready for analysis.

---

## Features

- Loads nested customer order data from \`customer_orders.pkl\`.
- Reads VIP customer IDs from \`vip_customers.txt\`.
- Handles dirty data, including prices with currency symbols and non-numeric quantities.
- Converts integer category codes to descriptive strings with fallback.
- Calculates total item prices and their percentage contribution to the order.
- Ensures strict data types and sorting order.
- Outputs a clean, flat DataFrame with all relevant columns for downstream analysis.

---

## Usage

1. Clone this repository:

   \`\`\`bash
   git clone <repo-url>
   cd <repo-folder>
   \`\`\`

2. Ensure the data files \`customer_orders.pkl\` and \`vip_customers.txt\` are in the repository root.

3. Run the script or import the class in your Python environment:

   \`\`\`bash
   python customer_data_extractor.py
   \`\`\`

4. (Optional) Export the resulting DataFrame to CSV by uncommenting the export line in the script.

---

## Output DataFrame Columns

| Column                     | Type            | Description                              |
|----------------------------|-----------------|------------------------------------------|
| customer_id                | int             | Unique customer identifier                |
| customer_name              | string          | Customer full name                        |
| registration_date          | datetime64[ns]  | Customer registration date (nullable)   |
| is_vip                    | bool            | VIP status based on \`vip_customers.txt\` |
| order_id                  | int             | Unique order identifier                   |
| order_date                | datetime64[ns]  | Date when order was placed (nullable)    |
| product_id                | int             | Unique product/item identifier            |
| product_name              | string          | Product name                             |
| category                  | string          | Product category (mapped with fallback) |
| unit_price                | float           | Price per unit item                      |
| item_quantity             | int             | Quantity ordered                         |
| total_item_price          | float           | unit_price * item_quantity               |
| total_order_value_percentage | float         | Percentage contribution to order total   |

---

## Dependencies

- Python 3.7+
- pandas
- pickle (built-in)

Install pandas via:

\`\`\`bash
pip install pandas
\`\`\`


---

## Contact

For questions or feedback, please contact Meri Asatryan at meri_asatryan@edu.aua.am].

