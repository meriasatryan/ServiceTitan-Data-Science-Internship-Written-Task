import pickle
import pandas as pd
from datetime import datetime


class CustomerDataExtractor:
    """
    A class to load, clean, and flatten customer order data for analysis.
    """

    CATEGORY_MAPPING = {
        1: 'Electronics',
        2: 'Apparel',
        3: 'Books',
        4: 'Home Goods'
    }

    COLUMN_TYPES = {
        'customer_id': 'int64',
        'customer_name': 'string',
        'registration_date': 'datetime64[ns]',
        'is_vip': 'bool',
        'order_id': 'int64',
        'order_date': 'datetime64[ns]',
        'product_id': 'int64',
        'product_name': 'string',
        'category': 'string',
        'unit_price': 'float64',
        'item_quantity': 'int64',
        'total_item_price': 'float64',
        'total_order_value_percentage': 'float64'
    }

    def __init__(self, orders_file='customer_orders.pkl', vip_file='vip_customers.txt'):
        self.orders_file = orders_file
        self.vip_file = vip_file
        self.raw_data = []
        self.vip_ids = set()

    def load_data(self):
        """Load customer orders and VIP IDs from files."""
        try:
            with open(self.orders_file, 'rb') as f:
                self.raw_data = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load orders file: {e}")

        try:
            with open(self.vip_file, 'r') as f:
                self.vip_ids = {int(line.strip()) for line in f if line.strip().isdigit()}
        except Exception as e:
            raise RuntimeError(f"Failed to load VIP customer IDs: {e}")

    @staticmethod
    def parse_int(value, default=0):
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def parse_price(price_raw):
        try:
            return float(str(price_raw).replace('$', '').replace(',', '').strip())
        except (ValueError, AttributeError):
            return 0.0

    def transform(self) -> pd.DataFrame:
        """Transform nested data into a flat DataFrame with clean, typed columns."""
        flat_records = []

        for customer in self.raw_data:
            customer_id = self.parse_int(customer.get('id'))
            customer_name = customer.get('name', '').strip()
            registration_date = pd.to_datetime(customer.get('registration_date'), errors='coerce')
            is_vip = customer_id in self.vip_ids

            for order in customer.get('orders', []):
                order_id = self.parse_int(order.get('order_id'))
                order_date = pd.to_datetime(order.get('order_date'), errors='coerce')
                items = order.get('items', [])

                # Calculate total order value once using list comprehension
                order_total = sum(
                    self.parse_price(item.get('price', 0.0)) * self.parse_int(item.get('quantity'))
                    for item in items
                )

                flat_records.extend([
                    {
                        'customer_id': customer_id,
                        'customer_name': customer_name,
                        'registration_date': registration_date,
                        'is_vip': is_vip,
                        'order_id': order_id,
                        'order_date': order_date,
                        'product_id': self.parse_int(item.get('item_id')),
                        'product_name': item.get('product_name', '').strip(),
                        'category': self.CATEGORY_MAPPING.get(item.get('category'), 'Misc'),
                        'unit_price': self.parse_price(item.get('price', 0.0)),
                        'item_quantity': self.parse_int(item.get('quantity')),
                        'total_item_price': self.parse_price(item.get('price', 0.0)) * self.parse_int(item.get('quantity')),
                        'total_order_value_percentage': (
                            (self.parse_price(item.get('price', 0.0)) * self.parse_int(item.get('quantity')) / order_total * 100)
                            if order_total > 0 else 0.0
                        )
                    }
                    for item in items
                ])

        df = pd.DataFrame(flat_records)

        # Enforce column types
        df = df.astype(self.COLUMN_TYPES)

        # Sort and reset index
        df.sort_values(by=['customer_id', 'order_id', 'product_id'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df


# --- Script Execution Block ---
if __name__ == '__main__':
    extractor = CustomerDataExtractor(
        orders_file='customer_orders.pkl',
        vip_file='vip_customers.txt'
    )

    try:
        extractor.load_data()
        df = extractor.transform()

        # Output DataFrame summary
        print("\n✅ Data loaded and transformed successfully!\n")
        print(df.info())
        print(df.head())

        # Optionally export
        # df.to_csv("flattened_customer_orders.csv", index=False)

    except Exception as e:
        print(f"\n❌ Error during processing: {e}\n")