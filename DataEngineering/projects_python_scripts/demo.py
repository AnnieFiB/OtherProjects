import os
import sys

from pyflakes import checker


def calculate_sales(price_per_unit, quantity_sold):
    """
    Calculate total sales revenue.
    
    Parameters:
    price_per_unit (float): Price of a single unit.
    quantity_sold (int): Number of units sold.
    
    Returns:
    float: Total sales revenue.
    """
    if price_per_unit < 0 or quantity_sold < 0:
        raise ValueError("Price and quantity must be non-negative values.")
    
    total_sales = price_per_unit * quantity_sold
    return total_sales

# Example usage
price = 10.5  # Example price per unit
quantity = 100  # Example quantity sold

total = calculate_sales(price, quantity)
print(f"Total Sales: ${total:.2f}")
