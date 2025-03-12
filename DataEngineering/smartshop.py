# create a list and a dictionary for inventory and completed orders
inventory = {}
completed_orders =[]

# ====================================
# 1. Inventory management module
# ====================================

def add_update_inventory():
    """
        add a new product or update an existing product 
    """
    product_id = input("Enter the product id: ").strip()
    name = input("Enter the product name: ").strip()
    try:
        price = float(input("Enter the product price: "))
        stock = int(input("Enter the product stock: "))
    except ValueError:
        print("Invalued input.Price must be a number and stock must be an integer")
        return
    
    if product_id in inventory:
        print("product already exists. Updating product")
    else:
        print("Adding new product")

    inventory[product_id]={
        "name":name,
        "price":price,
        "stock":stock
    }
    print("Product added/updated successfully")

def stock_validation(product_id, requested_quantity):
    """
    This function is used to check if the requested quantity of a product exists
    params: product_id: this is a string that points to the product in the inventory
             requested_quantity: this is the quantity of product requested by the customer
    """
    if product_id not in inventory:
        print("Product does not exist")
        return False
    
    available = inventory[product_id]["stock"]
    if available >= requested_quantity:
        return True
    else:
        print("Insufficient stock")
        return False

# ================================================================
# 2. Order management module
# =============================================================
def calculate_total(order):
    """
    This function is used to calculate the total cost of an order
    params: order: this is a dictionary that contains the products and their quantities.
    """
    total = 0.0
    for pid, qty in order.items():
        price = inventory[pid]["price"]
        total += price * qty
        return total

def generate_receipt(order, total):
    """"
    "This function is used to generate a receipt for a completed order"
    "params: order: this is a dict taht contains the products and their qty"
    "total: this is the total cost of the order"
    """
    print("\n---Receipt---")
    for pid, qty in order.items():
        name = inventory[pid]["name"]
        price = inventory[pid]["price"]
        line_total = price * qty
        print(f"{name} (ID: {pid}) ~ Quantity: {qty}, unit Price: ${price: .2f}, Subtotal: ${line_total: .2f}")
        print("__________________________")
        print("Total Amount: ${total: .2f}")
        print("__________________________")

def place_order():
    """
    this function is used to place an order
    """
    order = {}
    print("\nEnter your order. type 'done' when order completed")
    while True:
        product_id = input("Enter product_id: ").strip()
        if product_id == "done":
            break #or False
        if product_id not in inventory:
            print("Product does not exists")
            continue
        try:
            qty = int(input("Enter quantity: "))
        except ValueError:
            print("Invalid input. Quantity must be an integer")
            continue
        if qty <= 0:
            print("Quantity must not be greater than 0")
            continue
        if stock_validation(product_id, qty):
            order[product_id] = order.get(product_id, 0) + qty
            print("Product added to the dictionary")
        else:
            print("Cannot add order due to insufficient stock")
    if not order:
        print("No products were ordered")
    
    total = calculate_total(order)
    for pid, qty in order.items():
        inventory[pid]["Stock"] -= qty
    
    generate_receipt(order, total)
    completed_orders.append(f"Order: {order}, Total: {total}")
    print("Order placed successfully")
    
# ================================================================
# 3. Reporting module
# =============================================================

def sales_report():
    """
    Used to generate a sales report
    """
    if not completed_orders:
        print("No sales yet. \n")
        return
    print("\n----sales Report----")
    total_sales  = 0.0
    for i, order_records in enumerate(completed_orders, 1):
        print(f"\nOrder {i}:")
        order = order_records["order"]
        for pid, qty in order.items():
            name = inventory[pid]["name"]
            print(f"Name: {name} (ID: {pid}) ~ Quantity: {qty}")
        print(f"Order Total: ${order_records['total']:.2f}")
        total_sales += order_records["total"]
    print("\n--------------------------------")
    print(f"Total Sales:- ${total_sales:.2f}")
    print("\n--------------------------------")

def conditional_alerts():
    """
    function for alert byeond threshold
    """
    low_stock_found = False
    print("\n---------low stock alert---------")
    for pid, details in inventory.items():
        if details["stock"] <10:
            low_stock_found = True
            print(f"low stock alert: {details['name']}"
                  f"break(ID: {pid}) has less than 10 items left")
    if not low_stock_found:
        print("All products have sufficient stock. \n")


