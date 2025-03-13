import project1_library_modules as lm  # Import all modules

# ---------------- MAIN MENU ----------------
def main_menu():
    while True:
        print("\nLibrary Management System")
        print("1. Inventory Management: add, update, display or search for books")
        print("2. Order Processing: checkout or return books")
        print("3. Reporting: View all transanction summary")
        print("4. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            inventory_menu()
        elif choice == '2':
            order_menu()
        elif choice == '3':
            reporting_menu()
        elif choice == '4':
            print("Exiting Library Management System...")
            break
        else:
            print("Invalid choice! Please try again.")

# Inventory Menu
def inventory_menu():
    while True:
        print("\nInventory Management")
        print("1. Add or Update a Book")
        print("2. Display All Books")
        print("3. Search Books")
        print("4. Back to Main Menu")

        choice = input("Enter your choice: ")

        if choice == '1':
            lm.add_or_update_book()
        elif choice == '2':
            lm.display_books()
        elif choice == '3':
            lm.search_books()
        elif choice == '4':
            break

# Order Processing Menu
def order_menu():
    while True:
        print("\nOrder Processing")
        print("1. Checkout a Book")
        print("2. Return a Book")
        print("3. Back to Main Menu")

        choice = input("Enter your choice: ")

        if choice == '1':
            lm.checkout_book()
        elif choice == '2':
            lm.return_book()
        elif choice == '3':
            break

# Reporting Menu
def reporting_menu():
    while True:
        print("\nReporting")
        print("1. Real-Time Inventory Report")
        print("2. Popular Book Alerts")
        print("3. End-of-Day Transaction Summary")
        print("4. Back to Main Menu")

        choice = input("Enter your choice: ")

        if choice == '1':
            lm.inventory_report()
        elif choice == '2':
            lm.popular_books()
        elif choice == '3':
            lm.transaction_summary()
        elif choice == '4':
            break

# Run the Library Manager System
if __name__ == "__main__":
    main_menu()
