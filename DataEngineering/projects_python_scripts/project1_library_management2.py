from project1_library_modules2 import *

def reports_menu():
    while True:
        print("\n📈 Reports Menu")
        print("1. Inventory Report")
        print("2. Popular Books")
        print("3. Transaction History")
        print("4. Return to Main Menu")
        
        choice = get_input("Choose option: ")
        if choice in ['exit', '4']: return
        if choice == '1': inventory_report()
        elif choice == '2': popular_books()
        elif choice == '3': transaction_summary()
        else: print("❌ Invalid choice!")

def main_menu():
    while True:
        print("\n🏠 Main Menu")
        print("1. Add/Update Book")
        print("2. Display Books")
        print("3. Search Books")
        print("4. Checkout Book")
        print("5. Return Book")
        print("6. Reports")
        print("7. Exit Program")
        
        choice = get_input("Choose option: ")
        if choice in ['exit', '7']:
            print("\n🛑 Exiting program...")
            sys.exit()
        elif choice == '1': add_or_update_book()
        elif choice == '2': display_books()
        elif choice == '3': search_books()
        elif choice == '4': checkout_book()
        elif choice == '5': return_book()
        elif choice == '6': reports_menu()
        else: print("❌ Invalid choice!")

if __name__ == "__main__":
    main_menu()