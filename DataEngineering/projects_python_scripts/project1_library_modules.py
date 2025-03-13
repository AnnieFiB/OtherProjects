import datetime

# Book Inventory Dictionary
books = {}  # Format: {book_id: {'title': ..., 'author': ..., 'genre': ..., 'copies': ..., 'checkout_count': 0}}

# Borrowed Books Dictionary (Stores user transactions)
borrowed_books = {}  # Format: {user_name: {'book_id': ..., 'due_date': ..., 'return_date': None, 'fine': 0}}

# Transaction Log (Records all transactions: added, checked out, returned)
transaction_log = []  # Format: [{'type': 'added/checkout/returned', 'user': ..., 'book_id': ..., 'date': ..., 'fine': ...}]


# ---------------- INVENTORY MANAGEMENT MODULE ----------------
def add_or_update_book():
    while True:
        book_id = input("Enter Book ID (or type 'exit' to cancel): ").strip()
        if book_id.lower() == "exit":
            print("Returning to the previous menu...")
            return
        
        title = input("Enter Book Title (or type 'exit' to cancel): ").strip()
        if title.lower() == "exit":
            print("Returning to the previous menu...")
            return

        author = input("Enter Author Name (or type 'exit' to cancel): ").strip()
        if author.lower() == "exit":
            print("Returning to the previous menu...")
            return

        genre = input("Enter Genre (or type 'exit' to cancel): ").strip()
        if genre.lower() == "exit":
            print("Returning to the previous menu...")
            return

        copies_input = input("Enter Number of Copies (or type 'exit' to cancel): ").strip()
        if copies_input.lower() == "exit":
            print("Returning to the previous menu...")
            return

        try:
            copies = int(copies_input)
            break
        except ValueError:
            print("Invalid number. Please enter an integer value.")

    if book_id in books:
        books[book_id]['copies'] += copies  # Update copies if book exists
        print(f"Updated book '{title}'. Total copies: {books[book_id]['copies']}")
    else:
        books[book_id] = {'title': title, 'author': author, 'genre': genre, 'copies': copies, 'checkout_count': 0}
        print(f"Book '{title}' added successfully!")

    # Log transaction
    transaction_log.append({'type': 'added', 'user': 'Library Admin', 'book_id': book_id, 'date': datetime.date.today(), 'fine': 0})

def display_books():
    if not books:
        print("No books available in inventory.")
        return

    print("\nAvailable Books:")
    print(f"{'ID':<10}{'Title':<30}{'Author':<25}{'Genre':<20}{'Copies':<10}{'Checkouts':<10}")
    print("-" * 105)

    for book_id, details in books.items():
        print(f"{book_id:<10}{details['title']:<30}{details['author']:<25}{details['genre']:<20}{details['copies']:<10}{details['checkout_count']:<10}")

    # Option to exit to menu
    input("\nPress Enter to return to the previous menu...")

def search_books():
    while True:
        search_term = input("\nEnter title, author, or genre to search (or type 'exit' to cancel): ").strip()
        if search_term.lower() == "exit":
            print("Returning to the previous menu...")
            return

        found_books = [details for book_id, details in books.items() if 
                       search_term.lower() in details['title'].lower() or 
                       search_term.lower() in details['author'].lower() or 
                       search_term.lower() in details['genre'].lower()]

        if found_books:
            print("\nSearch Results:")
            print(f"{'ID':<10}{'Title':<30}{'Author':<25}{'Genre':<20}{'Copies':<10}{'Checkouts':<10}")
            print("-" * 105)

            for book_id, book in books.items():
                if (search_term.lower() in book['title'].lower() or 
                    search_term.lower() in book['author'].lower() or 
                    search_term.lower() in book['genre'].lower()):
                    print(f"{book_id:<10}{book['title']:<30}{book['author']:<25}{book['genre']:<20}{book['copies']:<10}{book['checkout_count']:<10}")

            input("\nPress Enter to search again or type 'exit' to return: ")
        else:
            print("No books found matching the search criteria.")


# ---------------- ORDER PROCESSING MODULE ----------------
def checkout_book():
    while True:
        book_id = input("Enter Book ID to checkout (or type 'exit' to cancel): ").strip()
        if book_id.lower() == "exit":
            print("Returning to the previous menu...")
            return

        if book_id in books and books[book_id]["copies"] > 0:
            user = input("Enter your name (or type 'exit' to cancel): ").strip()
            if user.lower() == "exit":
                print("Returning to the previous menu...")
                return

            books[book_id]["copies"] -= 1  # Reduce available copies
            books[book_id]["checkout_count"] += 1  # Track popular books
            due_date = datetime.date.today() + datetime.timedelta(days=14)  # Due in 14 days
            borrowed_books[user] = {"book_id": book_id, "due_date": due_date, "return_date": None, "fine": 0}
            print(f"Book checked out successfully! Due Date: {due_date}")

            # Log transaction
            transaction_log.append({'type': 'checkout', 'user': user, 'book_id': book_id, 'date': datetime.date.today(), 'fine': 0})
            break
        else:
            print("Book not available or invalid book ID.")

def return_book():
    while True:
        user = input("Enter your name (or type 'exit' to cancel): ").strip()
        if user.lower() == "exit":
            print("Returning to the previous menu...")
            return

        if user in borrowed_books:
            book_id = borrowed_books[user]["book_id"]
            due_date = borrowed_books[user]["due_date"]

            # Prompt the user to choose return date method
            print("\nChoose return date option:")
            print("1. Use today's date")
            print("2. Enter a custom return date")
            
            date_choice = input("Enter your choice (1 or 2): ").strip()

            if date_choice == "1":
                return_date = datetime.date.today()
            elif date_choice == "2":
                while True:
                    return_date_str = input("Enter return date (YYYY-MM-DD) or type 'exit' to cancel: ").strip()
                    if return_date_str.lower() == "exit":
                        print("Returning to the previous menu...")
                        return
                    try:
                        return_date = datetime.datetime.strptime(return_date_str, "%Y-%m-%d").date()
                        break  # Exit loop on valid input
                    except ValueError:
                        print("Invalid date format. Please enter the date in YYYY-MM-DD format.")
            else:
                print("Invalid choice. Returning to the menu...")
                return

            books[book_id]["copies"] += 1
            borrowed_books[user]["return_date"] = return_date

            if return_date > due_date:
                overdue_days = (return_date - due_date).days
                fine = float(overdue_days * 2.0)
                borrowed_books[user]["fine"] = fine
                print(f"Book returned late! You have a fine of ${fine:.2f}.")
            else:
                print("Book returned successfully on time!")

            # Log transaction (but do NOT delete borrowed book record)
            transaction_log.append({'type': 'returned', 'user': user, 'book_id': book_id, 'date': return_date, 'fine': borrowed_books[user]['fine']})

            print(f"Book '{books[book_id]['title']}' returned on {return_date}.")
            print(f"Available copies: {books[book_id]['copies']}")
            break
        else:
            print("No record of borrowed books for this user.")

# ---------------- REPORTING MODULE ----------------
def inventory_report():
    if not books:
        print("No books available in inventory.")
        return
    
    print("\n Real-Time Inventory Report ")
    print(f"{'ID':<10}{'Title':<30}{'Author':<25}{'Genre':<20}{'Copies':<10}{'Checkouts':<10}{'Returned':<10}")
    print("-" * 115)

    total_books = 0
    total_checkouts = 0
    total_returns = 0
    genre_count = {}

    for book_id, details in books.items():
        # Count returned books from transaction log
        returned_count = sum(1 for t in transaction_log if t['book_id'] == book_id and t['type'] == 'returned')
        total_returns += returned_count

        print(f"{book_id:<10}{details['title']:<30}{details['author']:<25}{details['genre']:<20}{details['copies']:<10}{details['checkout_count']:<10}{returned_count:<10}")

        total_books += details['copies']
        total_checkouts += details['checkout_count']

        # Track genre-wise book count
        genre_count[details['genre']] = genre_count.get(details['genre'], 0) + details['copies']

    print("\n Summary ")
    print(f"Total Books in Inventory: {total_books}")
    print(f"Total Checkouts: {total_checkouts}")
    print(f"Total Returned Books: {total_returns}")

    print("\n Books by Genre ")
    for genre, count in genre_count.items():
        print(f"{genre:<20}{count} books")

    # Option to exit to menu
    input("\nPress Enter to return to the previous menu...")


def popular_books():
    threshold = 3  # Define popularity threshold (e.g., books checked out more than 3 times)

    # Filter books with checkout counts above the threshold
    popular = [book for book in books.values() if book.get('checkout_count', 0) > threshold]

    if not popular:
        print("\n No popular books at the moment! Try checking back later.")
        return

    print("\n Popular Books Alert!")
    print(f"{'Title':<30}{'Author':<25}{'Genre':<20}{'Checkouts':<10}")
    print("-" * 85)

    for book_id, details in books.items():
        if details.get('checkout_count', 0) > threshold:
            print(f"{details['title']:<30}{details['author']:<25}{details['genre']:<20}{details['checkout_count']:<10}")

    # Option to exit to menu
    input("\nPress Enter to return to the previous menu...")


def transaction_summary():
    if not transaction_log:
        print("No transactions recorded yet.")
        return

    print("\n Complete Transaction Summary ")
    print(f"{'Type':<12}{'User':<20}{'Book ID':<10}{'Date':<15}{'Fine':<10}")
    print("-" * 75)

    for transaction in transaction_log:
        # Ensure the date is properly formatted
        if isinstance(transaction['date'], datetime.date):
            transaction_date = transaction['date'].strftime("%Y-%m-%d")
        else:
            transaction_date = "Invalid Date"

        # Format fine properly as currency
        fine_amount = f"${transaction['fine']:.2f}" if transaction['fine'] > 0 else "No Fine"

        print(f"{transaction['type']:<12}{transaction['user']:<20}{transaction['book_id']:<10}{transaction_date:<15}{fine_amount:<10}")

    # Option to exit to menu
    input("\nPress Enter to return to the previous menu...")
