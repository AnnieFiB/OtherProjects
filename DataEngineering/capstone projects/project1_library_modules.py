import datetime

# Book Inventory Dictionary
books = {}  # Format: {book_id: {'title': ..., 'author': ..., 'genre': ..., 'copies': ..., 'checkout_count': 0}}

# Borrowed Books Dictionary (Stores user transactions)
borrowed_books = {}  # Format: {user_name: {'book_id': ..., 'due_date': ..., 'fine': 0}}

# ---------------- INVENTORY MANAGEMENT MODULE ----------------
def add_or_update_book():
    book_id = input("Enter Book ID: ")
    title = input("Enter Book Title: ")
    author = input("Enter Author Name: ")
    genre = input("Enter Genre: ")
    copies = int(input("Enter Number of Copies: "))

    if book_id in books:
        books[book_id]['copies'] += copies  # Update copies if book exists
        print(f"Updated book '{title}'. Total copies: {books[book_id]['copies']}")
    else:
        books[book_id] = {'title': title, 'author': author, 'genre': genre, 'copies': copies, 'checkout_count': 0}
        print(f"Book '{title}' added successfully!")

def display_books():
    if not books:
        print("No books available in inventory.")
        return
    
    print("\nAvailable Books:")
    print(f"{'ID':<10}{'Title':<30}{'Author':<25}{'Genre':<20}{'Copies':<10}")
    print("-" * 95)
    for book_id, details in books.items():
        print(f"{book_id:<10}{details['title']:<30}{details['author']:<25}{details['genre']:<20}{details['copies']:<10}")

def search_books():
    search_term = input("Enter title, author, or genre to search: ").lower()
    found_books = [details for details in books.values() if 
                   search_term in details['title'].lower() or 
                   search_term in details['author'].lower() or 
                   search_term in details['genre'].lower()]

    if found_books:
        print("\nSearch Results:")
        print(f"{'Title':<30}{'Author':<25}{'Genre':<20}{'Copies':<10}")
        print("-" * 85)
        for book in found_books:
            print(f"{book['title']:<30}{book['author']:<25}{book['genre']:<20}{book['copies']:<10}")
    else:
        print("No books found matching the search criteria.")

# ---------------- ORDER PROCESSING MODULE ----------------
def checkout_book():
    book_id = input("Enter Book ID to checkout: ")
    
    if book_id in books and books[book_id]["copies"] > 0:
        user = input("Enter your name: ")
        books[book_id]["copies"] -= 1  # Reduce available copies
        books[book_id]["checkout_count"] += 1  # Track popular books
        due_date = datetime.date.today() + datetime.timedelta(days=14)  # Due in 14 days
        borrowed_books[user] = {"book_id": book_id, "due_date": due_date, "fine": 0}
        print(f"Book checked out successfully! Due Date: {due_date}")
    else:
        print("Book not available or invalid book ID.")

def return_book():
    user = input("Enter your name: ")

    if user in borrowed_books:
        book_id = borrowed_books[user]["book_id"]
        due_date = borrowed_books[user]["due_date"]
        return_date = datetime.date.today()
        books[book_id]["copies"] += 1  # Increase available copies

        # Calculate overdue fine
        if return_date > due_date:
            overdue_days = (return_date - due_date).days
            fine = float(overdue_days * 2.0)  # Fine = $2 per overdue day
            borrowed_books[user]["fine"] = fine
            print(f"Book returned late! You have a fine of ${fine:.2f}.")
        else:
            print("Book returned successfully on time!")

        del borrowed_books[user]  # Remove transaction record
    else:
        print("No record of borrowed books for this user.")

# ---------------- REPORTING MODULE ----------------
def inventory_report():
    if not books:
        print("No books available in inventory.")
        return
    
    print("\nReal-Time Inventory Report:")
    print(f"{'ID':<10}{'Title':<30}{'Copies Available':<20}")
    print("-" * 60)
    
    for book_id, details in books.items():
        print(f"{book_id:<10}{details['title']:<30}{details['copies']:<20}")

def popular_books():
    popular = [details for details in books.values() if details.get('checkout_count', 0) > 3]

    if not popular:
        print("No popular books at the moment.")
        return

    print("\nPopular Books (High Checkout Frequency):")
    print(f"{'Title':<30}{'Checkouts':<15}")
    print("-" * 50)
    
    for book in popular:
        print(f"{book['title']:<30}{book['checkout_count']:<15}")

def transaction_summary():
    if not borrowed_books:
        print("No transactions recorded for today.")
        return

    total_checkouts = len(borrowed_books)
    total_fines = sum(details["fine"] for details in borrowed_books.values())

    print("\nEnd-of-Day Transaction Summary:")
    print(f"Total Books Checked Out: {total_checkouts}")
    print(f"Total Outstanding Fines: ${total_fines:.2f}")
