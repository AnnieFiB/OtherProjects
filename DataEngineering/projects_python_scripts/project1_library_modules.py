# project1_library_modules2.py
import datetime
import sys

# ---------------------- DATA STORES ----------------------
books = {}
borrowed_books = {}
transaction_log = []
STOCK_ALERT = 2
POPULAR_THRESHOLD = 3

# ---------------------- HELPER FUNCTIONS ----------------------
def get_input(prompt):
    """Universal input handler with exit capability"""
    response = input(prompt).strip()
    if response.lower() == 'exit':
        print("\n🚪 Exiting program...")
        sys.exit(0)
    return response

def validate_name(prompt):
    """Validate user name input"""
    while True:
        name = get_input(prompt)
        if not name: return None
        if name.isalpha(): return name
        print("❌ Invalid name. Use letters only.")

def validate_number(prompt, num_type):
    """Validate numeric inputs"""
    while True:
        value = get_input(prompt)
        if not value: return None
        try:
            return num_type(value)
        except ValueError:
            print(f"Invalid {num_type.__name__} value")

# ---------------------- INVENTORY MANAGEMENT ----------------------
def add_or_update_book():
    while True:
        print("\n📘 Book Management (type 'exit' to quit)")
        user = validate_name("Your name (Enter to cancel): ")
        if not user: return

        book_id = get_input("Book ID (Enter to cancel): ")
        if not book_id: return

        if book_id in books:
            print(f"Updating {books[book_id]['title']}")
            print("1.Stock 2.Title 3.Author 4.Genre 5.Rent 6.Cancel")
            choice = get_input("Choose: ")
            
            if choice == '1':
                copies = validate_number("Copies to add: ", int)
                if copies:
                    books[book_id]['copies'] += copies
                    note = "Low stock!" if books[book_id]['copies'] <= STOCK_ALERT else ""
                    log_transaction('updated', user, book_id, copies, note)
        else:
            title = get_input("Title: ")
            author = get_input("Author: ")
            genre = get_input("Genre: ")
            rent = validate_number("Rent fee: $", float)
            copies = validate_number("Copies: ", int)
            
            if all([title, author, genre, rent, copies]):
                books[book_id] = {
                    'title': title, 'author': author, 'genre': genre,
                    'copies': copies, 'checkouts': 0, 'rent': rent
                }
                log_transaction('added', user, book_id, copies)
        
        get_input("\nPress Enter to continue...")

def display_books():
    if not books:
        print("\nNo books in inventory")
        get_input("\nPress Enter to return...")
        return
    
    print("\n📚 Current Inventory")
    print(f"{'ID':<10}{'Title':<25}{'Author':<20}{'Stock':<10}{'Checkouts':<10}Rent")
    print("-"*85)
    for bid, b in books.items():
        stock = f"{b['copies']}{'⚠️' if b['copies'] <= STOCK_ALERT else ''}"
        print(f"{bid:<10}{b['title'][:24]:<25}{b['author'][:19]:<20}"
              f"{stock:<10}{b['checkouts']:<10}${b['rent']:.2f}")
    get_input("\nPress Enter to return...")

def search_books():
    while True:
        term = get_input("\n🔍 Search (title/author/genre) or Enter to cancel: ").lower()
        if not term: return
        
        results = []
        for bid, b in books.items():
            if (term in b['title'].lower() or
                term in b['author'].lower() or
                term in b['genre'].lower()):
                results.append((bid, b))
        
        if not results:
            print("No matches found")
            continue
            
        print(f"\nFound {len(results)} results:")
        print(f"{'ID':<10}{'Title':<25}{'Author':<20}{'Stock':<10}")
        print("-"*65)
        for bid, b in results:
            print(f"{bid:<10}{b['title'][:24]:<25}{b['author'][:19]:<20}{b['copies']:<10}")
        get_input("\nPress Enter to search again...")

# ---------------------- ORDER PROCESSING ----------------------
def checkout_book():
    while True:
        print("\n📖 Checkout (type 'exit' to quit)")
        user = validate_name("Your name: ")
        if not user: return

        book_id = get_input("Book ID: ")
        if book_id not in books:
            print("Book not found")
            continue
            
        available = books[book_id]['copies']
        if available < 1:
            print("Out of stock")
            continue
            
        copies = validate_number(f"Copies (max {available}): ", int)
        if not copies or copies < 1 or copies > available:
            print("Invalid quantity")
            continue
            
        books[book_id]['copies'] -= copies
        books[book_id]['checkouts'] += copies
        due_date = datetime.date.today() + datetime.timedelta(days=14)
        
        if user not in borrowed_books:
            borrowed_books[user] = []
        borrowed_books[user].append({
            'book_id': book_id,
            'due_date': due_date,
            'copies': copies,
            'rent': books[book_id]['rent'] * copies
        })
        
        log_transaction('checkout', user, book_id, copies, f"Due: {due_date}")
        print(f"Checked out {copies} copies. Due: {due_date}")
        get_input("\nPress Enter to continue...")

def return_book():
    while True:
        print("\n🔄 Return (type 'exit' to quit)")
        user = validate_name("Your name: ")
        if not user or user not in borrowed_books:
            print("No borrowed books")
            return
            
        print("\nYour Borrowed Books:")
        for idx, item in enumerate(borrowed_books[user]):
            book = books[item['book_id']]
            print(f"{idx+1}. {book['title']} ({item['copies']} copies)")
            
        choice = validate_number("Select book (number): ", int)
        if not choice or choice < 1 or choice > len(borrowed_books[user]):
            print("Invalid selection")
            continue
            
        item = borrowed_books[user][choice-1]
        qty = validate_number(f"Return quantity (max {item['copies']}): ", int)
        if not qty or qty < 1 or qty > item['copies']:
            print("Invalid quantity")
            continue
            
        # Calculate fine
        return_date = datetime.date.today()
        overdue_days = (return_date - item['due_date']).days
        fine = max(0, overdue_days) * 2.0 * qty
        
        # Update records
        books[item['book_id']]['copies'] += qty
        item['copies'] -= qty
        note = f"Fine: ${fine:.2f}" if fine else ""
        
        if item['copies'] == 0:
            borrowed_books[user].pop(choice-1)
            if not borrowed_books[user]:
                del borrowed_books[user]
        
        log_transaction('returned', user, item['book_id'], qty, note)
        print(f"Returned {qty} copies. {note}")
        get_input("\nPress Enter to continue...")

# ---------------------- REPORTING ----------------------
def inventory_report():
    print("\n📊 Live Inventory Report")
    print(f"{'ID':<10}{'Title':<25}{'Stock':<10}{'Checkouts':<10}Status")
    print("-"*65)
    for bid, b in books.items():
        status = "Low Stock" if b['copies'] <= STOCK_ALERT else "Available"
        print(f"{bid:<10}{b['title'][:24]:<25}{b['copies']:<10}{b['checkouts']:<10}{status}")
    get_input("\nPress Enter to continue...")

def popular_books():
    popular = [(b['title'], b['checkouts']) for b in books.values() if b['checkouts'] >= POPULAR_THRESHOLD]
    if not popular:
        print("\nNo popular books currently")
        return
    
    print("\n🔥 Popular Books:")
    for title, count in sorted(popular, key=lambda x: x[1], reverse=True):
        print(f"{title:<30} ({count} checkouts)")
    get_input("\nPress Enter to continue...")

def transaction_summary():
    print("\n📜 Complete Transaction History")
    print(f"{'Date':<20}{'Type':<10}{'User':<15}{'Book':<25}{'Qty':<5}Details")
    print("-"*85)
    for t in transaction_log:
        book_title = books.get(t['book_id'], {}).get('title', 'Unknown')[:24]
        print(f"{t['date'].strftime('%Y-%m-%d %H:%M'):<20}"
              f"{t['type']:<10}{t['user']:<15}{book_title:<25}"
              f"{t['copies']:<5}{t.get('notes','')}")
    get_input("\nPress Enter to continue...")

def log_transaction(t_type, user, book_id, copies, notes=""):
    transaction_log.append({
        'type': t_type,
        'user': user,
        'book_id': book_id,
        'date': datetime.datetime.now(),
        'copies': copies,
        'notes': notes
    })