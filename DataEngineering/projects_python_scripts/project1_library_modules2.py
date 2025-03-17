import datetime
import sys

# ---------------------- DATA STORES ----------------------
books = {}
borrowed_books = {}
transaction_log = []
STOCK_ALERT = 2
POPULAR_THRESHOLD = 3

# ---------------------- INPUT HANDLERS ----------------------
def get_input(prompt):
    """Universal input handler with menu navigation"""
    response = input(prompt).strip()
    return response.lower() if response.lower() == 'exit' else response

def validate_name(prompt):
    """Validate user name with exit handling"""
    while True:
        name = get_input(prompt)
        if name == 'exit': return 'exit'
        if not name: return None
        if name.isalpha(): return name
        print("❌ Invalid name. Use letters only.")

def validate_number(prompt, num_type):
    """Validate numbers with exit handling"""
    while True:
        value = get_input(prompt)
        if value == 'exit': return 'exit'
        if not value: return None
        try: return num_type(value)
        except: print(f"❌ Invalid {num_type.__name__} value")

# ---------------------- CORE FUNCTIONS ----------------------
def add_or_update_book():
    while True:
        print("\n📘 Book Management (type 'exit' to return)")
        user = validate_name("Your name (Enter to cancel): ")
        if user in ['exit', None]: return

        book_id = get_input("Book ID (Enter to cancel): ")
        if book_id in ['exit', '']: return

        if book_id in books:
            print(f"\nUpdating: {books[book_id]['title']}")
            print("1. Stock  2. Title  3. Author")
            print("4. Genre  5. Rent  6. Cancel")
            
            choice = get_input("Choose option: ")
            if choice == 'exit': return
            
            if choice == '1':
                copies = validate_number("Copies to add: ", int)
                if copies == 'exit': return
                if copies and copies > 0:
                    books[book_id]['copies'] += copies
                    status = "⚠️ Low stock!" if books[book_id]['copies'] <= STOCK_ALERT else ""
                    log_transaction('stock_update', user, book_id, copies, status)
            
            elif choice in ('2','3','4','5'):
                fields = {'2':'title','3':'author','4':'genre','5':'rent'}
                new_value = get_input(f"New {fields[choice]}: ")
                if new_value == 'exit': return
                books[book_id][fields[choice]] = float(new_value) if choice == '5' else new_value
                log_transaction('info_update', user, book_id, 0, f"Updated {fields[choice]}")
            
            elif choice == '6': return
        
        else:
            title = get_input("Title: ")
            if title == 'exit': return
            author = get_input("Author: ")
            if author == 'exit': return
            genre = get_input("Genre: ")
            if genre == 'exit': return
            
            rent = validate_number("Rent fee per copy: $", float)
            if rent == 'exit': return
            copies = validate_number("Initial copies: ", int)
            if copies == 'exit': return
            
            if None in [title, author, genre, rent, copies]:
                continue
                
            books[book_id] = {
                'title': title, 'author': author, 'genre': genre,
                'copies': copies, 'checkouts': 0, 'rent': rent
            }
            log_transaction('added', user, book_id, copies)
        
        get_input("\nPress Enter to continue...")

def display_books():
    if not books:
        print("\n📭 No books in inventory")
        get_input("\nPress Enter to return...")
        return
    
    print("\n📚 Current Inventory")
    print(f"{'ID':<10}{'Title':<25}{'Author':<20}{'Genre':<15}{'Stock':<10}{'Checkouts':<10}Rent")
    print("-"*95)
    for bid, b in books.items():
        stock = f"{b['copies']}{' ⚠️' if b['copies'] <= STOCK_ALERT else ''}"
        print(f"{bid:<10}{b['title'][:24]:<25}{b['author'][:19]:<20}"
              f"{b['genre'][:14]:<15}{stock:<10}{b['checkouts']:<10}${b['rent']:.2f}")
    get_input("\nPress Enter to return...")


def search_books():
    while True:
        print("\n🔍 Search Books (type 'exit' to return)")
        term = get_input("Search by title/author/genre: ").lower()
        if term == 'exit': return
        if not term: return
        
        results = []
        for bid, b in books.items():
            if (term in b['title'].lower() or
                term in b['author'].lower() or
                term in b['genre'].lower()):
                results.append((bid, b))
        
        if not results:
            print("\n❌ No matches found")
            continue
            
        print(f"\n📚 Found {len(results)} results:")
        print(f"{'ID':<10}{'Title':<25}{'Author':<20}{'Stock':<10}")
        print("-"*65)
        for bid, b in results:
            stock = f"{b['copies']}{' ⚠️' if b['copies'] <= STOCK_ALERT else ''}"
            print(f"{bid:<10}{b['title'][:24]:<25}{b['author'][:19]:<20}{stock:<10}")
        get_input("\nPress Enter to search again...")

# ---------------------- TRANSACTION FUNCTIONS ----------------------
def checkout_book():
    while True:
        print("\n📖 Checkout (type 'exit' to return)")
        user = validate_name("Your name: ")
        if user in ['exit', None]: return

        book_id = get_input("Book ID: ")
        if book_id == 'exit': return
        if book_id not in books:
            print("❌ Book not found")
            continue
            
        available = books[book_id]['copies']
        if available < 1:
            print("❌ Out of stock")
            continue
            
        copies = validate_number(f"Copies (max {available}): ", int)
        if copies == 'exit': return
        if not copies or copies < 1 or copies > available:
            print("❌ Invalid quantity")
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
        print(f"✔ Checked out {copies} copies. Due: {due_date}")
        get_input("\nPress Enter to continue...")


def return_book():
    while True:
        print("\n🔄 Return Books (type 'exit' to return)")
        user = validate_name("Your name: ")
        if user in ['exit', None] or user not in borrowed_books:
            print("❌ No borrowed books")
            return
            
        # Display borrowed books with numbers
        print("\n📚 Your Borrowed Books:")
        print(f"{'#':<3}{'Book ID':<10}{'Title':<25}{'Genre':<15}{'Copies':<10}")
        print("-"*63)
        for idx, item in enumerate(borrowed_books[user]):
            book = books[item['book_id']]
            print(f"{idx+1:<3}{item['book_id']:<10}{book['title'][:24]:<25}"
                  f"{book['genre'][:14]:<15}{item['copies']:<10}")
            
        # Get user input
        book_input = get_input("\nEnter BOOK ID or LIST NUMBER (or 'back'): ").strip().lower()
        if book_input in ['exit', 'back']:
            return
            
        # Selection handling
        try:
            if book_input.isdigit() and int(book_input) <= len(borrowed_books[user]):
                choice = int(book_input) - 1
                selected_item = borrowed_books[user][choice]
            else:
                # Find by Book ID
                selected_items = [item for item in borrowed_books[user] if item['book_id'] == book_input]
                if not selected_items:
                    print("❌ Book ID not found in your borrowings")
                    continue
                selected_item = selected_items[0]
                
            book_id = selected_item['book_id']
            max_return = selected_item['copies']
            
            # Get return details
            qty = validate_number(f"Copies to return (1-{max_return}): ", int)
            if not qty or qty < 1 or qty > max_return:
                print("❌ Invalid quantity")
                continue
                
            # Get return date
            return_date_str = get_input("Return date [YYYY-MM-DD] (Enter for today): ")
            return_date = datetime.date.today()
            if return_date_str:
                try:
                    return_date = datetime.datetime.strptime(return_date_str, "%Y-%m-%d").date()
                except ValueError:
                    print("⚠️ Invalid date format. Using today's date")
            
            # Update inventory
            books[book_id]['copies'] += qty
            selected_item['copies'] -= qty
            
            # Preserve history - don't delete entries
            if selected_item['copies'] == 0:
                borrowed_books[user].remove(selected_item)
            
            # Log transaction without deleting original entry
            log_transaction(
                'return', 
                user, 
                book_id, 
                qty,
                f"Returned: {return_date} | Remaining: {selected_item['copies']}"
            )
            
            print(f"\n✔ Successfully returned {qty} copies of {books[book_id]['title']}")
            if not borrowed_books[user]:
                del borrowed_books[user]
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            
        get_input("\nPress Enter to continue...")
        
# ---------------------- REPORTS ----------------------
def inventory_report():
    print("\n📊 Live Inventory Report")
    print(f"{'ID':<10}{'Title':<25}{'Stock':<10}{'Checkouts':<10}Status")
    print("-"*65)
    for bid, b in books.items():
        status = "⚠️ Low Stock" if b['copies'] <= STOCK_ALERT else "✔ Available"
        print(f"{bid:<10}{b['title'][:24]:<25}{b['copies']:<10}{b['checkouts']:<10}{status}")
    get_input("\nPress Enter to continue...")

def popular_books():
    popular = [(b['title'], b['checkouts']) for b in books.values() if b['checkouts'] >= POPULAR_THRESHOLD]
    if not popular:
        print("\n📉 No popular books currently")
        get_input("\nPress Enter to continue...")
        return
    
    print("\n🔥 Popular Books:")
    for title, count in sorted(popular, key=lambda x: x[1], reverse=True):
        print(f"- {title[:30]} ({count} checkouts)")
    get_input("\nPress Enter to continue...")


def transaction_summary():
    print("\n📜 Complete Transaction History")
    print(f"{'Date':<20}{'Type':<12}{'User':<15}{'Book ID':<10}{'Title':<25}{'Qty':<5}Details")
    print("-"*95)
    for t in transaction_log:
        book_title = books.get(t['book_id'], {}).get('title', 'Unknown')[:24]
        details = f"{t.get('notes','')}"
        
        print(f"{t['date'].strftime('%Y-%m-%d %H:%M'):<20}"
              f"{t['type']:<12}"
              f"{t['user']:<15}"
              f"{t['book_id']:<10}"
              f"{book_title:<25}"
              f"{t['copies']:<5}"
              f"{details}")
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