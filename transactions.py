"""
transactions.py
Synthetic transaction generator for NudgeVest.
Generates realistic financial transactions with category, amount, day-of-week, and hour.
"""

import random
from typing import Dict, Any, List, Optional

CATEGORIES = ["coffee", "food", "transport", "shopping", "entertainment"]

CATEGORY_PRICE_RANGES = {
    "coffee": (2.75, 9.50),
    "food": (8.00, 52.00),
    "transport": (2.75, 38.00),
    "shopping": (18.00, 175.00),
    "entertainment": (12.00, 95.00),
}

CATEGORY_MERCHANTS = {
    "coffee": ["Blue Bottle Coffee", "Starbucks Reserve", "Local Artisan Cafe", "Dunkin'", "Peet's Coffee"],
    "food": ["Sweetgreen", "Chipotle", "Whole Foods Market", "Trader Joe's", "Corner Deli", "Bistro Bella"],
    "transport": ["Uber Ride", "Lyft Standard", "City Metro Transit", "BART Fare", "Yellow Cab Co."],
    "shopping": ["Target", "Amazon Order", "Uniqlo", "Zara", "Nordstrom Rack", "Best Buy Express"],
    "entertainment": ["AMC Theatres", "Spotify Family Sub", "Steam Game Store", "Concert Ticket", "Bowling Alley"],
}

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def get_amount_bucket(amount: float) -> str:
    """Categorizes the transaction amount into 'low', 'med', or 'high'."""
    if amount < 15.0:
        return "low"
    elif amount <= 50.0:
        return "med"
    else:
        return "high"


def calculate_round_up(amount: float) -> float:
    """
    Calculates round-up to the next dollar.
    If amount is an exact dollar, rounds up by $1.00.
    """
    cents = round(amount % 1.0, 2)
    if cents == 0.0:
        return 1.00
    return round(1.0 - cents, 2)


def generate_transaction(
    tick: Optional[int] = None,
    day_of_week: Optional[int] = None,
    hour: Optional[int] = None,
    category: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generates a single synthetic transaction for a given tick.
    
    Returns a feature dictionary containing:
      - tick: simulation step index
      - category: transaction category ('coffee', 'food', 'transport', 'shopping', 'entertainment')
      - amount: float transaction cost
      - amount_bucket: 'low' (<15), 'med' (15-50), 'high' (>50)
      - day_of_week: int 0 (Mon) to 6 (Sun)
      - day_name: 3-letter abbreviation
      - hour: int 0 to 23
      - merchant: descriptive merchant name
      - round_up_amount: spare change calculated to next dollar
    """
    # Category selection: slightly biased towards frequent daily items like coffee and food
    if category is None:
        category = random.choices(
            CATEGORIES,
            weights=[0.30, 0.30, 0.18, 0.12, 0.10],
            k=1
        )[0]
    
    # Category appropriate price range
    min_price, max_price = CATEGORY_PRICE_RANGES[category]
    raw_amount = random.uniform(min_price, max_price)
    amount = round(raw_amount, 2)

    # Time context
    if day_of_week is None:
        day_of_week = random.randint(0, 6)
    if hour is None:
        # Realistic hour generation based on category
        if category == "coffee":
            hour = random.choice([7, 8, 9, 10, 11, 14, 15])
        elif category == "food":
            hour = random.choice([11, 12, 13, 14, 18, 19, 20, 21])
        elif category == "entertainment":
            hour = random.choice([18, 19, 20, 21, 22, 23])
        elif category == "transport":
            hour = random.choice([8, 9, 17, 18, 19, 22])
        else:
            hour = random.randint(10, 21)

    merchant = random.choice(CATEGORY_MERCHANTS[category])
    round_up = calculate_round_up(amount)
    amount_bucket = get_amount_bucket(amount)

    return {
        "tick": tick if tick is not None else 0,
        "category": category,
        "amount": amount,
        "amount_bucket": amount_bucket,
        "day_of_week": int(day_of_week),
        "day_name": DAY_NAMES[day_of_week],
        "hour": int(hour),
        "merchant": merchant,
        "round_up_amount": round_up,
    }


def generate_batch_transactions(n: int) -> List[Dict[str, Any]]:
    """Generates a list of n synthetic transactions across consecutive ticks."""
    transactions = []
    current_day = random.randint(0, 6)
    current_hour = random.randint(7, 10)

    for i in range(n):
        tx = generate_transaction(tick=i + 1, day_of_week=current_day, hour=current_hour)
        transactions.append(tx)
        
        # Advance clock slightly for successive ticks
        current_hour += random.randint(1, 4)
        if current_hour >= 24:
            current_hour %= 24
            current_day = (current_day + 1) % 7

    return transactions
