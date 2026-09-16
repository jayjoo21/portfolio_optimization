from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(
    PROJECT_ROOT / ".env"
)


from pykrx import stock


TEST_DATES = [
    "20180205",
    "20180615",
    "20200102",
    "20230102",
    "20250102",
    "20260915"
]


history = {}


for date in TEST_DATES:

    members = stock.get_index_portfolio_deposit_file(
        "5300",
        date
    )

    members = list(members)

    history[date] = set(members)

    print(
        "=" * 60
    )

    print(
        "Date:",
        date
    )

    print(
        "Count:",
        len(members)
    )

    print(
        "First 10:",
        members[:10]
    )


print(
    "\n"
    + "=" * 60
)

print(
    "CONSTITUENT CHANGE TEST"
)


for previous_date, current_date in zip(
    TEST_DATES[:-1],
    TEST_DATES[1:]
):

    previous = history[
        previous_date
    ]

    current = history[
        current_date
    ]

    added = current - previous
    removed = previous - current

    print(
        f"\n{previous_date} -> {current_date}"
    )

    print(
        "Added:",
        len(added)
    )

    print(
        sorted(added)[:10]
    )

    print(
        "Removed:",
        len(removed)
    )

    print(
        sorted(removed)[:10]
    )