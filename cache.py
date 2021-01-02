"""Cache the results of expensive calculations here to save time."""

from metric import x1, x2, x3


# A dictionary that maps the value of n to the 3 unique elements of the Ricci tensor
ricci = {
    2: [
        (x1 - x2 + x3) * (x1 + x2 - x3) / (2 * x2 * x3),
        -(x1 - x2 - x3) * (x1 + x2 - x3) / (2 * x1 * x3),
        -(x1 - x2 - x3) * (x1 - x2 + x3) / (2 * x1 * x2),
    ],
    3: [
        (
            8 * x1 ** 3 * x2
            + x1 ** 3 * x3
            - 8 * x1 * x2 ** 3
            + 21 * x1 * x2 ** 2 * x3
            - 16 * x1 * x2 * x3 ** 2
            - 2 * x2 ** 3 * x3
        )
        / (16 * x1 * x2 ** 2 * x3),
        -(
            8 * x1 ** 3 * x2
            + 2 * x1 ** 3 * x3
            - 21 * x1 ** 2 * x2 * x3
            - 8 * x1 * x2 ** 3
            + 16 * x1 * x2 * x3 ** 2
            - x2 ** 3 * x3
        )
        / (16 * x1 ** 2 * x2 * x3),
        -3 * (x1 ** 2 - 2 * x1 * x2 + x2 ** 2 - 2 * x3 ** 2) / (4 * x1 * x2),
    ],
    4: [
        (
            4 * x1 ** 3 * x2
            + x1 ** 3 * x3
            - 4 * x1 * x2 ** 3
            + 13 * x1 * x2 ** 2 * x3
            - 4 * x1 * x2 * x3 ** 2
            - 2 * x2 ** 3 * x3
        )
        / (8 * x1 * x2 ** 2 * x3),
        -(
            4 * x1 ** 3 * x2
            + 2 * x1 ** 3 * x3
            - 13 * x1 ** 2 * x2 * x3
            - 4 * x1 * x2 ** 3
            + 4 * x1 * x2 * x3 ** 2
            - x2 ** 3 * x3
        )
        / (8 * x1 ** 2 * x2 * x3),
        -(x1 - x2 - x3) * (x1 - x2 + x3) / (x1 * x2),
    ],
    5: [
        (
            8 * x1 ** 3 * x2
            + 3 * x1 ** 3 * x3
            - 8 * x1 * x2 ** 3
            + 31 * x1 * x2 ** 2 * x3
            - 8 * x1 * x2 * x3 ** 2
            - 6 * x2 ** 3 * x3
        )
        / (16 * x1 * x2 ** 2 * x3),
        -(
            8 * x1 ** 3 * x2
            + 6 * x1 ** 3 * x3
            - 31 * x1 ** 2 * x2 * x3
            - 8 * x1 * x2 ** 3
            + 8 * x1 * x2 * x3 ** 2
            - 3 * x2 ** 3 * x3
        )
        / (16 * x1 ** 2 * x2 * x3),
        -5 * (x1 - x2 - x3) * (x1 - x2 + x3) / (4 * x1 * x2),
    ],
}
