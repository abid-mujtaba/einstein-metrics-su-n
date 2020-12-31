"""Test the wrapper Wedge class in the wedge module."""

from sympy import expand
from sympy.abc import a, b, x, y

import wedge as sut

from K_1_forms import K


def test_representation() -> None:
    """Test string representation of a WedgeProduct."""
    # GIVEN
    k_1 = K(1)
    k_2 = K(2)

    # WHEN
    wedge = sut.Wedge(k_1, k_2)

    # THEN
    assert str(wedge) == "(K₁ ∧ K₂)"


def test_zero_wedge_of_equal_1_forms() -> None:
    """Taking the wedge of a 1-form with itself results in 0."""
    # GIVEN
    k_1 = K(1)

    # WHEN
    wedge = sut.Wedge(k_1, k_1)

    # THEN
    assert wedge == 0


def test_zero_wedge_of_zero_left_operand() -> None:
    """Taking a wedge where one of the operands is zero should return a 0."""
    # GIVEN
    k_1 = K(1)

    # WHEN
    wedge = sut.Wedge(0, k_1)

    # THEN
    assert wedge == 0


def test_zero_wedge_of_zero_right_operand() -> None:
    """Taking a wedge where one of the operands is zero should return a 0."""
    # GIVEN
    k_1 = K(1)

    # WHEN
    wedge = sut.Wedge(k_1, 0)

    # THEN
    assert wedge == 0


def test_expand_K() -> None:
    """Test the expansion of K 1-form wedges over addition."""
    # GIVEN
    k_1 = K(1)
    k_2 = K(2)
    k_3 = K(3)
    k_4 = K(4)

    expr = sut.Wedge(k_1 + k_2, k_3 + k_4)

    expected_result = expand(
        sut.Wedge(k_1, k_3)
        + sut.Wedge(k_1, k_4)
        + sut.Wedge(k_2, k_3)
        + sut.Wedge(k_2, k_4)
    )

    # WHEN
    result = sut.expand_K(expr)

    # THEN
    assert result == expected_result


def test_extract_factor_K_left() -> None:
    """Test extraction of factor out of left operand of wedge over K 1-forms."""
    # GIVEN
    k_1 = K(1)
    k_2 = K(2)

    expr = sut.Wedge(a * k_1, k_2)

    # WHEN
    result = sut.extract_factor_K(expr)

    # THEN
    assert result == a * sut.Wedge(k_1, k_2)


def test_extract_factor_K_right() -> None:
    """Test extraction of factor out of right operand of wedge over K 1-forms."""
    # GIVEN
    k_1 = K(1)
    k_2 = K(2)

    expr = sut.Wedge(k_1, b * k_2)

    # WHEN
    result = sut.extract_factor_K(expr)

    # THEN
    assert result == b * sut.Wedge(k_1, k_2)


def test_extract_factor_K_both_sides() -> None:
    """Test extraction of factors out of both operands of wedge over K 1-forms."""
    # GIVEN
    k_1 = K(1)
    k_2 = K(2)

    expr = sut.Wedge(a * k_1, b * k_2)

    # WHEN
    result = sut.extract_factor_K(expr)

    # THEN
    assert result == a * b * sut.Wedge(k_1, k_2)


def test_antisymm_already_in_ascending_order() -> None:
    """No change when the Wedge is already in ascending order."""
    # GIVEN (k_1 < k_2)
    k_1 = K(1)
    k_2 = K(2)

    expr = sut.Wedge(k_1, k_2)

    # WHEN
    result = sut.antisymm(expr)

    # THEN
    assert result == expr


def test_antisymm_in_descending_order() -> None:
    """Flip order and sign when operands of the Wedge are in descending order."""
    # GIVEN (k_1 < k_2)
    k_1 = K(1)
    k_2 = K(2)

    expr = sut.Wedge(k_2, k_1)

    # WHEN
    result = sut.antisymm(expr)

    # THEN
    assert result == -1 * sut.Wedge(k_1, k_2)


def test_antisymm_complicated_expression() -> None:
    """Test antisymm on a slightly complicated expression to test recursion."""
    # GIVEN (k_1 < k_2)
    k_1 = K(1)
    k_2 = K(2)

    expr = a + b * sut.Wedge(k_2, k_1)

    # WHEN
    result = sut.antisymm(expr)

    # THEN
    assert result == a - b * sut.Wedge(k_1, k_2)


def test_extract_wedge_coeff_just_matching_wedge() -> None:
    """Extract coeff from an expression consisting only of a matching wedge."""
    # GIVEN
    b = 1
    c = 2

    expr = sut.Wedge(K(b), K(c))

    # WHEN
    result = sut.extract_wedge_coeff(expr, b, c)

    # THEN
    assert result == 1


def test_extract_wedge_coeff_just_not_matching_wedge() -> None:
    """Extract coeff from an expression consisting only of a non-matching wedge."""
    # GIVEN
    b = 1
    c = 2

    expr = sut.Wedge(K(b), K(c))

    # WHEN
    result = sut.extract_wedge_coeff(expr, 0, 1)

    # THEN
    assert result == 0


def test_extract_wedge_coeff_matching_wedge_with_multiplicative_factor() -> None:
    """Extract coeff from an expression consisting of a matching wedge with a factor."""
    # GIVEN
    b = 1
    c = 2

    expr = x * sut.Wedge(K(b), K(c))

    # WHEN
    result = sut.extract_wedge_coeff(expr, b, c)

    # THEN
    assert result == x


def test_extract_wedge_coeff_non_matching_wedge_with_multiplicative_factor() -> None:
    """Extract coeff from an expression consisting of a non-matching wedge with a factor."""
    # GIVEN
    b = 1
    c = 2

    expr = x * sut.Wedge(K(b), K(c))

    # WHEN
    result = sut.extract_wedge_coeff(expr, 0, 1)

    # THEN
    assert result == 0


def test_extract_wedge_coeff_matching_wedge_in_linear_combination() -> None:
    """Extract coeff from a linear combination containing a matching wedge."""
    # GIVEN
    b = 1
    c = 2

    expr = x * sut.Wedge(K(0), K(1)) + y * sut.Wedge(K(b), K(c))

    # WHEN
    result = sut.extract_wedge_coeff(expr, b, c)

    # THEN
    assert result == y


def test_extract_wedge_coeff_non_matching_wedge_in_linear_combination() -> None:
    """Extract coeff from a linear combination containing a non-matching wedge."""
    # GIVEN
    expr = x * sut.Wedge(K(0), K(1)) + y * sut.Wedge(K(1), K(2))

    # WHEN
    result = sut.extract_wedge_coeff(expr, 0, 2)

    # THEN
    assert result == 0
