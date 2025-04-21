import re
from sympy import symbols, Eq, solve, sympify


def solve_equation(equation_str):
    if equation_str == '':
        raise Exception("Empty equation string.")

    # Define the symbol (you can modify to detect more symbols if needed)
    x = symbols('x')

    # Split the equation string at '='
    left_expr, right_expr = equation_str.split('=')

    # Convert strings to sympy expressions
    left = sympify(left_expr)
    right = sympify(right_expr)

    # Create the equation
    equation = Eq(left, right)

    # Solve the equation
    solutions = solve(equation, x)

    return solutions


def process_equation(parsed_equation: list[str]):
    equation_str = ""
    for char in parsed_equation:
        equation_str += char

    # '=' is split when segmenting the image so
    # replacing '--' with '='
    equation_str = equation_str.replace("--", "=")
    # formatting equation for sympy
    equation_str = equation_str.replace('x', "*x")
    # replace x{num} with x^{num}; eg: x2 -> x^2
    equation_str = re.sub(r'x(\d+)', lambda match: f'x^{
                          match.group(1)}', equation_str)

    # Remove any characters that shouldn't be there
    equation_str = re.sub(r'[^0-9x+\-\^*/=]', '', equation_str)
    return equation_str


def main():
    equation_str = "2x2-8=0"
    equation_str = process_equation(equation_str)
    print(equation_str)
    solution = solve_equation(equation_str)
    print(solution)


if __name__ == "__main__":
    main()
