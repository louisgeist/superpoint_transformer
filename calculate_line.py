#!/usr/bin/env python3
"""
Script to calculate the equation of a line passing through two points
and find the y-value for a given x-value.
"""

def calculate_line_equation(x1, y1, x2, y2):
    """
    Calculate the equation of a line passing through two points.
    
    Args:
        x1, y1: Coordinates of the first point
        x2, y2: Coordinates of the second point
    
    Returns:
        tuple: (slope, intercept) where the line equation is y = slope * x + intercept
    """
    if x1 == x2:
        raise ValueError("The two points have the same x-coordinate. This is a vertical line (infinite slope).")
    
    # Calculate slope: a = (y2 - y1) / (x2 - x1)
    slope = (y2 - y1) / (x2 - x1)
    
    # Calculate intercept: b = y1 - a * x1
    intercept = y1 - slope * x1
    
    return slope, intercept


def get_y_from_x(slope, intercept, x):
    """
    Calculate y-value for a given x-value using the line equation.
    
    Args:
        slope: Slope of the line
        intercept: Y-intercept of the line
        x: X-value
    
    Returns:
        float: Y-value corresponding to x
    """
    return slope * x + intercept


def main():
    """Interactive function to use the line calculator."""
    print("Line Equation Calculator")
    print("=" * 40)
    
    # EZ-SP - min=5
    x1 = 19.7
    y1 = 97.0410
    z1 = 4086926.75
    
    # EZ-SP min 6
    x2 = 23.9
    y2 = 96.8775
    z2 = 4240279.5
    
    # PCP - min5
    x3=20.7
    y3=96.82962
    z3=3286898
    
    
    slope, intercept = calculate_line_equation(x1, y1, x2, y2)
    print(f"EZ-SP OmIoU at {x3} P/SP = {get_y_from_x(slope, intercept, x3):.2f}")
    print(f"PCP OmIoU at {x3} P/SP = {y3:.2f}")

    slope, intercept = calculate_line_equation(x1, z1, x2, z2)
    print(f"EZ-SP energy% at same P/SP as PCP = {get_y_from_x(slope, intercept, x3)/z3 * 100:.0f}%")


if __name__ == "__main__":
    # Example usage
    if len(__import__('sys').argv) > 1:
        # Command line mode: python calculate_line.py x1 y1 x2 y2 x3
        import sys
        if len(sys.argv) == 6:
            x1, y1, x2, y2, x3 = map(float, sys.argv[1:6])
            try:
                slope, intercept = calculate_line_equation(x1, y1, x2, y2)
                y3 = get_y_from_x(slope, intercept, x3)
                print(f"Line equation: y = {slope:.6f}x + {intercept:.6f}")
                print(f"For x = {x3}, y = {y3:.6f}")
            except ValueError as e:
                print(f"Error: {e}")
        else:
            print("Usage: python calculate_line.py x1 y1 x2 y2 x3")
            print("Example: python calculate_line.py 0 0 1 2 3")
    else:
        # Interactive mode
        main()
