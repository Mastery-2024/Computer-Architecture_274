"""
Arithmetic Algorithms Collection
Supports signed and unsigned integers (simulated with two's complement) and provides:
1. Addition and subtraction
2. Sequential (shift-add) multiplication
3. Booth's multiplication
4. Bit-by-bit (schoolbook) multiplication
5. Bit-pair (Radix-4 / Modified Booth) multiplication
6. Division - Restoring method
7. Division - Non-restoring method

This file contains implementations that operate on integers but simulate fixed-width registers using two's complement
representation. Each algorithm has a function that returns (quotient_or_product, remainder_if_any, log_steps).

How to use:
- Import functions from this file or run as a script.
- Example: python Arithmetic_Algorithms.py --test all

Notes:
- For signed operations, pass signed=True. The code will use two's complement representation with the provided
  bit width (or auto computed minimal width + 1 for sign).
- For unsigned operations, pass signed=False.

Authors: Provided by ChatGPT (GPT-5 Thinking mini)
"""

from typing import Tuple, List
import argparse
import random

# ---------- Utilities ----------

def needed_bits(a: int, b: int, signed: bool) -> int:
    """Return the minimal bit width to represent operands a and b in two's complement if signed else unsigned."""
    if signed:
        # Need bits to represent sign and magnitude for both
        max_abs = max(abs(a), abs(b))
        bits = max(1, (max_abs.bit_length() + 1))  # +1 for sign
    else:
        max_val = max(a & ((1<<1024)-1), b & ((1<<1024)-1))
        bits = max(1, max_val.bit_length())
    return bits


def to_twos(val: int, bits: int) -> int:
    """Return integer that represents the two's complement bit pattern of val in 'bits' width."""
    mask = (1 << bits) - 1
    return val & mask


def from_twos(val_twos: int, bits: int, signed: bool) -> int:
    """Convert a two's complement integer value (val_twos) back to Python int."""
    if not signed:
        return val_twos
    sign_bit = 1 << (bits - 1)
    if val_twos & sign_bit:
        # negative
        return val_twos - (1 << bits)
    return val_twos


def bin_str(x: int, bits: int) -> str:
    return format(x & ((1<<bits)-1), '0{}b'.format(bits))

# ---------- 1. Addition and Subtraction ----------

def add(a: int, b: int, bits: int = None, signed: bool = True) -> Tuple[int, List[str]]:
    """Add a and b in 'bits' width. Returns (result_int, log_steps)."""
    if bits is None:
        bits = needed_bits(a, b, signed)
    A = to_twos(a, bits)
    B = to_twos(b, bits)
    result = (A + B) & ((1 << bits) - 1)
    steps = [f"bits={bits}", f"A (twos) = {bin_str(A,bits)}", f"B (twos) = {bin_str(B,bits)}", f"Sum = {bin_str(result,bits)}"]
    res_int = from_twos(result, bits, signed)
    # detect overflow for signed addition
    if signed:
        signA = (A >> (bits-1)) & 1
        signB = (B >> (bits-1)) & 1
        signR = (result >> (bits-1)) & 1
        if signA == signB and signA != signR:
            steps.append('Overflow occurred (signed)')
        else:
            steps.append('No signed overflow')
    return res_int, steps


def subtract(a: int, b: int, bits: int = None, signed: bool = True) -> Tuple[int, List[str]]:
    """Compute a - b via two's complement: a + (-b)."""
    if bits is None:
        bits = needed_bits(a, b, signed)
    neg_b = -b
    return add(a, neg_b, bits=bits, signed=signed)

# ---------- 2. Sequential (shift-add) multiplication ----------

def sequential_multiply(multiplicand: int, multiplier: int, bits: int = None, signed: bool = True) -> Tuple[int, List[str]]:
    """Shift-add multiplication simulating registers. Returns (product_int, steps).
    Product register width = 2*bits."""
    if bits is None:
        bits = needed_bits(multiplicand, multiplier, signed)
    width = bits
    M = to_twos(multiplicand, width)
    Q = to_twos(multiplier, width)
    A = 0
    steps = [f"width={width} (operand bits), product width={2*width}"]
    for i in range(width):
        steps.append(f"Step {i}: A={bin_str(A,2*width)} Q={bin_str(Q,width)} (lsb={Q & 1})")
        if (Q & 1) == 1:
            A = (A + (M << width)) & ((1 << (2*width)) - 1)
            steps.append(f"  Q lsb=1 -> A += M<<{width} => A={bin_str(A,2*width)}")
        # logical right shift of (A,Q) combined
        combined = ((A << width) | Q) & ((1 << (3*width)) - 1)
        combined >>= 1
        A = (combined >> width) & ((1 << (2*width)) - 1)
        Q = combined & ((1 << width) - 1)
        steps.append(f"  After shift -> A={bin_str(A,2*width)} Q={bin_str(Q,width)}")
    product_twos = ((A << width) | Q) & ((1 << (2*width)) - 1)
    steps.append(f"Final product (twos {2*width} bits) = {bin_str(product_twos,2*width)}")
    # convert back to integer
    product_int = from_twos(product_twos, 2*width, signed)
    return product_int, steps

# ---------- 3. Booth's Multiplication (standard Booth) ----------

def booths_multiplication(multiplicand: int, multiplier: int, bits: int = None, signed: bool = True) -> Tuple[int, List[str]]:
    if bits is None:
        bits = needed_bits(multiplicand, multiplier, signed)
    width = bits
    M = to_twos(multiplicand, width)
    Q = to_twos(multiplier, width)
    A = 0
    Q_1 = 0
    steps = [f"width={width}"]
    for i in range(width):
        q0 = Q & 1
        steps.append(f"Step {i}: A={bin_str(A, width+1)} Q={bin_str(Q,width)} Q_-1={Q_1}")
        if q0 == 1 and Q_1 == 0:
            # A = A - M
            # extend M to width+1 with sign
            M_ext = M if (M >> (width-1)) == 0 else (M | (~((1<<width)-1)))
            A = (A - (M_ext & ((1<<(width+1))-1))) & ((1<<(width+1))-1)
            steps.append(f"  Q0=1 Q-1=0 => A = A - M => {bin_str(A, width+1)}")
        elif q0 == 0 and Q_1 == 1:
            M_ext = M if (M >> (width-1)) == 0 else (M | (~((1<<width)-1)))
            A = (A + (M_ext & ((1<<(width+1))-1))) & ((1<<(width+1))-1)
            steps.append(f"  Q0=0 Q-1=1 => A = A + M => {bin_str(A, width+1)}")
        # arithmetic right shift of (A,Q,Q-1)
        combined = (A << (width+1)) | (Q << 1) | Q_1
        # arithmetic shift right by 1
        sign = (combined >> (width+1+width)) & 1
        combined = (combined >> 1) | (sign << (width+1+width-1))
        # extract back
        Q_1 = combined & 1
        Q = (combined >> 1) & ((1<<width)-1)
        A = (combined >> (width+1)) & ((1<<(width+1))-1)
        steps.append(f"  After arithmetic RShift -> A={bin_str(A,width+1)} Q={bin_str(Q,width)} Q_-1={Q_1}")
    product_twos = ((A << width) | Q) & ((1 << (2*width)) - 1)
    steps.append(f"Final product (twos {2*width} bits) = {bin_str(product_twos,2*width)}")
    product_int = from_twos(product_twos, 2*width, signed)
    return product_int, steps

# ---------- 4. Bit multiplication (schoolbook) ----------

def bit_by_bit_multiply(multiplicand: int, multiplier: int, bits: int = None, signed: bool = True) -> Tuple[int, List[str]]:
    if bits is None:
        bits = needed_bits(multiplicand, multiplier, signed)
    width = bits
    M = to_twos(multiplicand, width)
    Q = to_twos(multiplier, width)
    product = 0
    steps = [f"width={width}"]
    for i in range(width):
        bit = (Q >> i) & 1
        if bit:
            product = (product + (M << i)) & ((1 << (2*width)) - 1)
            steps.append(f"bit {i}=1 -> add M<<{i} => product={bin_str(product,2*width)}")
        else:
            steps.append(f"bit {i}=0 -> no add")
    product_int = from_twos(product, 2*width, signed)
    steps.append(f"Final product (twos) = {bin_str(product,2*width)}")
    return product_int, steps

# ---------- 5. Bit-pair multiplication (Radix-4 / Modified Booth) ----------

def bit_pair_multiply(multiplicand: int, multiplier: int, bits: int = None, signed: bool = True) -> Tuple[int, List[str]]:
    # Radix-4 Modified Booth: examine bit pairs with overlap
    if bits is None:
        bits = needed_bits(multiplicand, multiplier, signed)
    width = bits
    M = multiplicand
    Q = multiplier
    # We'll operate on signed integers here for recoding
    # Generate booth digits of multiplier
    m = Q
    digits = []  # each digit in {-2,-1,0,1,2}
    extended = m & ((1<< (width+1)) - 1)
    i = 0
    steps = [f"width={width}"]
    while i < width:
        b0 = (extended >> i) & 1
        b1 = (extended >> (i+1)) & 1
        b_1 = (extended >> (i-1)) & 1 if i-1 >= 0 else 0
        window = (b1 << 1) | b0
        # Standard encoding using triple (b_{i+1}, b_i, b_{i-1})
        trip = ((extended >> (i-1)) if i-1 >= 0 else (extended & 1)) & 0b111
        # Simpler rule: look at bits b_{i+1}, b_i, b_{i-1}
        # We'll compute digit d = (b_{i+1}*2 + b_i*1 + b_{i-1}* -1) simplified via known table
        # Easier: compute value = bits from i-1 to i+1
        val = ((extended >> max(0,i-1)) & 0b111)
        if val in (0,1):
            d = 0
        elif val in (2,3):
            d = 1
        elif val in (4,5):
            d = -1
        elif val in (6,7):
            d = 0  # essentially will be handled with carry
        else:
            d = 0
        digits.append(d)
        steps.append(f"i={i} val={val:03b} digit={d}")
        i += 2
    # Now compute product using digits
    product = 0
    for idx, d in enumerate(digits):
        shift = 2*idx
        if d == 1:
            product += (M << shift)
            steps.append(f"digit {idx}=+1 -> add M<<{shift}")
        elif d == -1:
            product -= (M << shift)
            steps.append(f"digit {idx}=-1 -> sub M<<{shift}")
        elif d == 2:
            product += (2 * M) << shift
            steps.append(f"digit {idx}=+2 -> add 2*M<<{shift}")
        elif d == -2:
            product -= (2 * M) << shift
            steps.append(f"digit {idx}=-2 -> sub 2*M<<{shift}")
        else:
            steps.append(f"digit {idx}=0 -> no op")
    mask = (1 << (2*width)) - 1
    product_twos = product & mask
    steps.append(f"Final product (twos {2*width} bits) = {bin_str(product_twos,2*width)}")
    product_int = from_twos(product_twos, 2*width, signed)
    return product_int, steps

# ---------- 6. Division - Restoring method ----------

def restoring_division(dividend: int, divisor: int, bits: int = None, signed: bool = True) -> Tuple[int,int,List[str]]:
    """Return (quotient, remainder, steps). Works for signed/unsigned by converting to positive and fixing sign."""
    if divisor == 0:
        raise ZeroDivisionError('Division by zero')
    if bits is None:
        bits = needed_bits(dividend, divisor, signed)
    width = bits
    # handle sign
    sign = 1
    A_div = dividend
    B_div = divisor
    if signed:
        if dividend < 0:
            A_div = -A_div
            sign *= -1
        if divisor < 0:
            B_div = -B_div
            sign *= -1
    # now unsigned algorithm on A_div, B_div
    A = 0
    Q = A_div
    M = B_div
    steps = [f"width={width}"]
    for i in range(width):
        # left shift (A,Q)
        A = ((A << 1) | ((Q >> (width-1)) & 1)) & ((1 << (width+1)) - 1)
        Q = ((Q << 1) & ((1 << width) - 1))
        steps.append(f"Step {i}: After shift A={bin_str(A,width+1)} Q={bin_str(Q,width)}")
        A = A - M
        steps.append(f"  A = A - M => {bin_str(A & ((1<<(width+1))-1), width+1)}")
        if A < 0:
            # restore
            A = A + M
            steps.append(f"  A < 0 -> restore A += M => {bin_str(A, width+1)} ; Q[0]=0")
            # Q0 is already 0
        else:
            # set Q0 = 1
            Q = Q | 1
            steps.append(f"  A >=0 -> set Q0=1 => Q={bin_str(Q,width)}")
    quotient = Q
    remainder = A
    if signed:
        quotient *= sign
        if dividend < 0:
            remainder = -remainder
    # mask to width
    return quotient, remainder, steps

# ---------- 7. Division - Non-Restoring method ----------

def nonrestoring_division(dividend: int, divisor: int, bits: int = None, signed: bool = True) -> Tuple[int,int,List[str]]:
    if divisor == 0:
        raise ZeroDivisionError('Division by zero')
    if bits is None:
        bits = needed_bits(dividend, divisor, signed)
    width = bits
    sign = 1
    A_div = dividend
    B_div = divisor
    if signed:
        if dividend < 0:
            A_div = -A_div
            sign *= -1
        if divisor < 0:
            B_div = -B_div
            sign *= -1
    A = 0
    Q = A_div
    M = B_div
    steps = [f"width={width}"]
    for i in range(width):
        # left shift (A,Q)
        A = ((A << 1) | ((Q >> (width-1)) & 1))
        Q = ((Q << 1) & ((1<<width)-1))
        steps.append(f"Step {i}: After shift A={A} Q={Q}")
        if A >= 0:
            A = A - M
            steps.append(f"  A>=0 -> A=A-M => {A}")
        else:
            A = A + M
            steps.append(f"  A<0 -> A=A+M => {A}")
        if A >= 0:
            Q = Q | 1
            steps.append(f"  A>=0 -> Q0=1 => Q={bin_str(Q,width)}")
        else:
            steps.append(f"  A<0 -> Q0=0 => Q={bin_str(Q,width)}")
    # final correction
    if A < 0:
        A = A + M
        steps.append(f"Final correction: A<0 -> A+=M => {A}")
    quotient = Q
    remainder = A
    if signed:
        quotient *= sign
        if dividend < 0:
            remainder = -remainder
    return quotient, remainder, steps

# ---------- Simple test harness ----------

def run_quick_tests():
    pairs = [ (7,3), (-7,3), (7,-3), (-7,-3), (15,2), (18,5) ]
    print('--- Addition/Subtraction tests ---')
    for a,b in pairs:
        res, log = add(a,b)
        print(f"{a} + {b} = {res} ; {log[-1]}")
        res, log = subtract(a,b)
        print(f"{a} - {b} = {res} ; {log[-1]}")
    print('\n--- Multiplication tests ---')
    for a,b in [(7,3),(-7,3),(7,-3),(-7,-3),(13,11)]:
        print(f"\nOperands: {a}, {b}")
        for fn in (sequential_multiply, booths_multiplication, bit_by_bit_multiply, bit_pair_multiply):
            val, log = fn(a,b)
            print(f"{fn.__name__}: {val}")
    print('\n--- Division tests ---')
    for a,b in [(20,3),(-20,3),(20,-3),(-20,-3)]:
        q,r,_ = restoring_division(a,b)
        print(f"restoring: {a}/{b} => q={q} r={r}")
        q,r,_ = nonrestoring_division(a,b)
        print(f"nonrestoring: {a}/{b} => q={q} r={r}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arithmetic algorithms collection')
    parser.add_argument('--test', choices=['quick','all'], default=None, help='run quick tests')
    args = parser.parse_args()
    if args.test:
        if args.test == 'quick':
            run_quick_tests()
        elif args.test == 'all':
            run_quick_tests()
    else:
        print('This module provides functions for addition, subtraction, several multiplication algorithms, and division (restoring/non-restoring).')
        print('Import it or run with --test quick')
