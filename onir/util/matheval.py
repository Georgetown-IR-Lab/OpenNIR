# Evaluates basic math expressions without using eval()
# Based on https://stackoverflow.com/questions/2371436#answer-9558001

import ast
import operator as op

# supported operators
operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
             ast.Div: op.truediv, ast.Pow: op.pow, ast.BitXor: op.xor,
             ast.USub: op.neg, ast.Eq: op.eq, ast.NotEq: op.ne}

def matheval(expr, numtype=float):
    """
    >>> matheval('2^6')
    4
    >>> matheval('2**6')
    64
    >>> matheval('1 + 2*3**(4^5) / (6 + -7)')
    -5.0
    """
    return eval_(ast.parse(expr, mode='eval').body, numtype)

def eval_(node, numtype):
    if isinstance(node, ast.Num): # <number>
        return numtype(str(node.n))
    if isinstance(node, ast.BinOp): # <left> <operator> <right>
        return operators[type(node.op)](eval_(node.left, numtype), eval_(node.right, numtype))
    if isinstance(node, ast.Compare): # <left> <operator> <right>
        assert len(node.ops) == 1 and len(node.comparators) == 1
        return operators[type(node.ops[0])](eval_(node.left, numtype), eval_(node.comparators[0], numtype))
    if isinstance(node, ast.UnaryOp): # <operator> <operand> e.g., -1
        return operators[type(node.op)](eval_(node.operand, numtype))
    raise TypeError(node)
