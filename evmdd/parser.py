#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This file is part of the EVMDD Library for Python (pyevmdd).
# Copyright (C) 2016 Robert Mattmüller

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
This input and parser module is responsible for translating arithmetic
function terms specified in Python syntax to |EVMDDs|.
"""

import ast
import networkx as nx
from numbers import Integral

from .evmdd import EvmddManager

_LEGAL_EXPRESSIONS = [ast.Expression, ast.Load, ast.BinOp,
                      ast.Add, ast.Mult, ast.Num, ast.Name,
                      ast.UnaryOp, ast.USub, ast.Sub, ast.Pow]

def _build_dependency_graph(function_term_ast):
    """Identify dependencies between variables in a given function.

    Args:
        `function_term_ast`: a function represented as an abstract syntax tree.

    Returns:
        a dependency graph over the variables in the given function.
    """
    G = nx.Graph()
    for v in collect_variables(function_term_ast):
        G.add_node(v)
    for node in ast.walk(function_term_ast):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            for vl in collect_variables(node.left):
                for vr in collect_variables(node.right):
                    if vl != vr:
                        G.add_edge(vl, vr)
    return G

def _connected_components(function_term_ast):
    """Determine connected components of dependency graph of function term.

    Args:
        `function_term_ast`: a function represented as an abstract syntax tree.

    Returns:
        a list of lists of variables, where each sublist is a connected component.
    """
    dependency_graph = _build_dependency_graph(function_term_ast)
    return list(nx.connected_components(dependency_graph))

def _default_automatic_variable_ordering(function_term_ast):
    """Determine default variable ordering for |EVMDD|.

    This is the ordering where all independent parts (corresponding to connected
    components of the dependency graph) are ordered in sequence. The internal
    orders of the independent parts and the ordering of the independent parts
    are unspecified.

    Args:
        `function_term_ast`: a function represented as an abstract syntax tree.

    Returns:
        a variable ordering specific to that function term.
    """
    connected_components = _connected_components(function_term_ast)
    variable_order = []
    for cc in sorted(connected_components, key=lambda cc: min([c for c in cc])):
        variable_order.extend(sorted(list(cc)))
    return variable_order

def read_function_term(function_term):
    """Read a function term and transform it into an AST.

    Args:
        `function_term` (string): a function term using only :math:`+`,
        :math:`-`, :math:`*`, and :math:`**`.

    Returns:
        `AST`: the abstract syntax tree representing function_term.
    """
    expression = ast.parse(function_term, mode='eval')
    for node in ast.walk(expression):
        assert type(node) in _LEGAL_EXPRESSIONS
    return expression

def collect_variables(function_term_ast):
    """Determine the set of all variables occurring in a given function term.

    Args:
        `function_term_ast`: a term represented as an abstract syntax tree.

    Returns:
        `set[string]`: the set of variables occurring in function_term_ast.
    """
    variables = set()
    for node in ast.walk(function_term_ast):
        if isinstance(node, ast.Name):
            name = node.id
            variables.add(name)
    return variables

class _SubstituteName(ast.NodeTransformer):
    """Substitute names with numbers in an abstract syntax tree.
    """
    def __init__(self, subst):
        self._subst = subst

    def visit_Name(self, node):
        if node.id in self._subst:
            return ast.Num(n=self._subst[node.id])
        return node

def _replace(node, subst):
    """Substitute names with numbers in an abstract syntax tree.
    """
    return _SubstituteName(subst).visit(node)

def _to_evmdd_rec(node, manager):
    """Translate a function term represented as an AST to the corresponding
    EVMDD recursively.

    Base cases: if the given term is a number or a variable, return the
    corresponding constant or variable |EVMDD|. Recursive cases: if the
    term is an addition, subtraction, multiplication, power to a natural
    exponent, or unary negation, recurse into the subexpressions, recursively
    translate them, and compose the sub-|EVMDDs| accordingly.

    Args:
        `node` (AST node): the abstract syntax tree node representing the
        arithmetic term to be translated into an |EVMDD|.

        `manager` (EvmddManager): the manager responsible for `node`.

    Returns:
        `Edge`: the corresponding |EVMDD|.
    """
    if isinstance(node, ast.Num):
        return manager.make_const_evmdd(int(node.n))
    elif isinstance(node, ast.Name):
        return manager.make_var_evmdd_for_var(node.id)
    elif isinstance(node, ast.BinOp):
        left = _to_evmdd_rec(node.left, manager)
        right = _to_evmdd_rec(node.right, manager)
        if isinstance(node.op, ast.Add):
            return left + right
        elif isinstance(node.op, ast.Sub):
            return left - right
        elif isinstance(node.op, ast.Mult):
            return left * right
        else:
            assert isinstance(node.op, ast.Pow)
            if ((not isinstance(node.right, ast.Num)) or
                (not isinstance(node.right.n, Integral)) or
                (node.right.n < 0)):
                raise ValueError("EVMDDs may only be raised to a nonnegative integral power.")
            return left ** node.right.n
    else:
        assert isinstance(node, ast.UnaryOp)
        assert isinstance(node.op, ast.USub)
        return -_to_evmdd_rec(node.operand, manager)

def term_to_evmdd(function_term, **kwargs):
    """Translate a function term to the corresponding |EVMDD|.

    The variable names in the desired variable ordering can be optionally
    specified. If no variable ordering is specified, the variable names are
    determined from the function term and ordered lexicographically.

    Also, the user may optionally specify the domain sizes of the variables.
    If no domain sized are specified, they default to 2 for all variables.

    Args:
        `function_term` (string): a function term in Python syntax using only
        constants, variables, addition, subtraction exponentiation to natural-
        numbered powers, and multiplication.

        \*\*\ `kwargs`: optionally, the variable names in the desired ordering,
        `var_names` (list of strings), their domain sizes `var_domains`
        (dict from strings to ints), and a flag `fully_reduced`
        determining whether the |EVMDD| should be fully reduced or
        quasi-reduced; `substitution` (dict from strings to numbers) fixing
        numbers for some of the variable names occurring in `function_term`.

    Returns:
        a tuple consisting of the corresponding |EVMDD| and its manager.

    The following example code constructs the fully reduced |EVMDD| for the
    running example :math:`f(A,B,C) = AB^2 + C + 2` for variable ordering
    :math:`A, B, C` and evaluates it for the valuation :math:`A=1`, :math:`B=2`,
    and :math:`C=0`.

    Example:
        >>> from .evmdd import evaluate
        >>> expr = 'A*B**2 + C + 2'
        >>> var_names = ['A', 'B', 'C']
        >>> var_domains = {'A': 2, 'B': 3, 'C': 2}
        >>> evmdd, manager = term_to_evmdd(
        ...                      expr, var_names=var_names,
        ...                      var_domains=var_domains, fully_reduced=True)
        >>> valuation = {'A': 1, 'B': 2, 'C': 0}
        >>> evaluate(evmdd, valuation, manager)
        6

    The next example shows that this works across a range of function terms,
    variable orderings, valuations, and both for fully and quasi-reduced
    |EVMDDs|.

    Example:
        >>> from itertools import permutations, product
        >>> from .evmdd import evaluate
        >>>
        >>> def collect_variables(function_term):
        ...     variables = set()
        ...     expression = ast.parse(function_term, mode='eval')
        ...     for node in ast.walk(expression):
        ...         if(type(node) == ast.Name):
        ...             name = node.id
        ...             variables.add(name)
        ...     return variables
        ...
        >>>
        >>> exprs = [
        ...     '0', '1',
        ...     'A', 'B', '0*A', '2*A', '0*B', '2*B',
        ...     'A+B', 'B+A', '1-A', '1-B',
        ...     '-A', 'A-B', 'B-A', '-(A+B)',
        ...     'A*B + B', 'B + A*B',
        ...     'A*B*B + C + 2', 'A*B - 17',
        ...     'A*B - A*B', 'A-A',
        ...     '1*(A+2)', 'A*(B+2)', 'A*(B+2)*(B+2) + C + 2',
        ... ]
        ...
        >>> domain_size = 4
        >>>
        >>> all_results_as_expected = True
        >>> for expr in exprs:
        ...     var_set = collect_variables(expr)
        ...     var_domains = {var: domain_size for var in var_set}
        ...     for fully_reduced in [True, False]:
        ...         for var_names in permutations(var_set):
        ...             for valuation in product(range(domain_size), repeat=len(var_set)):
        ...                 valuation = {var:val for var, val in zip(var_names,valuation)}
        ...                 evmdd, manager = term_to_evmdd(expr,
        ...                                  var_names=var_names, var_domains=var_domains,
        ...                                  fully_reduced=fully_reduced)
        ...                 actual = evaluate(evmdd, valuation, manager)
        ...                 expected = eval(expr, valuation)
        ...                 if actual != expected:
        ...                     all_results_as_expected = False
        ...
        >>> all_results_as_expected
        True
    """

    var_names = kwargs.get('var_names', None)
    var_domains = kwargs.get('var_domains', None)
    fully_reduced = kwargs.get('fully_reduced', True)
    substitution = kwargs.get('substitution', None)

    function_term_ast = read_function_term(function_term)
    if substitution:
        function_term_ast = _replace(function_term_ast, substitution)

    if not var_names:
        var_names = sorted(list(collect_variables(function_term_ast)))
    elif var_names == 'AUTOMATIC':
        var_names = _default_automatic_variable_ordering(function_term_ast)

    assert collect_variables(function_term_ast) <= set(var_names)

    if not var_domains:
        var_domains = {var: 2 for var in var_names}

    assert all([var in var_domains for var in var_names])
    var_domains = [var_domains[var] for var in var_names]

    manager = EvmddManager(var_names, var_domains, fully_reduced)

    assert isinstance(function_term_ast, ast.Expression)
    return _to_evmdd_rec(function_term_ast.body, manager), manager


def _test():
    import doctest
    doctest.testmod()


if __name__ == "__main__":
    _test()
