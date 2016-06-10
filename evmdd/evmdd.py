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
This core module is responsible for the internal representation of |EVMDDs|
via Nodes and Edges, for arithmetic operations on them, and for the computation
of function values.

It supports quasi-reduced and fully reduced |EVMDDs|, but does not allow mixing
them. Many functions and methods have an optional boolean parameter
``fully_reduced`` or ``is_fully_reduced`` that determines whether they are
supposed to deal with fully reduced (if true) or quasi-reduced (if false)
|EVMDDs|.
"""

import logging
from numbers import Integral

from .util import memoize, EqualityMixin

_DEFAULT_IS_FULLY_REDUCED = True

def _make_sink_node(is_fully_reduced=_DEFAULT_IS_FULLY_REDUCED):
    """Create the unique 0-sink node.

    Since uniqueness is only enforced among |EVMDDs| of the same type,
    i.e., fully reduced or quasi-reduced, and we do not allow mixing
    these types, there will generally be two 0-sink nodes, one for these
    fully reduced case, and one for the quasi-reduced case.

    The requested type is specified by the argument `is_fully_reduced`.
    """
    return Node(level=0, children=[], is_fully_reduced=is_fully_reduced)

def _make_const_evmdd(number, is_fully_reduced=_DEFAULT_IS_FULLY_REDUCED):
    """Construct an |EVMDD| representing a given constant number.

    This is the |EVMDD| with a single edge immediately leading to the sink
    node, labeled with the given constant as its edge weight.
    """
    return Edge(weight=number, succ=_make_sink_node(is_fully_reduced),
                is_fully_reduced=is_fully_reduced)

def _aggregate_weights(weight1, weight2, oper):
    """Aggregate weights according to operator.

    If `oper` is addition/subtraction/multiplication,
    then add/subtract/multiply, respectively. Here, arithmetic operators
    on edges are mapped to the corresponding arithmetic operators on numbers.
    """
    if oper is Edge.__add__:
        return weight1 + weight2
    elif oper is Edge.__sub__:
        return weight1 - weight2
    else:
        assert oper is Edge.__mul__
        return weight1 * weight2

def _is_terminal_case(edge1, edge2, oper):
    """Test if `edge1` and `edge2` can be combined with operator `oper`
    without the need to recursively traverse either of the two |EVMDDs|.
    """
    edge1_is_sink = edge1.succ.is_sink_node()
    edge2_is_sink = edge2.succ.is_sink_node()

    if oper is Edge.__add__ or oper is Edge.__mul__:
        return edge1_is_sink or edge2_is_sink
    else:
        assert oper is Edge.__sub__
        return edge2_is_sink

def _terminal_value(edge1, edge2, oper, is_fully_reduced):
    """Compute the new |EVMDD| for `edge1 oper edge2` in the terminal case
    where no recursion is needed.

    For addition and subtraction, the constant weight of the constant summand
    can be added to (subtracted from) the edge weight of the other edge, and
    the subgraphs below the other edge can be preserved.

    For multiplication, both constants are multiplied and the result is used
    as the weight of a resulting new `Edge` to the sink node.

    All cases are non-destructive, i.e., the old argument |EVMDDs| are
    not modified. Rather, a new |EVMDD| is constructed or, if possible,
    retrieved from a lookup table to avoid duplicates.
    """
    assert _is_terminal_case(edge1, edge2, oper)
    if edge1.succ.is_sink_node():
        sink_edge, potential_non_sink_edge = edge1, edge2
    else:
        sink_edge, potential_non_sink_edge = edge2, edge1

    result_weight = _aggregate_weights(edge1.weight, edge2.weight, oper)

    if oper is Edge.__add__ or oper is Edge.__sub__:
        result_succ = potential_non_sink_edge.succ
    else:
        assert oper is Edge.__mul__
        if potential_non_sink_edge.succ.is_sink_node():
            result_succ = potential_non_sink_edge.succ
        else:
            succ = potential_non_sink_edge.succ
            scaled_children = tuple([sink_edge * child for child in succ.children])

            result_succ = Node(level=succ.level,
                               children=scaled_children,
                               is_fully_reduced=is_fully_reduced)

            if is_fully_reduced:
                result_succ = _perform_shannon_reduction(result_succ)

    return Edge(weight=result_weight, succ=result_succ, is_fully_reduced=is_fully_reduced)

def _determine_children_on_same_level(edge1, edge2):
    """In case one of the |EVMDDs| to which an arithmetic operation is
    applied is not quasi-reduced, it can happen that the two top-most nodes
    of the two |EVMDDs| are on different levels (= represent different
    variables). If this happens, they need to be synchronized. This is done
    by implicitly locally quasi-reducing the |EVMDD| with a skipped
    variable. This means that for the skipped variable, as many copies of
    the sub- |EVMDD| are created as the domain size of the skipped variable
    requires. Then, we can call `apply` recursively.

    If we have to locally quasi-reduce an |EVMDD|, then we have to set the
    weight of the new edges to zero.
    """
    if edge1.succ.level >= edge2.succ.level:
        modified_weight_children = [
                Edge(weight=child.weight+edge1.weight,
                     succ=child.succ,
                     is_fully_reduced=child.is_fully_reduced) for child in edge1.succ.children]
        return modified_weight_children
    else:
        return len(edge2.succ.children) * tuple([edge1])

def _log_apply(edge1, edge2, oper, result, terminal):
    """Log result of operator application to two edges.

    Args:
        `edge1` (Edge): first edge.

        `edge2` (Edge): second edge.

        `oper`: operator.

        `result` (Edge): resulting edge.

        `terminal` (bool): true iff this was a terminal application,
            and false iff this was a recursive application.
    """
    if terminal:
        place = 'terminal'
    else:
        place = 'recursive'
    logging.debug(('%s: applying %s to\n' % (place, oper)) +
                  ('    %s and\n' % repr(edge1)) +
                  ('    %s results in\n' % repr(edge2)) +
                  ('    %s\n' % repr(result)))

def _perform_shannon_reduction(succ):
    """Perform Shannon reduction while combining two |EVMDDs| into one.

    If all outgoing edges (`children`) from the next node (`succ`) reached via
    the edge in question carry the same weight (`first_child_weight`) and lead
    to the same successor node (`first_child_succ`), then we can skip `succ`
    and immediately go to `first_child_succ`. The implementation assumes that
    successor weights are already normalized, i.e., if all children carry the
    same weight, this is zero for all of them.
    """
    children = succ.children
    first_child_weight = children[0].weight
    first_child_succ = children[0].succ
    if (all([child.weight == first_child_weight for child in children]) and
            all([child.succ == first_child_succ for child in children])):
        assert first_child_weight == 0
        succ_after_reduction = first_child_succ
    else:
        succ_after_reduction = succ
    return succ_after_reduction

@memoize
class Edge(EqualityMixin):
    """An edge in an |EVMDD|, specifying weight and successor node.

    The `weight` of an edge `e` is the partial function value associated with
    the variable-value pair corresponding to `e` (conditional on variable-value
    pairs at higher levels).

    The successor `succ` of `e` is the next node reached by `e`, either a test
    node for the next variable, or the sink node.

    Additionally, each edge `e` has a flag `is_fully_reduced` that denotes
    whether the |EVMDD| represented by this edge is fully reduced or only
    quasi-reduced. By default, |EVMDDs| are fully reduced.

    The "dangling incoming edge" from the literature is also represented by an
    object of this class that is required to have exactly one child node in the
    collection `succ`, i.e., ``len(succ) == 1`` must hold for this edge. Since
    |EVMDDs| can be identified by their dangling incoming edge, we do not need
    a separate `EVMDD` class, but rather use `Edges` to represent |EVMDDs|.
    """

    def __init__(self, weight, succ, is_fully_reduced=_DEFAULT_IS_FULLY_REDUCED):
        """Initialize an `Edge` with weight and successor node."""
        self.weight = weight
        self.succ = succ
        self.is_fully_reduced = is_fully_reduced

    def nodes(self):
        """Get all nodes in the (sub-) |EVMDD| rooted at this edge."""
        return self.succ.nodes()

    def num_nodes(self):
        """Get the number of nodes in this |EVMDD|."""
        return len(self.nodes())

    def num_edges(self):
        """Get the number of edges in this |EVMDD|."""
        return sum([len(node.children) for node in self.nodes()]) + 1

    def _apply(self, other, oper):
        """Apply an arithmetic operator `oper` to two |EVMDDs| `self` and `other`.

        See:

        * Ciardo and Siminiceanu, Using Edge-Valued Decision Diagrams for
          Symbolic Generation of Shortest Paths, FMCAD 2002, Algorithm `UnionMin`.
        * Pedram and Vrudhula, Edge-Valued Binary-Decision Diagrams,
          Algorithm `apply`.
        """
        assert self.is_fully_reduced == other.is_fully_reduced

        if _is_terminal_case(self, other, oper):
            result = _terminal_value(self, other, oper, self.is_fully_reduced)
            _log_apply(self, other, oper, result, True)
            return result

        level = max(self.succ.level, other.succ.level)
        self_children = _determine_children_on_same_level(self, other)
        other_children = _determine_children_on_same_level(other, self)

        assert len(self_children) == len(other_children)

        children = tuple([oper(sc, oc) for sc, oc in zip(self_children, other_children)])

        min_child_weight = min([child.weight for child in children])
        children = tuple([Edge(weight=child.weight-min_child_weight, succ=child.succ,
                               is_fully_reduced=self.is_fully_reduced) for child in children])

        result_weight = min_child_weight
        result_succ = Node(level, children, self.is_fully_reduced)

        if self.is_fully_reduced:
            result_succ = _perform_shannon_reduction(result_succ)
        result = Edge(weight=result_weight, succ=result_succ, is_fully_reduced=self.is_fully_reduced)
        _log_apply(self, other, oper, result, False)
        return result

    def __add__(self, other):
        return self._apply(other, Edge.__add__)

    def __sub__(self, other):
        return self._apply(other, Edge.__sub__)

    def __mul__(self, other):
        return self._apply(other, Edge.__mul__)

    def __neg__(self):
        return _make_const_evmdd(0, self.is_fully_reduced) - self

    def __pow__(self, other):
        if not isinstance(other, Integral) or other < 0:
            raise ValueError("EVMDDs may only be raised to a nonnegative integral power.")
        if other == 0:
            return _make_const_evmdd(1, self.is_fully_reduced)
        return self * (self ** (other-1))

    def __str__(self):
        return 'Edge(%s,%s)' % (self.weight, self.succ)

    def __repr__(self):
        return ('Edge(weight=%s,succ=%s,is_fully_reduced=%s)' %
                (self.weight, repr(self.succ), self.is_fully_reduced))


@memoize
class Node(EqualityMixin):
    """A node in an |EVMDD| specifying level and children.

    The `level` of a node `n` specifies how far from the sink node `n` is
    located. Level 0 is reserved for the sink node, and levels increase
    bottom-up. The level of the top-most branching is the number
    of variables in the |EVMDD| (assuming that the |EVMDD| is quasi-reduced).

    The sink node is identified as the unique node without any outgoing edges.

    Notice that only for a given variable ordering, levels will uniquely
    correspond to variable names. |EVMDDs| are constructed independently of a
    list of variable names, and variable names only come into play when an
    |EVMDD| is evaluated.

    The list of `children` holds edges to lower-level nodes for all
    values of the variable tested at this node. The list has to be
    ordered, and the index in this ordering is the domain value. I.e., if the
    current node represents a test of variable :math:`v`, and :math:`v` can take
    the three values :math:`0`, :math:`1`, and :math:`2`, then `children`
    contains three edges to the successor nodes for the cases :math:`v=0`,
    :math:`v=1`, and :math:`v=2`, respectively, in that order.

    Like `Edges`, `Nodes` keep track of whether they belong to a fully reduced
    or a quasi-reduced |EVMDD| via the flag `is_fully_reduced`.
    """

    def __init__(self, level, children, is_fully_reduced=_DEFAULT_IS_FULLY_REDUCED):
        """Initialize an |EVMDD| node with level and children."""
        self.level = level
        self.children = tuple(children)
        self.is_fully_reduced = is_fully_reduced
        if level == 0:
            assert len(children) == 0
        assert all([child.is_fully_reduced == is_fully_reduced for child in children])

    def is_sink_node(self):
        """Test if this is the sink node."""
        return len(self.children) == 0

    def nodes(self):
        """Get all nodes in the (sub-) |EVMDD| rooted at this node."""
        result = set([self])
        for child in self.children:
            result.update(child.nodes())
        return result

    def __str__(self):
        if self.is_sink_node():
            return '0'
        children = '[%s]' % (','.join([str(child) for child in self.children]),)
        return 'Node(%s,%s)' % (self.level, children)

    def __repr__(self):
        return ('Node(level=%s,children=%s,is_fully_reduced=%s)' %
                (self.level, repr(self.children), self.is_fully_reduced))


class EvmddManager(object):
    """Manager for |EVMDDs| taking care of variable names and orderings,
    variable domain sizes, and construction of basic |EVMDDs| representing
    numeric constants and variables.

    The variable names `var_names` are given in the desired variable order.

    The domain sizes are given as a separate list `var_domains` in the same
    ordering as the variable names, i.e., the :math:`i`-th entry in `var_domains`
    specifies the domain size of the :math:`i`-th variable in `var_names`. Note
    that `var_domains` only holds the domain *sizes*, not the domains
    themselves. We assume throughout that all variables have domains from
    :math:`0` to `var_domains[i] - 1`.

    The |EVMDDs| generated and managed by this manager can be either fully
    reduced or quasi-reduced. They will be fully reduced iff the flag
    `fully_reduced` is set to true (default).
    """

    def __init__(self, var_names, var_domains, fully_reduced=_DEFAULT_IS_FULLY_REDUCED):
        """Initialize an `EvmddManager` with variable names and domain sizes.
        """
        assert len(var_names) == len(var_domains)
        self._var_names = var_names
        self._var_domains = var_domains
        self._fully_reduced = fully_reduced

    def _level_to_domain_size(self, level):
        """Get the domain size of the variable associated with nodes on a given `level`.

        This is only specified if the level neither encodes the dangling incoming
        edge nor the sink node, but rather a node with level between 1 and the
        number of variables.
        """
        assert 1 <= level <= len(self._var_domains)
        return self._var_domains[-level]

    def _level_to_var_name(self, level):
        """Get the variable name associated with |EVMDD| nodes on a given `level`.

        This is only specified if the level neither encodes the dangling incoming
        edge nor the sink node, but rather a node with level between 1 and the
        number of variables.
        """
        assert 1 <= level <= len(self._var_names)
        return self._var_names[-level]

    def _var_name_to_level(self, var_name):
        """Determine the level of a given variable name."""
        assert var_name in self._var_names
        return len(self._var_names) - self._var_names.index(var_name)

    def var_name_of(self, node):
        """Determine the variable name associated with a given node.

        Args:
            `node` (Node): an |EVMDD| node.

        Returns:
            string: the name of the variable associated with the given node.
        """
        return self._level_to_var_name(node.level)

    def make_const_evmdd(self, number):
        """Construct an |EVMDD| representing a given constant number.

        This is the |EVMDD| with a single edge immediately leading to the sink
        node, labeled with the given constant as its edge weight.

        Args:
            `number` (int): a constant arithmetic function.

        Returns:
            Edge: the |EVMDD| representing the given number.
        """
        return _make_const_evmdd(number, self._fully_reduced)

    def _make_var_evmdd_for_level(self, level):
        """Construct an |EVMDD| representing a given variable.

        This is the |EVMDD| with an edge with weight zero leading to a variable
        test node branching on the variable of the requested level. The test
        node has one outgoing edge for each value `d` in the domain of the tested
        variable. The weight of the edge for value `d` has weight `d`. All edges
        lead to the unique sink node.
        """
        sink = _make_sink_node(self._fully_reduced)
        domain_size = self._level_to_domain_size(level)
        children = [Edge(weight=d, succ=sink,
                         is_fully_reduced=self._fully_reduced) for d in range(domain_size)]
        var_node = Node(level=level, children=children, is_fully_reduced=self._fully_reduced)
        return Edge(weight=0, succ=var_node, is_fully_reduced=self._fully_reduced)

    def make_var_evmdd_for_var(self, var_name):
        """Construct an |EVMDD| representing a given variable.

        Args:
            `var_name` (string): a variable name.

        Returns:
            Edge: the |EVMDD| representing the given variable according to the
            variable order of this manager.

        Note:
            Fails if the given variable name is not known.
        """
        return self._make_var_evmdd_for_level(self._var_name_to_level(var_name))


def evaluate(evmdd, valuation, manager):
    """Evaluate an |EVMDD| `evmdd` for given valuation `valuation` and
    EvmddManager `manager`.

    This function traverses the given |EVMDD| from top to bottom, following
    the unique path consistent with `valuation`. Along the way, it adds up
    the encountered edge weights. In order to match the variable names
    mentioned in `valuation` to levels in the |EVMDD|, this function needs
    to have access to the variable ordering provided by the `manager`. At
    each interior node `n`, the variable name `v` associated with `n` is
    looked up by the `manager`, and the value that `v` has is looked up in
    `valuation`. Then, the corresponding edge is traversed.

    Args:
        `evmdd` (Edge): an |EVMDD|.

        `valuation` (dict[string->int]): a variable-value mapping to be evaluated.

        `manager` (EvmddManager): an EvmddManager that knows variable names and orderings.

    Returns:
        `int`: the value the function represented by `evmdd` has under `valuation`.

    Examples:
        >>> var_names = ['A', 'B']
        >>> var_domains = [2, 2]
        >>> manager = EvmddManager(var_names, var_domains)
        >>>
        >>> N00 = _make_sink_node()
        >>> N10 = Node(level=1, children=[Edge(weight=0, succ=N00), Edge(weight=1, succ=N00)])
        >>> N11 = Node(level=1, children=[Edge(weight=0, succ=N00), Edge(weight=2, succ=N00)])
        >>> N20 = Node(level=2, children=[Edge(weight=0, succ=N10), Edge(weight=1, succ=N11)])
        >>> evmdd = Edge(weight=2, succ=N20)
        >>>
        >>> s = { 'A': 0, 'B': 0 }
        >>> evaluate(evmdd, s, manager)
        2
        >>> s = { 'A': 0, 'B': 1 }
        >>> evaluate(evmdd, s, manager)
        3
        >>> s = { 'A': 1, 'B': 0 }
        >>> evaluate(evmdd, s, manager)
        3
        >>> s = { 'A': 1, 'B': 1 }
        >>> evaluate(evmdd, s, manager)
        5
    """
    current_edge = evmdd
    current_node = current_edge.succ
    result = current_edge.weight
    while not current_node.is_sink_node():
        assert min([child.weight for child in current_node.children]) == 0
        if current_edge.is_fully_reduced:
            assert all([child.succ.level < current_node.level for child in current_node.children])
            assert (len(set([child.succ for child in current_node.children])) > 1 or
                    max([child.weight for child in current_node.children]) > 0) # Shannon
        else:
            assert all([child.succ.level == current_node.level-1
                        for child in current_node.children])
        var_name = manager.var_name_of(current_node)
        var_value = valuation[var_name]
        assert 0 <= var_value < len(current_node.children)
        current_edge = current_node.children[var_value]
        result = result + current_edge.weight
        current_node = current_edge.succ
    return result


def _test():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()
