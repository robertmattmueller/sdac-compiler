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

"""Utilities for |EVMDD| library.
"""

from functools import wraps

def memoize(obj):
    """Memoizes an object.

    We use memoization for |EVMDD| nodes and edges to avoid duplicate nodes with
    identical level and children, and duplicate edges with identical weights and
    successor nodes, respectively. Such |EVMDDs| would represent the same
    arithmetic function, and it is both memory efficient to have only one copy
    of each such |EVMDD|, and necessary to guarantee isomorphism reduction of
    all constructed |EVMDDs|. If there is only one `EVMDD` object for each
    arithmetic function, then all |EVMDDs| will be isomorphism reduced by
    construction.

    Code taken from ``https://wiki.python.org/moin/PythonDecoratorLibrary#Memoize``.
    """
    cache = obj.cache = {}

    @wraps(obj)
    def memoizer(*args, **kwargs):
        key = repr(args) + repr(kwargs)
        if key not in cache:
            cache[key] = obj(*args, **kwargs)
        return cache[key]
    return memoizer


class EqualityMixin(object):
    """Mixin used in other classes to provide a default ``==`` and ``!=`` test
    and hash function.

    Equality, inequality and hash function are defined component-wise.
    """
    def __init__(self):
        pass

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
                and self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(frozenset(self.__dict__.items()))
