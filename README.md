SDAC-Compiler
==================================
This compiler takes a PDDL problem with state-dependent action costs and generates a PDDL problem with constant costs, based on the action-compilation with edge-valued multi-values decision diagrams (EVMDDs), as described in 
[*Delete Relaxations for Planning with State-Dependent Action Costs*](http://gki.informatik.uni-freiburg.de/papers/geisser-etal-ijcai2015.pdf).

Disclaimer
----------
This project is still in a __very__ early stage and therefore is not stable and bug-free. It is currently built into the translator tool of [Fast Downward](http://www.fast-downward.org/). While we definitely need the translator tool, we plan to decouple the first major version of our compiler. We are currently rewriting most parts of the code and hope to ship the first fully functional version around July.

Requirements
----------
- Python 3
- Python [NetworkX library](https://networkx.github.io/)
- Optional: [xdot](https://pypi.python.org/pypi/xdot) for EVMDD visualization

Usage
----------
./compile.py <path-to-domainfile> <path-to-problemfile> [--viz] \
This generates *domain-out.pddl* and *problem-out.pddl*, the compiled PDDL files with constant action cost.

Optional arguments:
- --viz: EVMDD visualization for action cost functions

Copyright
--------

Copyright (C) 2016 Florian Geißer and Robert Mattmüller

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>
