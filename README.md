SDAC-Compiler
==================================
This compiler takes a PDDL problem with state-dependent action costs and generates a PDDL problem with constant costs, based on the action-compilation with edge-valued multi-values decision diagrams (EVMDDs), as described in 
[*Delete Relaxations for Planning with State-Dependent Action Costs*](http://gki.informatik.uni-freiburg.de/papers/geisser-etal-ijcai2015.pdf).

An example PDDL domain file can be found [here](https://raw.githubusercontent.com/robertmattmueller/sdac-compiler/master/example/domain-sdac.pddl) and the BNF grammar for cost functions can be found [here](https://github.com/robertmattmueller/sdac-compiler/blob/master/documents/bnf.pdf). Note that the compiler currently does not support real numbers and the forall/exists language constructs.

We also have a working plugin for the Planning.Domains editor, thanks to Christian Muise. [This](http://editor.planning.domains/#http://www.haz.ca/tutorial2.js) link immediately loads the compiler plugin and changes the underlying planner to Fast Downward with lm-cut. Note that the online compiler has a **15 seconds timeout**, keep this in mind for complex cost functions.

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
./compile.py *path/to/domainfile* *path/to/problemfile* [--viz]

This generates *domain-out.pddl* and *problem-out.pddl*, the compiled PDDL files with constant action cost.

Optional arguments:
- --viz: EVMDD visualization for action cost functions

Example:
- ./compile.py example/domain-sdac.pddl example/prob02.pddl --viz

External tools
--------
This compiler uses the following external tools:
- [Fast Downward translator](http://www.fast-downward.org/)
- [pyevmdd](https://github.com/robertmattmueller/pyevmdd) as decision diagram library.
- [SymPy](http://www.sympy.org/de/index.html) to minimize cost function terms.

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
