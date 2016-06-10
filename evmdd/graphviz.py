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
Graphviz output and display of |EVMDDs|.
"""

from itertools import groupby
from subprocess import call
from sys import platform
import logging

class GraphvizWriter(object):
    """Helper class that writes |EVMDDs| to strings in Graphviz/DOT format."""

    _root_node_gvz_tmpl = '%s [style=invis];'
    _var_node_gvz_tmpl = '%s [style=filled,fillcolor=lightgrey,label="%s"];'
    _sink_node_gvz_tmpl = '%s [shape=box,height=0.25,width=0.5,rank=sink,label="0"];'
    _weight_node_gvz_tmpl = '%s [shape=box,height=0.25,width=0.5,label="%+d"];'

    _var_to_weight_edge_gvz_tmpl = '%s -> %s [arrowhead=none, label="%s"];'
    _weight_to_var_edge_gvz_tmpl = '%s -> %s;'

    _var_node_name_tmpl = '"s%s[level=%s]"'
    _weight_node_name_tmpl = '"s%s[level=%s]=%s"'

    def __init__(self, manager):
        self._manager = manager
        self._index = None

    def _root_node_gvz(self, var_node_name):
        return self._root_node_gvz_tmpl % var_node_name

    def _var_node_gvz(self, var_node_name, var_name):
        return self._var_node_gvz_tmpl % (var_node_name, var_name)

    def _sink_node_gvz(self, var_node_name):
        return self._sink_node_gvz_tmpl % var_node_name

    def _weight_node_gvz(self, weight_node_name, weight):
        return self._weight_node_gvz_tmpl % (weight_node_name, weight)

    def _var_to_weight_edge_gvz(self, var_node_name, weight_node_name, domain_idx):
        return (self._var_to_weight_edge_gvz_tmpl %
                (var_node_name, weight_node_name, domain_idx))

    def _weight_to_var_edge_gvz(self, weight_node_name, succ_var_node_name):
        return (self._weight_to_var_edge_gvz_tmpl %
                (weight_node_name, succ_var_node_name))

    def _var_node_name(self, node):
        return self._var_node_name_tmpl % (self._index[node], node.level)

    def _weight_node_name(self, node, domain_idx):
        return self._weight_node_name_tmpl % (self._index[node], node.level, domain_idx)

    def _root_edge_to_gvz(self, evmdd):
        var_node_name = 'dummyNode'
        weight_node_name = 'constantWeight'
        var_node = self._root_node_gvz(var_node_name)
        weight_node = self._weight_node_gvz(weight_node_name, evmdd.weight)
        var_to_weight_edge = self._var_to_weight_edge_gvz(var_node_name, weight_node_name, '')
        succ_var_node_name = self._var_node_name(evmdd.succ)
        weight_to_var_edge = self._weight_to_var_edge_gvz(weight_node_name, succ_var_node_name)
        return [var_node, weight_node, var_to_weight_edge, weight_to_var_edge]

    def _node_to_gvz(self, node):
        var_node_name = self._var_node_name(node)
        if node.is_sink_node():
            return [self._sink_node_gvz(var_node_name)]
        var_name = self._manager.var_name_of(node)
        var_node = self._var_node_gvz(var_node_name, var_name)
        result = [var_node]
        for domain_idx, child in enumerate(node.children):
            weight_node_name = self._weight_node_name(node, domain_idx)
            weight_node = self._weight_node_gvz(weight_node_name, child.weight)
            var_to_weight_edge = self._var_to_weight_edge_gvz(var_node_name,
                                                              weight_node_name, domain_idx)
            succ_var_node_name = self._var_node_name(child.succ)
            weight_to_var_edge = self._weight_to_var_edge_gvz(weight_node_name, succ_var_node_name)
            result.extend([weight_node, var_to_weight_edge, weight_to_var_edge])
        return result

    def _node_rank_to_gvz(self, rank):
        rank = ';'.join([self._var_node_name(node) for node in rank])
        return '{rank = same; %s;}' % rank

    def evmdd_to_gvz(self, evmdd):
        """Translate a given |EVMDD| to Graphviz format.

        Args:
            evmdd (Edge): an |EVMDD|.

        Returns:
            `string`: an encoding of the |EVMDD| in Graphviz/DOT format.
        """
        nodes = evmdd.nodes()
        sorting_fn = lambda node: node.level
        nodes = sorted(nodes, key=sorting_fn)

        self._index = {node: idx for idx, node in enumerate(nodes)}
        lines = ['digraph G {']
        lines.extend(self._root_edge_to_gvz(evmdd))
        for node in nodes:
            lines.extend(self._node_to_gvz(node))
        for _, group in groupby(nodes, key=sorting_fn):
            lines.append(self._node_rank_to_gvz(group))
        lines.extend(['}'])
        return '\n'.join(lines)


class EvmddVisualizer(object):
    """Helper class that displays |EVMDDs| on the screen.
    """

    def __init__(self, manager):
        self._gvz_writer = GraphvizWriter(manager)
        self._temp_files = []

    def __enter__(self):
        self._temp_files = []
        return self

    def __exit__(self, type, value, traceback):
        for filename in self._temp_files:
            self._remove_tmp_file(filename)
        self._temp_files = []

    def _add_temp_file(self, filename):
        self._temp_files.append(filename)

    def _encode_in_gvz_format(self, evmdd):
        return self._gvz_writer.evmdd_to_gvz(evmdd)

    def _write_to_tmp_file(self, gvz, dot_filename):
        try:
            with open(dot_filename, 'w') as dot_file:
                dot_file.write(gvz)
        except PermissionError as e:
            logging.error('EVMDD visualization failed. No permission to write to %s' % dot_filename)

    def _call_xdot(self, dot_filename):
        cmd = ['xdot', dot_filename]
        try:
            call(cmd)
        except FileNotFoundError as e:
            logging.error('Skipped EVMDD visualization due to error calling xdot.')

    def _convert_to_svg(self, dot_filename):
        cmd = ['dot', '-Tsvg', '-O', dot_filename]
        call(cmd)

    def _display_svg_macos(self, svg_filename):
        svg_fqdn = 'file://%s' % svg_filename
        cmd = ['open', '-a', 'safari', svg_fqdn]
        call(cmd)

    def _remove_tmp_file(self, filename):
        cmd = ['rm', filename]
        try:
            call(cmd)
        except FileNotFoundError as e:
            logging.warning('Could not delete temp file %s.' % filename)

    def visualize(self, evmdd, file_prefix=None, keep_files_after_use=False):
        """Visualize a given |EVMDD|.

        The given |EVMDD| is first exported in Graphviz/DOT syntax and
        written to a temporary DOT file. Then, depending on the OS, the
        DOT file is either displayed using `xdot` (under Linux), or converted
        to SVG using `dot` and subsequently openend in a Browser (Safari,
        under Mac OS X).

        In both cases, we assume that Graphviz is installed.

        Notice that generated DOT/SVG files are not deleted after visualization.

        Args:
            `evmdd` (Edge): the |EVMDD| to be visualized.

            `file_prefix` (string, optional): path and file name of dot file to
            write to (``/tmp/evmdd-gvz-UUID`` by default).

            `keep_files_after_use` (boolean, optional): do not delete dot file
            (and possible svg file) after use (``False``, i.e., delete, by default).

        Returns:
            `None`

        Side effect:
            visualization of the given |EVMDD|.
        """
        if not file_prefix:
            import uuid
            rnd = uuid.uuid4()
            file_prefix = '/tmp/evmdd-gvz-%s' % rnd
        dot_filename = file_prefix + '.dot'
        svg_filename = file_prefix + '.dot.svg'
        if not keep_files_after_use:
            self._add_temp_file(dot_filename)
        gvz = self._encode_in_gvz_format(evmdd)
        self._write_to_tmp_file(gvz, dot_filename)
        if platform == 'darwin':
            try:
                self._convert_to_svg(dot_filename)
                if not keep_files_after_use:
                    self._add_temp_file(svg_filename)
                self._display_svg_macos(svg_filename)
            except FileNotFoundError as e:
                logging.error('Skipped EVMDD visualization due to dot/Safari issue.')
        else:
            self._call_xdot(dot_filename)
