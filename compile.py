#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import sys
from collections import defaultdict
from copy import deepcopy
from itertools import product

import instantiate
import normalize
import timers
import options
import pddl
import fact_groups
import pddl_parser

from sdac_compilation import EVMDDActionCompiler
from sdac_to_pddl import SdacPDDLWriter

simplified_effect_condition_counter = 0
added_implied_precondition_counter = 0


def strips_to_sas_dictionary(groups, assert_partial):
    dictionary = {}
    for var_no, group in enumerate(groups):
        for val_no, atom in enumerate(group):
            dictionary.setdefault(atom, []).append((var_no, val_no))
    if assert_partial:
        assert all(len(sas_pairs) == 1
                   for sas_pairs in dictionary.values())
    return [len(group) + 1 for group in groups], dictionary


def build_mutex_key(strips_to_sas, groups):
    group_keys = []
    for group in groups:
        group_key = []
        for fact in group:
            if strips_to_sas.get(fact):
                for var, val in strips_to_sas[fact]:
                    group_key.append((var, val))
            else:
                print("not in strips_to_sas, left out:", fact)
        group_keys.append(group_key)
    return group_keys


def compile(task):
    with timers.timing("Instantiating", block=True):
        (relaxed_reachable, atoms, actions, axioms,
         reachable_action_params) = instantiate.explore(task)

    # Transform logical terms of all cost functions into arithmetic terms.
    for a in actions:
        if isinstance(a.cost, pddl_parser.CostNode):
            a.cost.transform_logic()    

    # writing value tuples to atoms
    predicate_dict = dict((p.name, p ) for p in task.predicates)
    for a in atoms:    
        p = predicate_dict.get(a.predicate)
        if p and len(p.value_mapping) > 0:
            a.value = p.value_mapping.get(a.args)
    if not relaxed_reachable:
        return unsolvable_sas_task("No relaxed solution")
    # HACK! Goals should be treated differently.
    if isinstance(task.goal, pddl.Conjunction):
        goal_list = task.goal.parts
    else:
        goal_list = [task.goal]
    for item in goal_list:
        assert isinstance(item, pddl.Literal)

    with timers.timing("Computing fact groups", block=True):
        groups, mutex_groups, translation_key, atom_groups = fact_groups.compute_groups(
            task, atoms, reachable_action_params)

    with timers.timing("Building STRIPS to SAS dictionary"):
        ranges, strips_to_sas = strips_to_sas_dictionary(
            groups, assert_partial=options.use_partial_encoding)
    with timers.timing("Building dictionary for full mutex groups"):
        mutex_ranges, mutex_dict = strips_to_sas_dictionary(
            mutex_groups, assert_partial=False)
    with timers.timing("Building mutex information", block=True):
        mutex_key = build_mutex_key(strips_to_sas, mutex_groups)

    compiler = EVMDDActionCompiler()
    actions = compiler.evmdd_action_compilation(actions)
    pddl_writer = SdacPDDLWriter(compiler._fact_name_dict)
    pddl_writer.write_pddl_files(options.domain, options.task, actions)
    print("done!")


def main():
    timer = timers.Timer()
    
    with timers.timing("Parsing", True):
        task = pddl_parser.open(task_filename=options.task, domain_filename=options.domain)
    with timers.timing("Normalizing task"):
        normalize.normalize(task)

    compile(task)
    print("Done! %s" % timer) 


if __name__ == "__main__":
    main()
