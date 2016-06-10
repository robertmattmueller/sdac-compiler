#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pddl
import pddl_parser
import itertools
from action_compilation import Fact as MDDFact, Action as MDDAction, ActionCompiler
import options

class EVMDDActionCompiler(object):

    def __init__(self, semaphore='sem', auxiliary='aux-'):
        self._fact_name_dict = {}
        self._sem = semaphore
        self._aux = auxiliary

    def evmdd_action_compilation(self, actions):
        """ Creates the evmdd based action compilation for a list of pddl
        actions. 
        """
        mdd_actions = []
        for action in actions:
            if self._is_useful(action):
                mdd_actions.append(self._create_evmdd_action(action))
        var_names = self._read_ordering()
        compiler = ActionCompiler(semaphore = self._sem, var_names = var_names,
                var_domains = {}, fully_reduced=True)
        visualize = options.viz
        compilations = [compiler.action_compilation(
            action, '%s%s' % (self._aux, auxidx), visualize)
                for auxidx, action in enumerate(mdd_actions)]
        return itertools.chain.from_iterable(compilations)

    def _read_ordering(self):
        if options.order:
            return [line.strip('\n').strip() for line in open(options.order, 'r')]
        else:
            return 'AUTOMATIC'

    def _create_evmdd_action(self, action):
        """ Takes an instantiated pddl action consisting of name, precondition, 
        add and delete effects and a (possibly constant) cost function 
        and creates an action object in internal representation required 
        for the action compiler from pyMDD.
        """
        #TODO action parameters? conditional effects?
        name = action.name[1:-1]
        precondition = []
        for prec in action.precondition:
            fact_string = self._atom_to_string(prec)
            if prec.negated:
                precondition.append(MDDFact(fact_string, 0))
            else:
                precondition.append(MDDFact(fact_string, 1))
        effect = []
        for eff in action.del_effects:
            effect.append(MDDFact(self._atom_to_string(eff[1]), 0))
        for eff in action.add_effects:
            effect.append(MDDFact(self._atom_to_string(eff[1]), 1))
        cost_function = self._get_cost_function(action.cost)
        if len(str(cost_function)) > 3:
            print("Cost of action " + name + ":" + cost_function)
        return MDDAction(name, precondition, effect, cost_function)

    def _atom_to_string(self, atom):
        """ String representation of an atom for MDD computation. Illegal
        symbols that exist in PDDL (e.g. " ", "-", ",") are replaced by a space
        and saved in _fact_name_dict.
        """
        args = ' '.join([str(arg) for arg in atom.args])
        name = atom.predicate + " " + args
        # Remove illegal symbols for mdd variables
        # TODO Find a better way to handle the multiple PDDL symbols
        result = name.replace(" ","_").replace("-","_").replace(",","_")
        result = result.replace("(","_").replace(")","_")
        #if result in self._fact_name_dict:
        #    raise SystemExit(
        #        "error: atom %s has replacement name %s, but this name already
        #        exists for atom %s. Consider editing the original PDDL file." %
        #        (name, result, self._fact_name_dict[result]))
        self._fact_name_dict[result] = name
        return result

    def _get_cost_function(self, cost):
        """ Returns a string representation of the simplified cost function.
        """
        if isinstance(cost, pddl_parser.CostNode):
            #TODO Rewrite methods
            simple_cost = cost.get_simplified_function()
            mdd_str = simple_cost.to_evmdd_str()[1:-1]
            cost_atoms = cost.get_atoms_set()
            symbols = []
            function = simple_cost.get_simplified_function_rek()
            for atom in cost_atoms:
                symbol = self._atom_to_string(atom)
                function = function.replace(str(atom), symbol)
            # Remove '(' and ')'
            return function[1:-1]
        return str(cost)

    def _is_useful(self, action):
        """
        If the effect is a subset of the precondition
        the action does not do anything.
        """
        for cond, effect_atom in action.add_effects:
            eff_is_subset = False
            for prec_atom in action.precondition:
                if not prec_atom.negated and prec_atom == effect_atom:
                    eff_is_subset = True
                    break
            if not eff_is_subset:
                return True    
        for cond, effect_atom in action.del_effects:
            eff_is_subset = False
            for prec_atom in action.precondition:
                if prec_atom.negated and prec_atom == effect_atom:
                    eff_is_subset = True
                    break
            if not eff_is_subset:
                return True
        return False
