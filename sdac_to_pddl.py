#!/usr/bin/env python3
# -*- coding: utf-8 -*-

try:
    # Python 3.x
    import builtins
except ImportError:
    # Python 2.x
    import __builtin__ as builtins

import pprint
from pddl_parser import lisp_parser
from action_compilation import Fact as MDDFact, Action as MDDAction

class SdacPDDLWriter(object):

    def __init__(self, fact_name_dict, semaphore = 'sem', auxiliary = 'aux'):
        self._fact_name_dict = fact_name_dict
        self._sem = semaphore
        self._aux = auxiliary
        self._aux_predicates = set()

    def _mdd_action_to_pddl(self, action):
        """ Binarize auxiliary variables and add increase total cost effect. 
        Replaces the finite domain auxiliary variable with a binary auxiliary
        variable and adds that to the set of auxiliary predicates.
        """
        new_precs = []
        new_effs = []
        for fact in action.pre:
            if self._aux in fact.var:
                # We do not need aux=0, as the semaphore has the same effect
                if fact.val != 0:
                    new_precs.append(MDDFact(fact.var + "-" + str(fact.val), 1))
                    new_effs.append(MDDFact(fact.var + "-" + str(fact.val), 0))
                    self._aux_predicates.add(fact.var + "-" + str(fact.val))
            else:
                new_precs.append(fact)
        for fact in action.eff:
            if self._aux in fact.var:
                if fact.val != 0:
                    new_effs.append(MDDFact(fact.var+"-"+str(fact.val), 1))
            else:
                new_effs.append(fact)
        # add cost function effect
        new_effs.append(MDDFact("increase (total-cost) " + str(action.cost), 1))
        return MDDAction(action.name, new_precs, new_effs, action.cost)
    
    def _action_to_pddl_string(self, action):
        """ Returns the PDDL string of an action
        """
        result = "(:action "
        result += action.name.replace(" ", "-") + "\n"
        #TODO What if we had parameters?
        result += ":parameters ()\n"
        result += ":precondition " + self._mdd_facts_to_pddl_string(action.pre) + "\n"
        result += ":effect " + self._mdd_facts_to_pddl_string(action.eff) + "\n)"
        return result
    
    def _mdd_facts_to_pddl_string(self, facts):
        """ Returns the PDDL string of a set of facts
        """
        if len(facts) > 1:
            return "(and " + ' '.join(
                    [self._mdd_fact_to_pddl_string(fact) for fact in facts]) + ")"
        elif len(facts) == 1:
            return self._mdd_fact_to_pddl_string(facts[0])
        else:
            return ""
    
    def _mdd_fact_to_pddl_string(self, fact):
        """ Returns the PDDL string of a fact
        """
        if fact.var in self._fact_name_dict:
            fact_name = self._fact_name_dict[fact.var]
        else:
            fact_name = fact.var
        fact_string = "(" + fact_name + ")"
        if fact.val == 0:
            return "(not " + fact_string + ")"
        return fact_string
    
    def _add_requirements(self, domain_tokens):
        """ Adds necessary requirements to the pddl token list, if not already
        present.
        """
        for token_list in domain_tokens:
            if 'requirements' in token_list[0]:
                if not any('action-costs' in token for token in token_list):
                    token_list.append(':action-costs')
                if not any('negative-preconditions' in token for token in token_list):
                    token_list.append(':negative-preconditions')
                return
        # append requirements if there are none defined
        domain_tokens.insert(2, [':requirements', ':action-costs',
            ':negative-preconditions'])
    
    def _add_aux_predicates(self, domain_tokens):
        """ Adds auxiliary predicates to the PDDL description
        """
        for token_list in domain_tokens:
            if 'predicates' in token_list[0]:
                for pred in sorted(self._aux_predicates):
                    token_list.append([pred])
                token_list.append([self._sem])
                return
        # append predicates if there are none defined
        domain_tokens.insert(3, [':predicates', sorted(self._aux_predicates),
            self._sem])
    
    def _add_total_cost_function(self, domain_tokens):
        """ Adds total cost function to the PDDL description if not already
        present.
        """
        for token_list in domain_tokens:
            if 'functions' in token_list[0]:
                if not any('total-cost' in token for token in token_list):
                    token_list.extend(['total-cost'], '-', 'number')
                return
        # append requirements if there are none defined
        domain_tokens.insert(4, [':functions', ['total-cost'], '-', 'number'])


    def _prob_obj_to_constants(self, domain_tokens, problem_tokens):
        constants = []
        for token_list in problem_tokens:
            if 'objects' in token_list[0]:
                constants = [':constants'] + token_list[1:]
                problem_tokens.remove(token_list)
                break
        if constants:
            domain_tokens.append(constants)
    
    def _get_pddl_string(self, tokens):
        """ Returns the PDDL string of a set of pddl tokens
        """
        pp = pprint.PrettyPrinter()
        output = pp.pformat(tokens)
        output = output.replace("[", "(").replace("]",")")
        output = output.replace(",","").replace("'", "")
        return output
    
    def _add_metric(self, problem_tokens):
        """ Adds the minimize metric to the PDDL description, if not already
        present.
        """
        if not any(':metric' in token for token in problem_tokens):
            problem_tokens.append([":metric minimize (total-cost)"])

    
    def write_pddl_files(self, domain_file, problem_file, mdd_actions):
        """ Writes the domain and problem pddl files for a sdac compilation.
        """
        #TODO A better design would be to take an action compilation as input and
        # generate the pddl description for the compilation. This way we could
        # use this method for the exponential compilation as well
        domain_tokens = lisp_parser.parse_nested_list(builtins.open(domain_file))
        problem_tokens = lisp_parser.parse_nested_list(builtins.open(problem_file))
        # delete actions in token list
        domain_tokens = [x for x in domain_tokens if not (':action' in x[0])]
        # TODO decouple add_predicates and mdd_action_to_pddl
        actions = [self._mdd_action_to_pddl(mdd_action) for mdd_action in mdd_actions]
        self._add_requirements(domain_tokens)
        self._add_aux_predicates(domain_tokens)
        self._add_total_cost_function(domain_tokens)
        self._prob_obj_to_constants(domain_tokens, problem_tokens)
        # Remove last ')' as we have to append actions
        output = self._get_pddl_string(domain_tokens)[:-1] + "\n"

        #TODO handle indent in output
        output += '\n'.join(str(self._action_to_pddl_string(action)) for action in
                actions)
        output += ')'
        with open("domain-out.pddl", "w") as output_file:
            print(output, file=output_file)
    
        self._add_metric(problem_tokens)
        #TODO Prettier output for objects (no newlines)
        with open("problem-out.pddl", "w") as output_file:
            print(self._get_pddl_string(problem_tokens), file=output_file)


