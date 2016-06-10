from __future__ import print_function
try:
    # Python 3.x
    import builtins
except ImportError:
    # Python 2.x
    import __builtin__ as builtins

import pprint

import pddl
import pddl_parser
import itertools
import copy
import os
import ast
import options
from evmdd import term_to_evmdd, evaluate, EvmddVisualizer
from action_compilation import *

from . import lisp_parser

class Node:
    def __init__(self, node_id, atom, val):
        self.node_id = node_id
        self.atom = atom
        self.val = val
        self.parents = []
        self.children = []
        self.aux_id = -1

def transform_exp_actions(actions, mutex_groups):
    """ 
    Exponential transformation of actions with sdac 
    into actions with constant action costs.
    """
    
    mutex_dict = dict()
    for group in mutex_groups:
        for elem in group:
            mutex_dict[elem] = group
    del_actions = []
    new_actions = actions
    add_actions = []
    for action in new_actions:
        if isinstance(action.cost, pddl_parser.CostNode):
            del_actions.append(action)
            if not is_usefull(action):
                continue
            at_list = []
            consistent_actions = []
            used_dict = dict()
            for atom in action.cost.get_atoms_set():
                if atom not in mutex_dict:
                    at_list.append([atom, pddl.NegatedAtom(atom.predicate, atom.args)])
                elif len(mutex_dict[atom]) == 1:
                    at_list.append([atom, pddl.NegatedAtom(atom.predicate, atom.args)])
                elif str(mutex_dict[atom]) not in used_dict:
                    at_list.append(mutex_dict[atom])
                    used_dict[str(mutex_dict[atom])] = True
            for combi in itertools.product(*at_list):
                pre = []
                pre.extend(action.precondition)
                pre.extend(combi)
                if not is_consistent(combi, action.precondition, mutex_dict):
                    continue
                if isinstance(action, pddl.Action):
                    action_copy = pddl.Action(action.name, action.parameters, action.num_external_parameters, pre, action.effects, str(action.cost.get_cost(combi)))
                else:
                    action_copy = pddl.PropositionalAction(action.name, pre, [], str(action.cost.get_cost(combi)))
                    action_copy.add_effects.extend(action.add_effects)
                    action_copy.del_effects.extend(action.del_effects)
                consistent_actions.append(action_copy)
            add_actions.extend(consistent_actions)
    for a in del_actions:
        new_actions.remove(a)
    new_actions.extend(add_actions)
    return new_actions

def atom_to_string(atom):
    args = ' '.join([str(arg) for arg in atom.args])
    result = atom.predicate + " " + args
    # Remove illegal symbols for mdd variables
    result = result.replace(" ","_").replace("-","|").replace(",",";")
    return result

def atom_to_mdd_fact(atom):
    val = 0 if atom.negated else 1
    return Fact(atom_to_string(atom), val)

def get_cost_function(cost):
    if isinstance(cost, pddl_parser.CostNode):
        #TODO Rewrite methods
        simple_cost = cost.get_simplified_function()
        mdd_str = simple_cost.to_evmdd_str()[1:-1]
        cost_atoms = cost.get_atoms_set()
        symbols = []
        function = simple_cost.get_simplified_function_rek()
        for atom in cost_atoms:
            symbol = atom_to_string(atom)
            function = function.replace(str(atom), symbol)
        # Remove '(' and ')'
        return function[1:-1]
    return str(cost)

def gather_var_domains(actions):
    result = {}
    for action in actions:
        for fact in itertools.chain(action.pre, action.eff):
            result[fact.var] = 2
    return result

def fd_action_to_pddl(action, aux_string, aux_predicates):
    new_precs = []
    new_effs = []
    for fact in action.pre:
        if aux_string in fact.var:
            # We do not need aux=0, as the semaphore has the same effect
            if fact.val != 0:
                new_precs.append(Fact(fact.var + "-" + str(fact.val), 1))
                new_effs.append(Fact(fact.var + "-" + str(fact.val), 0))
                aux_predicates.add(fact.var + "-" + str(fact.val))
        else:
            new_precs.append(fact)
    for fact in action.eff:
        if aux_string in fact.var:
            if fact.val != 0:
                new_effs.append(Fact(fact.var+"-"+str(fact.val), 1))
        else:
            new_effs.append(fact)
    # add cost function effect
    new_effs.append(Fact("increase_(total-cost)_" + str(action.cost), 1))
    return Action(action.name, new_precs, new_effs, action.cost)

def mdd_action_to_pddl(action):
    result = "(:action "
    result += action.name.replace(" ", "-") + "\n"
    #TODO What if we had parameters?
    result += ":parameters ()\n"
    result += ":precondition " + mdd_facts_to_pddl(action.pre) + "\n"
    result += ":effect " + mdd_facts_to_pddl(action.eff) + "\n)"
    return result

def mdd_facts_to_pddl(facts):
    if len(facts) > 1:
        return "(and " + ' '.join([mdd_fact_to_pddl(fact) for fact in facts]) + ")"
    elif len(facts) == 1:
        return mdd_fact_to_pddl(facts[0])
    else:
        return ""

def mdd_fact_to_pddl(fact):
    #TODO dictionary for different replacements
    replaced = fact.var.replace("_", " ").replace("|","-").replace(";",",")
    fact_string = "(" + replaced + ")"
    if fact.val == 0:
        return "(not " + fact_string + ")"
    return fact_string

def create_evmdd_action(action):
    #TODO action parameters?
    name = action.name[1:-1]
    eff_list = []
    #TODO conditional effects?
    for eff in action.add_effects:
        eff_list.append(eff[1])
    for eff in action.del_effects:
        eff_list.append(pddl.NegatedAtom(eff[1].predicate, eff[1].args))
    prec_facts = []
    eff_facts = []
    for prec in action.precondition:
        prec_facts.append(atom_to_mdd_fact(prec))
    for eff in eff_list:
        eff_facts.append(atom_to_mdd_fact(eff))
    return Action(name, prec_facts, eff_facts, get_cost_function(action.cost))

def transform_evmdd_actions2(actions):
    mdd_actions = []
    for pddl_op in actions:
        if not is_useful(pddl_op):
            continue
        mdd_actions.append(create_evmdd_action(pddl_op))
    var_domains = {}
    var_domains = gather_var_domains(mdd_actions)
    # compiler
    compiler = ActionCompiler(semaphore = 'sem', var_names = 'AUTOMATIC',
            var_domains = var_domains, fully_reduced=True)
    compilations = [compiler.action_compilation(action, '%s%s' % ('aux-', auxidx))
                    for auxidx, action in enumerate(mdd_actions)]
    result = []
    aux_predicates = set()
    for compilation in compilations:
        print(len(compilation))
        for action in compilation:
            action = fd_action_to_pddl(action, 'aux-', aux_predicates)
            action = mdd_action_to_pddl(action)
            #print(action)
            result.append(action)
        #print('-----------------------------------------')
    write_pddl_files(options.domain, aux_predicates, result)
    return result

def add_requirements(domain_tokens):
    for token_list in domain_tokens:
        if 'requirements' in token_list[0]:
            if not any('action-costs' in token for token in token_list):
                token_list.append(':action-costs')
            if not any('negative-preconditions' in token for token in token_list):
                token_list.append(':negative-preconditions')
            return
    # append requirements if there are none defined
    domain_tokens.insert(2, [':requirements', ':action-costs',
    'negative-preconditions'])

def add_aux_predicates(domain_tokens, aux_predicates):
    for token_list in domain_tokens:
        if 'predicates' in token_list[0]:
            for aux in sorted(aux_predicates):
                token_list.append([aux])
            #TODO parameterize sem
            token_list.append(['sem'])
            return
    # append predicates if there are none defined
    domain_tokens.insert(3, [':predicates', sorted(aux_predicates), 'sem'])

def add_total_cost_function(domain_tokens):
    for token_list in domain_tokens:
        if 'functions' in token_list[0]:
            if not any('total-cost' in token for token in token_list):
                token_list.extend(['total-cost'], '-', 'number')
            return
    # append requirements if there are none defined
    domain_tokens.insert(4, [':functions', ['total-cost'], '-', 'number'])

def get_pddl_string(tokens):
    pp = pprint.PrettyPrinter(indent=4)
    output = pp.pformat(tokens)
    output = output.replace("[", "(").replace("]",")")
    output = output.replace(",","").replace("'", "")
    return output

def add_metric(problem_tokens):
    if not any(':metric' in token for token in problem_tokens):
        problem_tokens.append([":metric minimize (total-cost)"])

def write_pddl_files(domain_file, aux_predicates, new_actions):
    domain_tokens = lisp_parser.parse_nested_list(builtins.open(domain_file))
    # delete actions in token list
    domain_tokens = [x for x in domain_tokens if not (':action' in x[0])]
    add_requirements(domain_tokens)
    add_aux_predicates(domain_tokens, aux_predicates)
    add_total_cost_function(domain_tokens)
    # Remove last ')' as we have to append actions
    output = get_pddl_string(domain_tokens)[:-1] + "\n"
    #TODO handle indent in output
    output += '\n'.join(str(action) for action in new_actions)
    output += ')'
    with open("domain-out.pddl", "w") as output_file:
        print(output, file=output_file)

    problem_tokens = lisp_parser.parse_nested_list(builtins.open(options.task))
    add_metric(problem_tokens)
    #TODO Prettier output for objects (no newlines)
    with open("problem-out.pddl", "w") as output_file:
        print(get_pddl_string(problem_tokens), file=output_file)

    
def transform_evmdd_actions(actions, mutex_groups, task):
    """
    Evmdd-based transformation of actions with sdac 
    into actions with constant action costs.
    """
    
    ret_actions = []
    del_actions = []
    add_actions = []
    mutex_dict = dict()
    aux_set = set()
    aux_set.add("evmdd-aux-init")
    
    for group in mutex_groups:
        for elem in group:
            mutex_dict[elem] = group
    for a in actions:
        if isinstance(a.cost, pddl_parser.CostNode):
            a.cost = a.cost.get_simplified_function()
            del_actions.append(a)
            if not is_usefull(a):
                continue
            mdd_str = a.cost.to_evmdd_str()[1:-1]
            cost_atoms = a.cost.get_atoms_set()
            atoms = a.cost.get_atoms_set()
            symbols = []
            i = 0
            function = a.cost.get_simplified_function_rek()
            for atom in atoms:
                symbol = "v_" + str(i)
                #symbol = str(atom)
                symbol = symbol.replace("(", "").replace(")", "").replace(" ", "").replace(",", "").replace("-", "")
                symbols.append(symbol)
                i += 1
                function = function.replace(str(atom), symbol)
            # EVMDD construction
            function_term = function
            print(function)
            var_names = symbols
            var_domains = dict()
            for s in symbols:
                var_domains[s] = 2
            evm, manager = term_to_evmdd(function_term, var_names=var_names,
                               var_domains=var_domains, fully_reduced=True)
            action_list = []
            build_evmdd_actions(evm, action_list, a, atoms, aux_set, mutex_dict)
            add_actions.extend(action_list)
            visualizer = EvmddVisualizer(manager)
            visualizer.visualize(evm)
    ret_actions = actions
    for a in del_actions:
        ret_actions.remove(a)
    ret_actions.extend(add_actions)
    task.inst_actions = ret_actions
    print_pddl(options.domain, [], task, aux_set)


def build_evmdd_actions_rek(evm, action_list, action, cost_atoms, aux_dict, mutex_dict, weight, val, first, parent_aux, parent_level, visited):
    aux_string = "evmdd-aux-" + action.name[1:-1].replace(" ", "-") + "-"
    is_first = first
    if str(type(evm)) == "<class 'evmdd.evmdd.Node'>":
        if evm in visited:
            return
        visited[evm] = True
        for i in range(0, len(evm.children)):
            build_evmdd_actions_rek(evm.children[i], action_list, action, cost_atoms, aux_dict, mutex_dict, weight, i, is_first, aux_dict[evm], evm.level, visited)
    elif str(type(evm)) == "<class 'evmdd.evmdd.Edge'>":
        if is_first:
            aux_atom = pddl.Atom(aux_string + aux_dict[evm.succ], [])
            action_copy = pddl.PropositionalAction(action.name[:-1] + "-init-", action.precondition, [], str(evm.weight))
            action_copy.add_effects.append(([], aux_atom))
            aux_atom_init = pddl.Atom("evmdd-aux-init", [])
            action_copy.add_effects.append(([], aux_atom_init))
            action_copy.precondition.append(aux_atom_init.negate())
            is_first = False
        else:
            if not evm.succ.is_sink_node():
                aux_atom = pddl.Atom(aux_string + aux_dict[evm.succ], [])
            aux_pre_atom = pddl.Atom(aux_string + parent_aux, [])
            pre_atom = cost_atoms[parent_level - 1]
            if val == 0:
                pre_atom = pddl.NegatedAtom(pre_atom.predicate, pre_atom.args)
            pre_copy = []
            pre_copy.append(aux_pre_atom)
            pre_copy.append(pre_atom)
            pre_atom_str = str(pre_atom.predicate)
            for a in pre_atom.args:
                pre_atom_str += "-" + str(a)
            pre_atom_str += "=" + str(val) + "-"
            action_copy = pddl.PropositionalAction(action.name[:-1] + "-" + pre_atom_str, pre_copy, [], str(evm.weight))
            if evm.succ.is_sink_node():
                action_copy.add_effects.extend(action.add_effects)
                action_copy.del_effects.append(([], aux_pre_atom))
                action_copy.del_effects.extend(action.del_effects)
                aux_atom_init = pddl.Atom("evmdd-aux-init", [])
                action_copy.del_effects.append(([], aux_atom_init))
            else:
                action_copy.add_effects.append(([], aux_atom))
                action_copy.del_effects.append(([], aux_pre_atom))
        action_list.append(action_copy)
        build_evmdd_actions_rek(evm.succ, action_list, action, cost_atoms, aux_dict, mutex_dict, evm.weight, -1, is_first, parent_aux, parent_level, visited)

def build_evmdd_actions(evm, action_list, action, cost_atoms, aux_set, mutex_dict):
    """
    Builds actions with constant action costs given an evmdd.
    """
    aux_string = "evmdd-aux-" + action.name[1:-1].replace(" ", "-") + "-"
    if not consistent_precondition(action.precondition, action.precondition):
        return []
    aux_dict = dict()
    i = 0
    for n in evm.nodes():
        aux_dict[n] = str(i)
        aux_set.add(aux_string + str(i))
        i += 1
    build_evmdd_actions_rek(evm, action_list, action, cost_atoms, aux_dict, mutex_dict, -1, -1, True, -1, -1, dict())
            
def print_pddl(domain_file, sas_task, task, aux_set):
    """
    Prints domain-out.pddl and problem-out.pddl given a parsed task.
    """
    domain_list = lisp_parser.parse_nested_list(builtins.open(domain_file))
    problem_list = lisp_parser.parse_nested_list(builtins.open(options.task))
    insert_aux_init(domain_list)
    insert_increase_total_cost(domain_list, task)
    with open("domain-out.pddl", "w") as output_file:
        out_str = get_pddl_str(domain_list, 0, sas_task, task, aux_set)
        n = out_str.count("#!seperator!#")
        out_str = out_str.replace("#!seperator!#", "", n-1)
        out_str = out_str.replace("#!seperator!#", get_action_str(sas_task, task), 1)        
        
        print(out_str, file=output_file)

    with open("problem-out.pddl", "w") as output_file:
        insert_min_metric(problem_list)
        out_str = get_pddl_str(problem_list, 0, sas_task, task, aux_set)
        print(out_str, file=output_file)
    
def get_pddl_str(domain_list, depth, sas_task, task, aux_set):
    out_str = ""
    out_str += "("
    if len(domain_list) > 0 and domain_list[0] == ":action":
        if is_exp_action(domain_list[1], sas_task, task):
            return "#!seperator!#\n" 
    for entry in domain_list:
        if isinstance(entry, str):
            out_str += entry
            if entry == ":requirements":
                out_str += " :action-costs  "
            if entry == ":predicates":
                out_str += " "
                if options.evmdd:
                    for aux in aux_set:
                        out_str += "(" + aux + ")" + "\n"
            if(domain_list.index(entry) < len(domain_list) - 1):
                out_str += " "
        else:
            out_str += get_pddl_str(entry, depth + 1, sas_task, task, aux_set)
    out_str += ")"
    out_str += "\n"
    return out_str
    
def insert_increase_total_cost(domain_list, task):
    if task.use_min_cost_metric:
        return
    if isinstance(domain_list, str) and domain_list == ":effect":
        return True
    elif isinstance(domain_list, list):
        for index in range(0, len(domain_list)):
            if insert_increase_total_cost(domain_list[index], task) and len(domain_list) - 1 > index:
                if isinstance(domain_list[index + 1], list):
                    domain_list[index + 1].append("(increase (total-cost) 1)")
    
def insert_aux_init(domain_list):
    if not options.evmdd:
        return
    if isinstance(domain_list, str) and domain_list == ":precondition":
        return True
    elif isinstance(domain_list, list):
        for index in range(0, len(domain_list)):
            if insert_aux_init(domain_list[index]) and len(domain_list) - 1 > index:
                if isinstance(domain_list[index + 1], list):
                    domain_list[index + 1].append("(not (evmdd-aux-init))")

def insert_min_metric(problem_list):
    if [':metric', 'minimize', ['total-cost']] not in problem_list:
        problem_list.append("(:metric minimize(total-cost))") 

def is_exp_action(name, sas_task, task):
    if options.evmdd:
        for pddl_op in task.inst_actions:
            if isinstance(pddl_op.cost, str):
                return pddl_op.name.split(" ")[0][1:] == name
    else:
        for sas_op in sas_task.operators:
            if isinstance(sas_op.cost, str) and name in sas_op.name:
                return True
    return False

def get_action_str(sas_task, task):
    out_str = ""
    if options.evmdd:
        sas_operator_names = []
    else:
        sas_operator_names = [op.name[1:-1] for op in sas_task.operators]
    for pddl_op in task.inst_actions:
        pddl_op_name = pddl_op.name[1:-1]
        if isinstance(pddl_op.cost, str):
            if options.evmdd or pddl_op_name in sas_operator_names:
                out_str += "(:action " + pddl_op_name.replace(" ", "-") + "-" + pddl_op.cost
                out_str += "\n" + ":parameters ()"
                precon_str = "\n:precondition (and "
                eff_list = []
                for atom in pddl_op.precondition:
                    if atom.negated:
                        precon_str += "(not ("
                    else:
                        precon_str += "("
                    precon_str += atom.predicate + " "
                    for arg in atom.args:
                        precon_str += arg + " "
                    if atom.negated:
                        precon_str += "))"
                    else:
                        precon_str += ")"
                for eff in pddl_op.add_effects:
                    eff_list.append(eff[1])
                for eff in pddl_op.del_effects:
                    eff_list.append(pddl.NegatedAtom(eff[1].predicate, eff[1].args))
                out_str += precon_str + ")"
                eff_str = "\n:effect (and "
                for atom in eff_list:
                    if atom.negated:
                        eff_str += "(not ("
                    else:
                        eff_str += "("
                    eff_str += atom.predicate + " "
                    for arg in atom.args:
                        eff_str += arg + " "
                    if atom.negated:
                        eff_str += "))"
                    else:
                        eff_str += ")"
                out_str += eff_str + "(increase (total-cost) " + pddl_op.cost + "))"
                out_str += ")\n"
                
            
    return out_str
    
def is_consistent(combi, precondition, mutex_dict):
    """
    Check if a possible atom configuration is consistent with
    the precondition of an action
    """
    for ac in combi:
        if "evmdd-aux" in ac.predicate:
            continue
        ac_g = ac
        ac_pos = True
        if ac.negated:
            ac_g = pddl.Atom(ac_g.predicate, ac_g.args)
            ac_pos = False
        for ap in precondition:
            if "evmdd_aux" in ap.predicate:
                continue
            ap_g = ap
            ap_pos = True
            if ap.negated:
                ap_g = pddl.Atom(ap_g.predicate, ap_g.args)
                ap_pos = False
            if ap_g == ac_g and ap_pos != ac_pos:
                return False
            if ap_g != ac_g and mutex_dict[ap_g] == mutex_dict[ac_g] and ap_pos == ac_pos:
                return False
    return True

def is_useful(action):
    """
    Check if action is changing the worldstate:
    If the effect is a subset of the precondition
    the action is useless.
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
    
def is_usefull(action):
    """
    Check if action is changing the worldstate:
    If the effect is a subset of the precondition
    the action is useless.
    """
    matches = 0
    for atom1 in action.precondition:
        if atom1.negated:
            negated1 = True
        else:
            negated1 = False
        atom1_n = pddl.Atom(atom1.predicate, atom1.args)
        for con, atom2 in action.add_effects:
            if not negated1 and atom1 == atom2:
                matches +=1
        for atom2 in action.del_effects:
            if negated1 and atom1 == atom2:
                matches +=1
    if matches >= len(action.del_effects) + len(action.add_effects):
        return False
    return True
    
def consistent_precondition(precondition, mutex_dict):
    """
    Check if the precondition of an action is satisfiable.
    """
    for atom1 in precondition:
        if atom1.negated:
            negated1 = True
        else:
            negated1 = False
        atom1_n = pddl.Atom(atom1.predicate, atom1.args)
        
        for atom2 in precondition:
            if atom2.negated:
                negated2 = True
            else:
                negated2 = False
            atom2_n = pddl.Atom(atom2.predicate, atom2.args)
            if atom1_n == atom2_n and negated1 != negated2:
                return False
    return True
