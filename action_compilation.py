#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from evmdd import term_to_evmdd, EvmddVisualizer

class Fact(object):
    def __init__(self, var, val):
        self.var = var
        self.val = val

    def __repr__(self):
        return 'Fact(var=%s, val=%s)' % (repr(self.var), repr(self.val))

    def __str__(self):
        return '%s=%s' % (str(self.var), str(self.val))


class Action(object):
    def __init__(self, name, pre, eff, cost):
        self.name = name
        self.pre = pre
        self.eff = eff
        self.cost = cost

    def __repr__(self):
        return ('Action(name=%s, pre=%s, eff=%s, cost=%s)' %
                (repr(self.name), repr(self.pre), 
                    repr(self.eff), repr(self.cost)))

    def __str__(self):
        pre = ', '.join([str(p) for p in self.pre])
        eff = ', '.join([str(e) for e in self.eff])
        return '([%s], [%s], %s) %s' % (pre, eff, self.cost, self.name)


class ActionCompiler(object):

    def __init__(self, semaphore='sem', var_names='AUTOMATIC',
                 var_domains=None, fully_reduced=True):
        self._semaphore = semaphore
        self._var_names = var_names
        self._var_domains = var_domains
        self._fully_reduced = fully_reduced
        self._name_index = 0

    def _initialize(self, action, auxvar):
        substitution = {fact.var: fact.val for fact in action.pre}
        self._action = action
        self._auxvar = auxvar
        self._name_index = 0
        self._evmdd, self._manager = term_to_evmdd(
                                         self._action.cost,
                                         var_names=self._var_names,
                                         var_domains=self._var_domains,
                                         fully_reduced=self._fully_reduced,
                                         substitution=substitution)

    def _topsort_evmdd(self):
        self._topsorted_nodes = sorted(self._evmdd.nodes(),
                                       key = lambda node: node.level,
                                       reverse=True)
        self._topsort_idx = {node: idx+1 for idx, node in
                             enumerate(self._topsorted_nodes)}

    def _make_precondition_action(self):
        return Action(self._action.name + "-pre",
                      pre=list(self._action.pre) + [Fact(self._auxvar, 0),
                                                    Fact(self._semaphore, 0)],
                      eff=[Fact(self._auxvar, 1), Fact(self._semaphore, 1)],
                      cost=self._evmdd.weight)

    def _make_intermediate_action(self, node, var_val, edge):
        var_name = self._manager._level_to_var_name(node.level)
        name = self._action.name + "-" + var_name + " is " + str(bool(var_val)) + "_" + str(self._topsort_idx[node])
        return Action(name,
                      pre=[Fact(self._auxvar, self._topsort_idx[node]),
                           Fact(var_name, var_val)],
                      eff=[Fact(self._auxvar, self._topsort_idx[edge.succ])],
                      cost=edge.weight)

    def _make_effect_action(self, node):
        return Action(self._action.name + "-eff",
                      pre=[Fact(self._auxvar, self._topsort_idx[node])],
                      eff=list(self._action.eff) + [Fact(self._auxvar, 0),
                                                    Fact(self._semaphore, 0)],
                      cost=0)

    def action_compilation(self, action, auxvar, visualize = False):
        self._initialize(action, auxvar)
        self._topsort_evmdd()

        if visualize and not self._evmdd.succ.is_sink_node():
            with EvmddVisualizer(self._manager) as visualizer:
                visualizer.visualize(self._evmdd, action.name)

        # Actions with constant costs need not be split
        # and only have to add the semaphore
        if self._evmdd.succ.is_sink_node():
            copy = action
            copy.pre.append(Fact(self._semaphore, 0))
            return [copy]

        result = [self._make_precondition_action()]
        for node in self._topsorted_nodes:
            for var_val, edge in enumerate(node.children):
                self._name_index += 1
                result.append(self._make_intermediate_action(node, var_val, edge))
            if node.is_sink_node():
                result.append(self._make_effect_action(node))
        return result



if __name__ == '__main__':
    action1 = Action('action1',[Fact('D',1), Fact('E',2)], [Fact('F',1), Fact('D', 0)], 'A*B**2 + C + 2')
    action2 = Action('action2',[Fact('D',0)], [Fact('E', 1)], '10-F')
    action3 = Action('action3',[Fact('D',0)], [Fact('E', 1)], '5')
    action4 = Action('action4',[Fact('D',1), Fact('B',3)], [Fact('F',1), Fact('D', 0)], 'A*B**2 + C + 2')
    actions = [action1, action2, action3, action4]
    auxvar = 'α'
    #auxvars = ['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ',
    #           'ν', 'ξ', 'ο', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω']

    for action in actions:
        print(action)
    print('=========================================')

    var_names = 'AUTOMATIC'
    var_domains = {'A': 2, 'B': 3, 'C': 2, 'F': 5}
    compiler = ActionCompiler(semaphore='Σ', var_names=var_names,
                              var_domains=var_domains, fully_reduced=True)
    compilations = [compiler.action_compilation(action, '%s%s' % (auxvar, auxidx))
                    for auxidx, action in enumerate(actions)]
    #compilations = [compiler.action_compilation(action, auxvar)
    #                for action, auxvar in zip(actions, auxvars)]

    for compilation in compilations:
        for action in compilation:
            print(action)
        print('-----------------------------------------')
    print('=========================================')

    compiled_actions = [action for compilation in compilations for action in compilation]
