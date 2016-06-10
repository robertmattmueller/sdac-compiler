import pddl
from pddl import conditions
import itertools
import ast
from sympy import *

class LogicalOperator:
    """
    Represents logical operators of a cost function:
    and, or, not
    """

    def __init__(self, op_type):
        self.and_op = False
        self.or_op = False
        self.not_op = False
        if op_type == "and":
            self.and_op = True
        if op_type == "or":
            self.or_op = True
        if op_type == "not":
            self.not_op = True
    
    def __str__(self):
        if self.and_op:
            return "and"
        if self.or_op:
            return "or"
        if self.not_op:
            return "not"

class OperatorType():
    """
    Represents arithmetic operators of a cost function:
    +, -, *
    """
    def __init__(self, op_type):
        self.add = False
        self.sub = False
        self.mul = False
        self.const = False
        if op_type == "+":
            self.add = True
        if op_type == "-":
            self.sub = True;
        if op_type == "*":
            self.mul = True;
        if op_type == "c":
            self.const = True;

class CostNode(object):
    """
    Treestructure that represents a cost function.
    """
    def __init__(self, operator):
        self.operator = operator
        self.children = []
        self.is_parameter = False
        if isinstance(operator, pddl.Atom):
            self.is_parameter = True
        self.string = ""
        self.isConstant = False
        self.isUsed = False
        self.del_init_true = False
        self.parent = None
        self.unbound_pars = []
        self.is_multi_op = False
        self.bound_pars = []
        self.ordering = []
        self.unbound_check = []
        self.cost = 0
        self.ar_helper = False
        self.power = 1
        
    def __str__(self):
        self.string = ""
        self.to_string_rek(self, self)
        return self.string
        
    def to_string_rek(self, init, parent):
        if self.is_parameter and not self.isUsed:
            if self.del_init_true:
                init.string += "($ 1"
            else:
                init.string += "($ 0"
        else:
            if isinstance(self.operator, OperatorType) and self.operator.add:
                init.string += "(+"
            elif isinstance(self.operator, OperatorType) and self.operator.mul:
                init.string += "(*"
            elif isinstance(self.operator, OperatorType) and self.operator.sub:
                init.string += "(-"
            else:
                init.string += "(" + self.operator.__str__()
        for c in self.children:
            c.to_string_rek(init, self)
        init.string += ")"
    
    def insert_atoms(self, atoms, symbols):
        try:
            i = symbols.index(str(self.operator))
            self.operator = atoms[i]
        except:
            pass
        for c in self.children:
            c.insert_atoms(atoms, symbols)
    
    def insert_brackets(self, function):
        new_function = ""
        last_op = ""
        last_op_index = -1
        i = 0
        while i < len(function):
            if function[i] == "+" or function[i] == "-":
                if last_op == "*" or last_op == "/":
                    new_function += ")"
                new_function += function[i]
                last_op = function[i]
                last_op_index = len(new_function) #+ 1
            elif function[i] == "*" or function[i] == "/":
                if last_op == "+" or last_op == "-":
                    new_function = new_function[:last_op_index] + "(" + new_function[last_op_index:]
                elif last_op == "":
                    new_function = "(" + new_function[0:]
                new_function += function[i]
                last_op = function[i]
                last_op_index = len(new_function) #+ 1
            elif function[i] == "(":
                f, add_i =  self.insert_brackets(function[i + 1:])
                new_function += "(" + f + ")"
                i += add_i + 1
            elif function[i] == ")":
                if last_op == "*" or last_op == "/":
                    new_function += ")"
                yield new_function
                yield i
                return
            else:
                new_function += function[i]
            i += 1
        if last_op == "*" or last_op == "/":
            new_function += ")"
        yield new_function
        yield i
    
    def get_simplified_function(self):
        """
        Simplify cost function using sympy.
        """
        atoms = self.get_atoms_set()
        symbols = []
        i = 0
        function = self.get_simplified_function_rek()
        for atom in atoms:
            symbol = "v_" + str(i)
            symbols.append(symbol)
            i += 1
            function = function.replace(str(atom), symbol)
        function = str(simplify(function)).replace("C_", "$ ").replace(" ", "")
        function, i = self.insert_brackets(function)
        function = function.replace("(", "[").replace(")", "]")
        function = self.make_parsable(function)
        function = ast.literal_eval(function)
        if isinstance(function, str):
            function = [function]
        function = self.str_to_costnode(function, atoms, symbols)
        function.insert_atoms(atoms, symbols)
        return function
        
    def str_to_costnode(self, function, atoms, symbols):
        total_add = 0
        total_sub = 0
        total_mul = 0
        sub = ""
        pos = []
        neg = []
        for el in function:
            if isinstance(el, list):
                sub_node = self.str_to_costnode(el, atoms, symbols)
                if self.get_last_operator(sub) == "-":
                    neg.append(sub_node)
                else:
                    pos.append(sub_node)
            else:
                total_mul += el.count("*")
                total_add += el.count("+")
                total_sub += el.count("-")
                el = el.replace("+", " + ").replace("*", " * ").replace("-", " - ")
                el = el.replace("*  *", "**")
                el_list = el.split(" ")
                el_sub = ""
                for e in el_list:
                    el_sub += e
                    if len(e) > 0 and not("+" in e or "-" in e or "*" in e):
                        if self.get_last_operator(el_sub) == "**":
                            assert(e.isdigit())
                            sub_node.power = int(e)
                            continue
                        sub_node = CostNode(e)
                        sub_node.is_parameter = True
                        sub_node.isUsed = True
                        if self.get_last_operator(el_sub) == "+" or self.get_last_operator(el_sub) == "*":
                            pos.append(sub_node)
                        elif self.get_last_operator(el_sub) == "-":
                            neg.append(sub_node)
                        elif self.get_last_operator(sub) == "+" or self.get_last_operator(el_sub) == "*":
                            pos.append(sub_node)
                        elif self.get_last_operator(sub) == "-":
                            neg.append(sub_node)
                        else:
                            pos.append(sub_node)
                        if sub_node.operator.isdigit():
                            sub_node.operator = "$ " + sub_node.operator
                            sub_node.isConstant = True
                            sub_node.is_parameter = False
                sub += el
        assert(not(total_mul > 0 and total_sub + total_add > 0))
        if total_mul > 0:
            root_node = CostNode(OperatorType("*"))
            for p in pos:
                for i in range(1, p.power + 1):
                    root_node.children.append(p)
        else:
            if len(pos) == 0:
                root_node = CostNode(OperatorType("-"))
                zero_node = CostNode("$ 0")
                root_node.children.append(zero_node)
                for n in neg:
                    root_node.children.append(n)
            elif len(neg) > 0:
                root_node = CostNode(OperatorType("-"))
                if len(pos) > 0:
                    pos_node = CostNode(OperatorType("+"))
                    for p in pos:
                        pos_node.children.append(p)
                else:
                    pos_node = pos[0]
                root_node.children.append(pos_node)
                for n in neg:
                    root_node.children.append(n)
            else:
                root_node = CostNode(OperatorType("+"))
                for p in pos:
                    root_node.children.append(p)
        return root_node
    
    def get_last_operator(self, sub):
        sub_rev = sub[::-1]
        for i in range(0, len(sub_rev)):
            if sub_rev[i] == "*":
                if i < len(sub_rev) - 1:
                    if sub_rev[i+1] == "*":
                        return "**"
                return sub_rev[i]
            elif sub_rev[i] == "-" or sub_rev[i] == "+":
                return sub_rev[i]
        return None
    
    def make_parsable(self, function):
        ret_function = ""
        part = ""
        for i in range(0, len(function)):
            if function[i] == "[":
                if part != "":
                    ret_function += "'" + part + "'," + function[i] 
                else:
                    ret_function += function[i]
                part = ""
            elif function[i] == "]":
                if part != "":
                    ret_function += "'" + part + "'" + function[i]        
                else:
                    ret_function += function[i]
                if i + 1 < len(function):
                        if function[i + 1] != "]":
                            ret_function += ","
                part = ""
            else:
                part += function[i]
        if part != "":
            ret_function += "'" + part + "'"
        return ret_function

    def get_simplified_function_rek(self):
        ret_str = ""
        if isinstance(self.operator, OperatorType):
            ret_str += "("
            for c in self.children:
                if self.operator.add:
                    ret_str += c.get_simplified_function_rek() + "+"
                elif self.operator.mul:
                    ret_str += c.get_simplified_function_rek() + "*"
                elif self.operator.sub:
                    ret_str += c.get_simplified_function_rek() + "-"
            ret_str = ret_str[:-1]
            ret_str += ")"
        else:
            if self.is_parameter:
                ret_str = str(self.operator)
            else:
                ret_str = str(self.operator.replace("$ ", ""))
        return ret_str
        
    def to_evmdd_str(self):
        self.string = ""
        self.to_evmdd_str_rek(self, self)
        return self.string

    def to_evmdd_str_rek(self, init, parent):
        if isinstance(self.operator, OperatorType):
            if self.operator.add:
                init.string += "(+ "
            elif self.operator.mul:
                init.string += "(* "
            elif self.operator.sub:
                init.string += "(- "
        elif not self.isConstant:
            init.string += "($s(" + self.operator.__str__() + ")"
        else:
            init.string += "(" + self.operator.__str__().replace("$", "").replace("C_", "")
        for c in self.children:
            c.to_evmdd_str_rek(init, self)
        init.string += ")"

    def add_child(self, child):
        children.append(child)
    
    def set_children(self, children):
        self.children = children
        
    def dump(self):
        print(self.operator)
        for c in self.children:
            c.dump()
    
    def get_atoms_rek(self, atom_list):
        if self.is_parameter:
            atom_list.append(self.operator)
        else:
            for c in self.children:
                c.get_atoms_rek(atom_list)
                
    def get_atoms(self):
        atom_list = []
        self.get_atoms_rek(atom_list)
        return atom_list
    
    def get_atoms_set(self):
        atom_set = []
        self.get_atoms_set_rek(atom_set)
        return atom_set
        
    def get_atoms_set_rek(self, atom_set):
        if self.is_parameter and self.operator not in atom_set:
            atom_set.append(self.operator)
        else:
            for c in self.children:
                c.get_atoms_set_rek(atom_set)
      
    def get_predicates_rek(self, pred_list):
        if isinstance(self.operator, pddl.Predicate):
            pred_list.append(self.operator)
        else:
            for c in self.children:
                c.get_predicates_rek(pred_list)
                
    def get_predicates(self):
        pred_list = []
        self.get_predicates_rek(pred_list)
        return pred_list
        
    def get_parameters(self):
        parameters = []
        self.get_parameters_rek(parameters)
        return parameters
        
    def get_parameters_rek(self, parameters):
        if self.args != None:
            for a in self.args:
                if a not in parameters:
                    parameters.append(a)
        for c in self.children:
            c.get_parameters_rek(parameters)
            
    def copy(self):
        if isinstance(self.operator, pddl.Atom):
            node = CostNode(pddl.Atom(self.operator.predicate, self.operator.args))
        else:
            node = CostNode(self.operator)
        node.unbound_pars = self.unbound_pars
        node.parent = self.parent
        node.is_multi_op = self.is_multi_op
        node.bound_pars = self.bound_pars
        node.ordering = self.ordering
        node.isConstant = self.isConstant
        self.copy_rek(node.children)
        return node
        
    def copy_rek(self, children):
        
        for c in self.children:
            if isinstance(c.operator, pddl.Atom):
                node = CostNode(pddl.Atom(c.operator.predicate, c.operator.args))
            else:
                node = CostNode(c.operator)
            node.parent = self
            node.unbound_pars = c.unbound_pars
            node.is_multi_op = c.is_multi_op
            node.bound_pars = c.bound_pars
            node.ordering = c.ordering
            node.isConstant = c.isConstant
            children.append(node)
            c.copy_rek(node.children)
    
    def get_cost(self, state):
        """
        Returns cost value given a worldstate.
        """
        if self.is_parameter:
            if self.operator in state:
                return 1
            else:
                return 0
        if self.isConstant:
            return int(self.operator.replace("$ ", ""))
            
        if isinstance(self.operator, OperatorType) and self.operator.mul:
            res = 1
        else:
            res = 0
        for c in self.children:
            if isinstance(self.operator, OperatorType) and self.operator.add:
                res += c.get_cost(state)
            if isinstance(self.operator, OperatorType) and self.operator.mul:
                res *= c.get_cost(state) 
            if isinstance(self.operator, OperatorType) and self.operator.sub:
                res -= c.get_cost(state)
        return res
    
    def instantiate_rek(self, var_mapping, init_facts, fluent_facts,
            objects_by_type, result):
        objects_product = None
        objects_list = []
        unbound_pars = []
        if self.is_parameter:
            args = list(self.operator.args)
            for index in range(0, len(args)):
                if args[index] in self.ordering:
                    i = self.ordering.index(args[index])
                    args[index] = self.bound_pars[i]
            at = pddl.Atom(self.operator.predicate, args)
            try:
                at.instantiate(var_mapping, init_facts, fluent_facts, result)
            except conditions.Impossible:
                self.operator = "$ 0"
                self.isConstant = True
                self.is_parameter = False
                return
            self.operator = at
                
        elif self.is_multi_op:
            for par in self.unbound_pars:
                parl = par.split(" - ")
                assert(len(parl) <= 2)
                par_name = parl[0]
                par_type = "object"
                if len(parl) > 1:
                    par_type = parl[1]
                objects = objects_by_type.get(par_type)
                objects_list.append(objects)
                unbound_pars.append(par_name)
            child = self.children[0]
            self.children.remove(child)
            for combi in itertools.product(*objects_list):
                new_child = child.copy()
                new_child.bound_pars = list(combi)
                new_child.bound_pars.extend(self.bound_pars)
                new_child.ordering = list(unbound_pars)
                new_child.ordering.extend(self.ordering)
                self.children.append(new_child)
        else:
            for c in self.children:
                c.bound_pars = self.bound_pars
                c.ordering = self.ordering  
                
        for c in self.children:
            #c.instantiate_rek(objects_by_type)
            c.instantiate_rek(var_mapping, init_facts, fluent_facts,
                objects_by_type, result)

    def instantiate(self, var_mapping, init_facts, fluent_facts,
            objects_by_type, result):
        """
        Replace predicate-objects tuples with corresponding atoms.
        """
        cost = self.copy()
        atoms = cost.get_atoms()
        args = []
        for a in atoms:
            for arg in a.args:
                if var_mapping.get(arg):
                    args.append(var_mapping.get(arg))
                else:
                    args.append(arg)
            if args:
                a.args = tuple(args)
            args = []

        cost.instantiate_rek(var_mapping, init_facts, fluent_facts,
                objects_by_type, result)
        return cost
        
    def get_math_nodes_rek(self, node_list):
        if isinstance(self.operator, OperatorType) and (self.operator.add or self.operator.mul or self.operator.sub):
            node_list.append(self)
        for c in self.children:
            c.get_math_nodes_rek(node_list)
                
    def get_math_nodes(self):
        """
        Return all cost nodes that have an arithmetic operator.
        """
        node_list = []
        self.get_math_nodes_rek(node_list)
        return node_list
    
    def make_unused(self):
        self.isUsed = False
        for c in self.children:
            c.make_unused()
    
    def to_sas(self, variables, atom_dict, deleted_true_variables):
        """
        Replace pddl atoms with SAS variable-value pairs
        """
        self.make_unused()
        for atom in self.get_atoms():
            delin = False
            found = False
            for index in range(0, len(variables)):
                if str(atom) in str(variables[index]):
                    node = self.get_node(atom)
                    node.operator = str(index) + " " + str(variables[index].index(atom))
                    node.isUsed = True;
                    found = True
                    break
                else:
                    for i in range(0, len(deleted_true_variables)):
                        if str(atom) in deleted_true_variables[i]:
                            node = self.get_node(atom)
                            node.del_init_true = True 
                            delin = True
            if not found and not delin:
                raise ValueError(atom, "is not a valid fact")
        for pre in self.get_predicates():
            node = self.get_node(pre)
            node.operator = pre.cost_str
            node_list = []
            for index in range(0, len(variables)):  
                for j in range(0, len(variables[index])):
                    if not variables[index][j].negated:
                        at_str = variables[index][j]
                        at_list = at_str.split("Atom ")
                        at_list = at_list[1].split("(")
                        if pre.name in at_list[0]:
                            child = CostNode(str(index) + " " + str(j))
                            node_list.append(child)
                    
            for i in range(0, len(deleted_true_variables)):
                at_str = deleted_true_variables[i][0]
                at_list = at_str.split("Atom ")
                at_list = at_list[1].split("(")
                if pre.name in at_list[0]:
                    child = CostNode("$ 1")
                    node_list.append(child)
            node.children = node_list
                        
                    
    def get_node(self, atom):
        if self.operator is atom:
            return self
        for c in self.children:
            node = None
            node = c.get_node(atom)
            if node != None:
                return node

    def transform_logic(self):
        """
        Transorm nodes with logical operators with into nodes
        with corresponding arithmetic operators.
        """
        self.transform_logic_rek(False)
    
    def transform_logic_rek(self, logic_subtree):
        logic = logic_subtree
        if isinstance(self.operator, LogicalOperator):
            logic = True
            if self.operator.not_op:
                self.operator = OperatorType("-")
                new_children = []
                new_op = CostNode("$ 1")
                new_op.isConstant = True
                new_children.append(new_op)
                new_children.append(self.children[0])
                self.children = new_children
            elif self.operator.and_op:
                if len(self.children) == 0:
                    self.operator = "$ 0"
                    self.isConstant
                elif len(self.children) == 1:
                    self.operator = self.children[0].operator
                    self.is_parameter = self.children[0].is_parameter
                    self.children = self.children[0].children
                else:
                    self.operator = OperatorType("*")
            elif self.operator.or_op:
                if len(self.children) == 0:
                    self.operator = "$ 0"
                    self.isConstant = True
                elif len(self.children) == 1:
                    self.operator = self.children[0].operator
                    self.is_parameter = self.children[0].is_parameter
                    self.children = self.children[0].children
                else:
                    self.operator = OperatorType("-")
                    new_children = []
                    add_node = CostNode(OperatorType("+"))
                    add_node.ar_helper = True
                    add_rest = CostNode(LogicalOperator("or"))
                    add_node.children.append(self.children[0])
                    add_node.children.append(add_rest)
                    add_rest.children = self.children[1:]
                    mul_node = CostNode(OperatorType("*"))
                    mul_node.ar_helper = True
                    mul_rest = CostNode(LogicalOperator("or"))
                    for i in range(1, len(self.children)):
                        c_copy = self.children[i].copy()
                        mul_rest.children.append(c_copy)
                    mul_node.children.append(self.children[0].copy())
                    mul_node.children.append(mul_rest)
                    new_children.append(add_node)
                    new_children.append(mul_node)
                    self.children = new_children
                
        elif isinstance(self.operator, OperatorType) and logic_subtree and not self.ar_helper:
            raise ValueError("Arithmetic expression in logical formula")     
        
        for c in self.children:
            c.transform_logic_rek(logic)
         
    def parse_cost(self, alist, outer_op, predicate_dict, parameters_dict, parent):
        """
        Parse cost function from a nested list.
        """
        iterator = iter(alist)
        tag = alist[0]
        if(tag.startswith("?")):
            self.set_unbound_pars(alist, parent)
            return
        try:
            float(tag)
        #if(tag.isdigit()):
            outer_op = CostNode("$ " + tag)
            outer_op.parent = parent
            outer_op.isConstant = True
            return outer_op
        except:
            pass
        if tag == "and" or tag == "or" or tag == "not" or tag == "forall" or tag == "exists":
            if tag == "not" and len(alist[1:]) != 1:
                raise ValueError("Operator", tag, "must have 1 arguments.")
            if tag == "forall":
                outer_op = CostNode(LogicalOperator("and"))
            elif tag == "exists":
                outer_op = CostNode(LogicalOperator("or"))
            else:
                outer_op = CostNode(LogicalOperator(tag))
            outer_op.parent = parent
            outer_op.set_children(
            [self.parse_cost(cost_arg, None, predicate_dict, parameters_dict, outer_op) for cost_arg in alist[1:]])
            if outer_op.children[0] == None:
                outer_op.children.remove(outer_op.children[0])
            if(outer_op.is_multi_op):
                assert(len(outer_op.children) == 1)
                if not (tag == "forall" or tag == "exists"):
                    raise ValueError(tag, "cant have unbound variables")
            elif tag == "forall" or tag == "exists":
                raise ValueError(tag, "must have at least one unbound varaible")
            return outer_op
        if tag == "sum" or tag == "prod" or tag == "+" or tag == "-" or tag == "*":
            if tag == "sum":
                outer_op = CostNode(OperatorType("+"))
            elif tag == "prod":
                outer_op = CostNode(OperatorType("*"))
            else:
                outer_op = CostNode(OperatorType(tag))
            outer_op.parent = parent
            outer_op.unbound_check = list(parent.unbound_check)
            outer_op.set_children(
            [self.parse_cost(cost_arg, None, predicate_dict, parameters_dict, outer_op) for cost_arg in alist[1:]])
            if outer_op.children[0] == None:
                outer_op.children.remove(outer_op.children[0])
            if outer_op.is_multi_op:
                assert(len(outer_op.children) == 1)
                if not (tag == "sum" or tag == "prod"):
                    raise ValueError(tag, "cant have unbound variables")
            elif tag == "sum" or tag == "prod":
                raise ValueError(tag, "must have at least one unbound varaible")
            return outer_op
        else:
            atom = self.get_atom_parameter(alist, predicate_dict, parameters_dict, parent)
            outer_op = CostNode(atom)
            outer_op.parent = parent
            return outer_op
    
    def set_unbound_pars(self, alist, parent):
        """
        Initiate unbound variables of a cost term.
        """
        par_str = ""
        for tag in alist:
            if tag.startswith("?"):
                par_str = tag
                parent.unbound_pars.append(par_str)
                parent.unbound_check.append(par_str)
                parent.is_multi_op = True
            else:
                parent.unbound_pars[len(parent.unbound_pars) - 1] += " " + tag
                parent.unbound_check[len(parent.unbound_check) - 1] += " " + tag

    def get_atom_parameter(self, alist, predicate_dict, parameters_dict, parent):
        """
        Initiate Atoms of a cost-term
        """
        par = None
        pre_arg = 0
        args = []
        alist_save = list(alist)
        try:
            pre = predicate_dict[alist[0]]
        except:
            if(alist[0] not in parent.unbound_check):
                raise ValueError(alist[0] , " is not a predicate or parameter")
        alist = alist[1:]
        for index in range(0, len(alist)):
            par = parameters_dict.get(alist[index])
            for un_par in parent.unbound_check:
                parl = un_par.split(" - ")
                assert(len(parl) <= 2)
                par_str1 = parl[0]
                par_str2 = "object"
                if len(parl) > 1:
                    par_str2 = parl[1]
                if alist[index] == par_str1:
                    par = pddl.TypedObject(par_str1, par_str2)
                    break
            if not par:
                par = alist[index]
            if not isinstance(par, str) and par.type_name != pre.arguments[index].type_name:
                raise ValueError("Predicate", pre.name, "must have argument of type", pre.arguments[index].type_name, "at index", index ,". Got" , par.type_name)
            pre_arg += 1
            if isinstance(par, str):
                args.append(par)
            else:
                args.append(par.name)
        if pre_arg != pre.get_arity():
            raise ValueError("Number of arguments of predicate", pre.name , "does not match the number of arguments in costfunction")
        return pddl.Atom(pre.name, args)
    
