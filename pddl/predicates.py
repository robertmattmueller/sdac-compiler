class Predicate(object):
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments
        self.value_mapping = dict()
        self.cost_str = ""

    def __str__(self):
        return "%s(%s)" % (self.name, ", ".join(map(str, self.arguments)))

    def get_arity(self):
        return len(self.arguments)
    
    def get_arguments(self):
        return self.arguments
        
    def add_value(self, args, values):
        self.value_mapping[args] = values
