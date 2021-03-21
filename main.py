import doctest
import sys
from exceptions import SnekError, SnekSyntaxError, SnekNameError, SnekEvaluationError


# Tokenize and Parse
############################

def tokenize(user_input):
    '''
    Splits the user's input to meaningfull items. These include parens,
    functions, ints, etc...
    Arguments:
        user_input (str): user expression input
    '''
    #print('user_input: {}'.format(user_input))
    tokenized = []
    # seperate into list of lines
    user_input = user_input.split('\n')
    # in each line, only keep the part that is before the semicolon
    for line in user_input:
        if ';' in line:
            semicol_index = line.index(';')
            tokenized.append(line[:semicol_index])
        else:
            tokenized.append(line)
    # join all these parts of the lines you are keeping, and add spaces around the brackets
    tokenized = ' '.join(tokenized)
    tokenized = tokenized.replace('(', ' ( ')
    tokenized = tokenized.replace(')', ' ) ')
    # split this string up between the spaces to get the individual 'tokens'
    tokenized = tokenized.split()
    #print('tokenized:  {}'.format(tokenized))
    return tokenized

def parse(tokens):
    '''
    Parses a list of tokens produced form tokenize where:
        - symbols represented in strings
        - numbers are converted ints or floats
        - S-expressions are represented as Python lists
    Calls a recursive parse_expression function.
    Arguments:
        tokens (list): a list of strings that are tokens
    '''
    print('***NEXT INPUT***: {}'.format(tokens))

    def parse_s_expression(index, tokens):
        # cur_tok is the curent token you are analyzing
        cur_tok = tokens[index]
        # bracket mean S-expressino so: create a list for the things in this s-expression
        s_expression = []
        index += 1
        item = None
        # once it is an open bracket now we keep adding items to this s_expression until we find a closing bracket
        while tokens[index:index + 1] and tokens[index:index + 1] != [')']:
            item, index = parse_expression(index)
            print(f'new element added index is now {index}')
            s_expression.append(item)
        # at this point you have completed the calculation of the s expression
        # now check if its define or lambda
        if s_expression:
            if s_expression[0] == 'define':
                if len(s_expression) != 3:
                    raise SnekSyntaxError(f'{s_expression[0]} must have 3 components, not {len(s_expression)}')
                if not (s_expression[1] and isinstance(s_expression[1], (str, list)) and all(isinstance(i, str) for i in s_expression[1])):
                    raise SnekSyntaxError('Second element in define must be either a name or a list')
            if s_expression[0] == 'lambda':
                if len(s_expression) != 3:
                    raise SnekSyntaxError(f'{s_expression[0]} must have 3 components, not {len(s_expression)}')
                if (not isinstance(s_expression[1], list)) or (not all(isinstance(i, str) for i in s_expression[1])):
                    raise SnekSyntaxError('Second element in lambda must be a list of parameters')
        # if you are in this statement then we ran s_expression of tokens withs_expression finding the closing bracket
        if tokens[index:index + 1] != [')']:
            raise SnekSyntaxError('missing closing round bracket')
        return s_expression, index + 1

    def parse_expression(index):
        '''
        Returns a python representation of the list of tokens from tokens.
        Arguments:
            index (int): index of the tokens to check
        '''
        # cur_tok is the curent token you are analyzing
        cur_tok = tokens[index]
        # if the current token is a bracket.
        if cur_tok == ')' or cur_tok == '(':
            return parse_s_expression(index, tokens)

        # if it not a bracket is is either an int, float or var
        else:
            # check if cur_tok is int, float, var
            # check int
            if cur_tok.replace('-', '', 1).isdigit():
                if (cur_tok.count('-') == 1 and cur_tok[0] == '-') or cur_tok.count('-') == 0:
                    return int(cur_tok), index + 1
            # float check
            try:
                float_tok = float(cur_tok)
                return float_tok, index + 1
            # if it isnt a float or int, it is a variable, leave it as a string and move on
            except:
                return cur_tok, index + 1

    expression, index = parse_expression(0)
    print(f'index is {index}')
    # we have now parsed the whole expression. If there are any items left, there must be a double bracket
    if tokens[index:]:
        raise SnekSyntaxError('unexpected closing round bracket')
    print('### LISP INPUT ###   {}'.format(expression))
    return expression


######################
# Built-in Functions #
######################

def multiply(inputs):
    '''built in multiply funtion'''
    product = 1
    for item in inputs:
        product *= item
    return product

def divide(inputs):
    '''built in snek divide function'''
    first = inputs[0]
    rest = inputs[1:]
    for item in rest:
        first /= item
    return first

def create_comparison_function(comparison):
    '''
    based on the operator funcion passed in as comparison, this returns a
    function object that will operate on a list object using the comparison given
    '''
    def compare(list_of_inputs):
        len_input = len(list_of_inputs)
        if len_input > 1:
            return LispBool(all([comparison(list_of_inputs[i], list_of_inputs[i+1]) for i in range(len_input - 1)]))
        else:
            raise SnekSyntaxError('only one input given to boolean operator')
    return compare

class LispBool(int):
    '''
    Boolean class representation for Snek
    same class attirbutes as a integer however,
    0 is represented as  #f and 1 is represented as #t.
    Since it inherits all int methods, comparisons will still work on it.
    '''
    def __repr__(self):
        #represent a 0 as #f and 1 a #t
        return '#t' if self else '#f'

# LISTS
####################################################################################
def create_pair(args):
    '''
    Function to create an instance of pair object based on the list of arguments
    passed to it by the evaluator.
    List of arguments must be of length 2 where car is the first element and cdr is second.
    '''
    if len(args) != 2:
        raise SnekEvaluationError(f'Pair was passed a list of agruments of length {len(args)} instead of length 2')
    car = args[0]
    cdr = args[1]
    output = Pair(car, cdr)
    print('pair creaated: {}'.format(output))
    return output

class Pair(object):
    '''
    Pair object used for implementation of linked lists.
    Has two attributes: car and cdr and associated getter methods.
    '''
    def __init__(self, car, cdr):
        #initialize with car and cdr attributes
        self.car = car
        self.cdr = cdr
    def get_car(self):
        #get the car attribute
        return self.car
    def get_cdr(self):
        #get the cdr attribute
        return self.cdr
    def __repr__(self):
        return f'[{self.car}, {self.cdr}]'

    def get_last_element(self):
        cur_hold = self
        while not isinstance(cur_hold.cdr, Nil):
            cur_hold = cur_hold.cdr
        print(f'last element is {cur_hold}')
        return cur_hold

def get_car(pair_object_list):
    '''
    Return the car attribute from a Pair object.
    '''
    if len(pair_object_list) != 1:
        raise SnekEvaluationError('car expected exactly one argument (got %s)' % len(pair_object_list))
    pair = pair_object_list[0]
    if not isinstance(pair, Pair):
        raise SnekEvaluationError(f'called get_car on an {type(pair_object_list)} instead of a Pair')
    return pair.get_car()

def get_cdr(pair_object_list):
    '''
    Return the cbr attribute from a Pair object
    '''
    if len(pair_object_list) != 1:
        raise SnekEvaluationError('car expected exactly one argument (got %s)' % len(pair_object_list))
    pair = pair_object_list[0]
    if not isinstance(pair, Pair):
        raise SnekEvaluationError(f'called get_cdr on an {type(pair_object_list)} instead of a Pair')
    return pair.get_cdr()

class Nil:
    '''
    None type object equivalent
    '''
    def __bool__(self):
        return False
    def __repr__(self):
        return 'nil'
NIL = Nil()

def build_list(args):
    '''
    Return par object that recursively concatenates the rest of the arguments as another pair object in its cdr
    '''
    if not args:
        return NIL
    rest = ['list']
    rest.extend(args[1:])
    print(f'rest: {rest},  args: {args}')
    return evaluate(['cons', args[0], rest])

def list_iter(linked_list):
    '''
    List element generator that iterates through a list and yields its car
    '''
    if isinstance(linked_list, Nil):
        return
    if not isinstance(linked_list, Pair):
        raise SnekEvaluationError('can only interate on lists')
    print(f'iterating at {linked_list.car}')
    yield linked_list.car
    print(f'next is {linked_list.cdr}')
    yield from list_iter(linked_list.cdr)

def len_list(linked_list):
    '''
    Returns the length of a list by iterating through its elements
    '''
    length = 0
    if linked_list == NIL:
        return 0
    linked_list = linked_list[0]
    if not linked_list:
        return 0
    if not isinstance(linked_list, Pair):
        raise SnekEvaluationError(f'trying to calculate len of {type(linked_list)} but it should  be called on a Pair object')
    for item in list_iter(linked_list):
        length += 1
    print(f'the length was calculated to be {length}')
    return length

def get_at_index(arguments):
    '''
    Returns the list argument at the given index
    '''
    if len(arguments) != 2:
        raise SnekError(f'need 2 inputs for get_index, only given {len(arguments)}')
    list_input, index = arguments[0], arguments[1]

    if not isinstance(list_input, Pair):
        raise SnekEvaluationError('Did not input a list')
    if index == 0:
        return list_input.get_car()
    if index not in range(len_list(arguments[:1])):
        raise SnekEvaluationError('index called on list is outside the list range')
    for num, item in enumerate(list_iter(list_input)):
        if num == index:
            return item

def concatenate_lists(arguments):
    '''
    Return a list that is the concatenation of the input lists without modifying the inputs
    '''
    cur = NIL
    for lists in arguments:
        # copy of the input list
        input_copy = build_list([item for item in list_iter(lists)])
        # if this is the first list in arguments, set cur to this list
        if cur == NIL:
            cur = input_copy
        # else if this isnt the first list, add this list onto the ongoing list cur
        else:
            cur_last = cur.get_last_element()
            cur_last.cdr = input_copy
    return cur

def map(arguments):
    '''
    Return list of results from applying the input function to all items in the input list
    '''
    function = arguments[0]
    input_list = arguments[1]
    eval_list = [function([item]) for item in list_iter(input_list)]
    output_list = build_list(eval_list)
    return output_list

def filter(arguments):
    '''
    Return list of items from input list that are true when evaluated by the input function
    '''
    function = arguments[0]
    input_list = arguments[1]
    eval_list = [item for item in list_iter(input_list) if function([item])]
    output_list = build_list(eval_list)
    return output_list

def reduce(arguments):
    '''
    Return the the result fo iteratively applying the input function
    to the input list while maintaining an intermetiate result
    '''
    function = arguments[0]
    input_list = arguments[1]
    val = arguments[2]
    for item in list_iter(input_list):
        val = function([val, item])
    print(f'reduce output = {val}')
    return val

def begin(arguments):
    '''
    Return the last elemnt of a list
    '''
    return arguments[-1]

def evaluate_file(file_name, env = None):
    '''
    Evaluate user input from a file with filename in the input
    '''
    if env is None:
        env = Env({}, parent=make_builtin(snek_builtins))
    file = open(file_name, 'r')
    expression = file.read()
    return evaluate(parse(tokenize(expression)), env)

snek_builtins = {
    #operators
    '+': sum,
    '-': lambda args: -args[0] if len(args) == 1 else (args[0] - sum(args[1:])),
    '*': multiply,
    '/': divide,
    #booleans
    '#t': LispBool(True),
    '#f': LispBool(False),
    #comparisons
    '=?': create_comparison_function(lambda x, y: True if x == y else False),
    '>': create_comparison_function(lambda x, y: True if x > y else False),
    '>=': create_comparison_function(lambda x, y: True if x >= y else False),
    '<': create_comparison_function(lambda x, y: True if x < y else False),
    '<=': create_comparison_function(lambda x, y: True if x <= y else False),
    #lists
    'cons': create_pair,
    'car': get_car,
    'cdr': get_cdr,
    'nil': NIL,
    'list': build_list,
    'length': len_list,
    'elt-at-index': get_at_index,
    'concat': concatenate_lists,
    'map': map,
    'filter': filter,
    'reduce': reduce,
    #other
    'begin': begin,
    'evaluate_file': evaluate_file
}

def short_circuit (items, combinator, env):
    '''
    Return result of evaluating the cominator on all items
    '''
    # One false makes the whole expression false, else its true
    if combinator == 'and':
        for item in items:
            if evaluate(item, env) == False:
                return LispBool(False)
        return LispBool(True)
    # One true makes the whole expression true, else its false
    if combinator == 'or':
        for item in items:
            if evaluate(item, env) == True:
                return LispBool(True)
        return LispBool(False)

##############
# Evaluation #
##############

def evaluate(tree, env = None):
    '''
    Evaluate the given syntax tree according to the rules of the language.

    Arguments:
        tree (type varies): a fully parsed expression, as the output from the
                            parse function
    '''
    # For the first call we build the built-in environment
    # and build an empty environment whose parent is the built-in
    if env is None:
        builtin_env = make_builtin(snek_builtins)
        env = build_environment({}, builtin_env)
    # If the input is simply a float or integer, return that number
    if type(tree) == float or type(tree) == int:
        return tree
    # If the input is a pair, return the pair
    if isinstance(tree, Pair):
        print(f'tree: {tree} is recognised as a Pair object and returned')
        return tree
    # If the expression is a list it is an s-expression so we need to do some work to evaluate this
    if type(tree) == list:
        if not tree:
            raise SnekEvaluationError('No function for function call')
        #the operator is first, store that in first_el
        first_el = tree[0]
        len_tree = len(tree)
        #if the operator is an expression we need to evaluate it to see what it is
        if type(first_el) == list:
            print('List within a list. Inner[0]:  {}'.format(first_el))
            first_el = evaluate(first_el, env)
            return first_el(gen_list_items(tree[1:], env))
        #if first_el is the only item in the list we want to return its value
        #first check if it is a number,
        #but if it is not a number check if it is a string with a value stored in the environment
        if len_tree == 1:
            if type(tree) == float or type(tree) == int:
                # if so, return that number
                print('single number in a list  {}'.format(tree))
                return tree

            if type(tree) == str:
                if first_el in env:
                    print('single variable in list {}'.format(first_el))

                    snek_func = env[first_el]
                    # do this on the rest of the elements
                    return snek_func
                else:
                    raise SnekNameError
            else:
                first_el = evaluate(first_el, env)
                return first_el(gen_list_items(tree[1:], env))
        #if it is not a singular expression, we now need to do some work to evaluate it
        else:
            #check if the first element is something we have saved in our environment
            if first_el in env:
                snek_func = env[first_el]
                # do this on the rest of the elements
                return snek_func(gen_list_items(tree[1:], env))
            #if it is not something we have save, check if it is one of the special functions
            #if it is 'define', here is how we evaluate it:
            elif first_el == 'define':
                if len_tree != 3:
                    raise SnekEvaluationError
                name = tree[1]
                value = tree[2]

                if not isinstance(name, (list, str)):
                     raise SnekEvaluationError("Expected symbol or function signature after define keyword")

                #case for easier fuction definition
                if type(name) == list:
                    if len(name) == 0:
                        raise SnekSyntaxError('need at least one parameter when defining a function')
                    return evaluate(['define', name[0], ['lambda', name[1:], tree[2]]], env)

                #regular function definition with lambda
                else:
                    print(f'defining => name: {name},  value:  {value}')
                    result = evaluate(value, env)
                    print(f'evaluate the value: {result}')
                    env[name] = result
                    return result

            elif first_el == 'lambda':
                if len_tree != 3:
                    raise SnekEvaluationError
                param = tree[1]
                code = tree[2]
                return SnekLambdaFunction(param, code, env)

            elif first_el == 'if':
                if len_tree != 4:
                    raise SnekEvaluationError
                cond = tree[1]
                cond_eval = evaluate(cond, env)

                if cond_eval == True:
                    print(f'if cond eval false, returning:  {tree[2]}')
                    return evaluate(tree[2], env)
                else:
                    print(f'if cond eval false, returning:  {tree[3]}')
                    return evaluate(tree[3], env)

            elif first_el == 'and':
                return short_circuit(tree[1:], 'and', env)

            elif first_el == 'or':
                return short_circuit(tree[1:], 'or', env)

            elif first_el == 'not':
                rest = tree[1:]
                if not rest:
                    raise SnekEvaluationError('nothing after not statement')
                rest = rest[0]
                rest_evaled = evaluate(rest)
                return LispBool(not rest_evaled)
            elif first_el == 'let':
                variables, values = zip(*tree[1])
                new_expr = [['lambda', list(variables), tree[2]]] + list(values)
                return evaluate(new_expr, env)
            elif first_el == 'set!':
                var = tree[1]
                exp = tree[2]
                var_env = env.which_env(var)
                if var_env is None:
                    raise SnekNameError(f'variable to be set, {var} was not found in any environment')
                eval_exp = evaluate(exp, env)
                var_env[var] = evaluate(exp, env)
                return eval_exp

            #here if we dont have the value saved in the envorinement and its not a special function
            else:
                raise SnekEvaluationError

    elif isinstance(tree, Nil):
        return NIL

    #if its not a list or a float, it is a string
    #check if we have the string saved as a variable in the environment
    elif tree in env:
        return env[tree]
    #if we dont have it saved raise an error
    else:
        raise SnekNameError("Symbol called not found in the environment")


def result_and_env(expr, env=None):
    '''
    For testing purposes, return both the result of the given evaluation and
    the environment in which the expression was evaluated.
    '''
    if env is None:
        env = Env({}, parent=make_builtin(snek_builtins))
    return evaluate(expr, env), env


def gen_list_items(some_list, env):
    '''
    From a list of items, evaluate every item and return a list of the evaluated items
    '''
    print('generating list  of evaluated')
    test = [evaluate(item, env) for item in some_list]
    print(f'generationg a list of items for func:  {test}')
    return test


class Env(dict):
    '''
    Custom environment class that can be searched for variables and has a parent
    environment that can be searched of the variable is not found in the curren env
    '''
    def __init__(self, *args, parent=None):
        '''Initialize with the properties of a dict and a parent pointer'''
        dict.__init__(self, *args)
        self.parent = parent

    def __contains__(self, key):
        '''Search this environment the recursively search the parent environments'''
        if dict.__contains__(self, key):
            return True
        if self.parent is not None:
            return key in self.parent
        else:
            return False

    def __getitem__(self, key):
        '''Return the vallue assosiated with the input key'''
        environment = self.which_env(key)
        if environment is None:
            raise SnekNameError
        else:
            return dict.__getitem__(environment, key)

    def which_env(self, key):
        '''Return the first environment the key is found via the recursive search'''
        if dict.__contains__(self, key):
            return self
        elif self.parent is not None:
            return self.parent.which_env(key)
        else:
            return None

    def get_parent(self):
        '''Return the parent environment'''
        if self.parent is not None:
            return self.parent
        else:
            return None

def build_environment(arg_dict, parent_env):
    '''Return an environment with the input dictionary of values and input parent environment'''
    return Env(arg_dict, parent = parent_env)

def make_builtin(snek_builtins):
    '''
    Return an environment with the builtin operations and no parent environment
    This is the highest level environment and the last to be searched in all cases
    '''
    return build_environment(snek_builtins, None)

class SnekLambdaFunction:
    '''Create Lambda function'''
    def __init__(self, param, code, env):
        self.param = param
        self.code = code
        self.env = env
        print(f'made lambda => param: {self.param}  code:  {self.code}  env:  {self.env}')


    def __call__(self, args):
        if len(self.param) != len(args):
            raise SnekEvaluationError('len args not same as len params')
        # evaluate all of the arguments to the function in the current environment (from which the function is being called).
        # evaluated = [evaluate(item, self.env) for item in args]
        # make a new environment whose parent is the environment in which the function was defined
        # in that new environment, bind the function's parameters to the arguments that are passed to it.
        d = dict(zip(self.param, args))
        func_env = build_environment(d, self.env)
        # evaluate the body of the function in that new environment.
        print('code to be evald:  {}'.format(self.code))
        return evaluate(self.code, func_env)


def repl(env=None):
    '''
    Drop the user into a Read-Evaluate-Print Loop in the given environment.
    '''
    print('howdy')
    try:
        inp = None
        if env is None:
            env = Env({}, parent=make_builtin(snek_builtins))
        while inp != 'quit':
            if inp:
                print()
                try:
                    expr = parse(tokenize(inp))
                    print('expression evaluated:',evaluate(expr, env), ', env:', env)
                except SnekSyntaxError as e:
                    print('snek syntax',  e.args[0])
                except SnekEvaluationError as e:
                    print('snek eval', e)
                except SnekNameError as e:
                    print('sneknameerror', e.args[0])
                except RecursionError:
                    print("Maximum recursion limit exceeded!")
                print()
            try:
                inp = input('insert expression\n')
            except KeyboardInterrupt:
                inp = ''
                print()
    except EOFError:
        print()
    print('BYE!')

def make_builtin_env():
    '''
    Helper function to create a new environment containing the builtins
    '''
    return Env(snek_builtins)

if __name__ == '__main__':
    # code in this block will only be executed if lab.py is the main file being
    # run (not when this module is imported)

    # uncommenting the following line will run doctests from above
    env = Env({}, parent=make_builtin_env())
    files_to_load = sys.argv[1:]
    for fname in files_to_load:
        try:
            evaluate_file(fname, env)
        except:
            print('an error occurred when loading file %s' % fname)
    repl(env)
    pass
