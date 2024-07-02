"""6.009 Lab 9: Snek Interpreter Part 2"""

from os import curdir
import sys
import doctest
sys.setrecursionlimit(10_000)

###########################
# Snek-related Exceptions #
###########################

class SnekError(Exception):
    """
    A type of exception to be raised if there is an error with a Snek
    program.  Should never be raised directly; rather, subclasses should be
    raised.
    """
    pass

class SnekSyntaxError(SnekError):
    """
    Exception to be raised when trying to evaluate a malformed expression.
    """
    def __str__(self):
        return 'SnekSyntaxError'

class SnekNameError(SnekError):
    """
    Exception to be raised when looking up a name that has not been defined.
    """
    def __str__(self):
        return 'SnekNameError'

class SnekEvaluationError(SnekError):
    """
    Exception to be raised if there is an error during evaluation other than a
    SnekNameError.
    """
    def __str__(self):
        return 'SnekEvaluationError'


############################
# Tokenization and Parsing #
############################


def number_or_symbol(x):
    """
    Helper function: given a string, convert it to an integer or a float if
    possible; otherwise, return the string itself

    >>> number_or_symbol('8')
    8
    >>> number_or_symbol('-5.32')
    -5.32
    >>> number_or_symbol('1.2.3.4')
    '1.2.3.4'
    >>> number_or_symbol('x')
    'x'
    """
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return x

def tokenize(source):
    """
    Splits an input string into meaningful tokens (left parens, right parens,
    other whitespace-separated values).  Returns a list of strings.

    Arguments:
        source (str): a string containing the source code of a Snek
                      expression
    """
    
    if '(' not in source:
        return [source]
    
    res = []
    inComment = False
    upTo = ''
    for let in source:
        if let == ';':
            inComment = True
        elif let == '\n':
            if upTo and not inComment: 
                res.append(upTo)
            inComment = False
            upTo = ''
        elif let == '(' and not inComment:
            upTo = ''
            res.append(let)
        elif let == ' ' and not inComment:
            if upTo: 
                res.append(upTo)
                upTo = ''
        
        elif let == ')' and not inComment:
            if upTo: 
                res.append(upTo)
                upTo = ''
            res.append(let)
        else:
            upTo += let
    if upTo and not inComment:
        res.append(upTo)
    return res        

def parse(tokens):
    """
    Parses a list of tokens, constructing a representation where:
        * symbols are represented as Python strings
        * numbers are represented as Python ints or floats
        * S-expressions are represented as Python lists

    Arguments:
        tokens (list): a list of strings representing tokens
    """
    checkNotValid(tokens)
    parsed = recParse(tokens)
    checkParse(parsed)
    return parsed

def checkParse(parsed):
    """Checks parsed to ensure that it is valid expression and if not, raises Syntax Error

    Args:
        parsed (str): newly parsed segment to analyze
    """
    if not isinstance(parsed, list) or not parsed:
        return
    
    if parsed[0] == 'define':
        if len(parsed) != 3 or isinstance(parsed[1], (int,float)) or parsed[1] == []:
            raise SnekSyntaxError
        for elem in parsed[1]:
            if not isinstance(elem,str):
                raise SnekSyntaxError
            
            
    if parsed[0] == 'lambda':
        if len(parsed) != 3 or not isinstance(parsed[1], list):
            raise SnekSyntaxError
        for elem in parsed[1]:
            if not isinstance(elem, str):
                raise SnekSyntaxError
            
    if parsed[0] == 'if':
        if len(parsed) != 4:
            raise SnekSyntaxError
        
    for elem in parsed:
        if isinstance(elem,list):
            checkParse(elem)
    
def checkNotValid(tokens):
    """Initial check to see if par counts are off in input before parsing as well as ensuring that each token is unique and correct

    Args:
        tokens (list): List of tokens after tokenize to parse
    """
    if len(tokens) > 1 and '(' not in tokens:
        raise SnekSyntaxError
    if tokens[0] == 'define' or tokens[0] == 'lambda':
        raise SnekSyntaxError
    parCount = 0
    for tok in tokens:
        if ' ' in tok:
            raise SnekSyntaxError
        if tok == '(':
            parCount += 1
        if tok == ')':
            parCount -= 1
            if parCount < 0:
                raise SnekSyntaxError
    if parCount != 0:
        raise SnekSyntaxError
      
def recParse(tokens):
    """Recursive parsing to fully parse any set of tokens

    Args:
        tokens (list): list of tokens to be parsed into correct format

    Returns:
        list: nested lists that describe correct LISP syntax that we can work with
    """
    
    if len(tokens) == 1:
        return number_or_symbol(tokens[0])
    res = []
    i = 1
    while i < len(tokens) - 1:
        if tokens[i] == '(':
            lastParIndex = findLastPar(tokens[i:])
            res.append(recParse(tokens[i:lastParIndex + i]))
            i = lastParIndex + i - 1
        else:
            res.append(number_or_symbol(tokens[i]))
        i += 1
    return res

def findLastPar(tokens):
    """Function that finds last parentasis that matches the start of the first (used for finding total expressions)

    Args:
        tokens (list): token list to find last Par

    Returns:
        int: index of the last parenthasis that matches
    """
    parCount = 0
    index = 0
    for tok in tokens:
        if tok == '(': parCount += 1
        if tok == ')': parCount -= 1
        index += 1
        if parCount == 0:
            return index
 
            
######################
#    Object Classes  #
######################

class enviornment(object):
    def __init__(self,name = None, vars = dict(), parent = None):
        self.name = name
        self.vars = vars
        self.parent = parent
        
    def addVar(self,varName, val):
        self.vars[varName] = val
        return val
    
    def getVar(self, var):
        """Gets variable from itself or parent

        Args:
            var (str): name of variable

        Raises:
            SnekNameError: if variable is not in frame or any parent frames

        Returns:
            any: value stored at that variable
        """
        if var in self.vars:
            return self.vars[var]
        elif self.parent != None:
            return self.parent.getVar(var)
        raise SnekNameError
        
    def deleteVar(self, var):
        """Deletes the variable in the current enviornment

        Args:
            var (str): name of variable to delete

        Returns:
            any: value stored at that variable
        """
        if var in self.vars:
            val = self.vars[var]
            del self.vars[var]
            return val
        raise SnekNameError
    
    def setVar(self, var, val):
        """Sets variable to new value and if not already in the vars dict, searches parent frames

        Args:
            var (str): name of variable
            val (any): new value to set at that name
            
        Returns:
            any: value that we set the variable to
        """
        if var in self.vars:
            self.vars[var] = val
            return val
        elif self.parent != None:
            return self.parent.setVar(var,val)
        raise SnekNameError

class SnekFunction(object):
    def __init__(self, params, expr, frame):
        self.params = params
        self.expr = expr
        self.frame = frame
            
class Pair(object):
    def __init__(self, head, tail, isList = True):
        self.head = head
        self.tail = tail
        self.isList = isList
    
    def getLastPair(self):
        """Gets last pair in the linked list by iterating through entire list and returning at end

        Returns:
            Pair: Pair object at the end of the list
        """
        if self.isList:
            curPair = self
            while(curPair.tail != None):
                curPair = curPair.tail
            return curPair


######################
# Built-in Functions #
######################

def add(tree, frame):
    res = 0
    for elem in tree[1:]:
        res += evaluate(elem, frame)
    return res

def sub(tree, frame):
    args = [evaluate(elem,frame) for elem in tree[1:]]
    return -args[0] if len(args) == 1 else (args[0] - add(args[:], frame))

def product(tree, frame):
    args = [evaluate(elem,frame) for elem in tree[1:]]
    res = 1
    for arg in args:
        res *= arg
    return res

def div(tree, frame):
    args = [evaluate(elem,frame) for elem in tree[1:]]
    res = args[0]
    for arg in args[1:]:
        res /= arg
    return res

def defineFunc(tree, frame):
    """defines new variable or function

    Args:
        tree (list): tree that we are operating on
        frame (enviornment): current enviornment that we are operating in

    Returns:
        any: value of variable that has been set (either int or object)
    """
    varName = tree[1]
    if not isinstance(varName, list):
        val = evaluate(tree[2], frame)
        return frame.addVar(varName, val)
    else:
        func = makeFunc(varName[1:],tree[2],frame)
        return frame.addVar(varName[0],func)

def lambdaFunc(tree, frame):
    return makeFunc(tree[1],tree[2],frame)

def compAll(func):
    """Create comparison Functions

    Args:
        func (func): function to use to define comparison functions
    """
    def comp(tree, frame):
        args = [evaluate(elem, frame) for elem in tree[1:]]
        for i in range(0,len(args) - 1):
            if not func(args[i], args[i+1]):
                return False
        return True
    return comp

def listFunc(tree, frame):
    return makeList(tree[1:], frame)

def tailFunc(tree, frame):
    """returns tail of given pair and checks to make sure it is a pair

    Args:
        tree (list): tree that we are operating on
        frame (enviornment): current enviornment that we are operating in

    Returns:
        any: value at the tail of the pair
    """
    if len(tree) == 2 and isinstance(evaluate(tree[1], frame), Pair):
        return evaluate(tree[1], frame).tail
    raise SnekEvaluationError

def headFunc(tree, frame):
    """returns head of given pair and checks to make sure it is a pair

    Args:
        tree (list): tree that we are operating on
        frame (enviornment): current enviornment that we are operating in

    Returns:
        any: value at the head of the pair
    """
    if  len(tree) == 2 and isinstance(evaluate(tree[1], frame), Pair):
        return evaluate(tree[1], frame).head
    raise SnekEvaluationError

def pairFunc(tree, frame):
    """creates new pair given head and tail arguments

    Args:
        tree (list): tree that we are operating on
        frame (enviornment): current enviornment that we are operating in

    Returns:
        Pair: pair with the values given
    """
    if len(tree) != 3:
        raise SnekEvaluationError
    if isinstance(evaluate(tree[2], frame), int):
        return Pair(evaluate(tree[1],frame),evaluate(tree[2], frame), False)
    else:
        return Pair(evaluate(tree[1],frame),evaluate(tree[2], frame))

def orFunc(tree, frame):
    """Or that evaluates true if any of the elements is true and false otherwise supports short circuiting.

    Args:
        tree (list): tree that we are operating on
        frame (enviornment): current enviornment that we are operating in

    Returns:
        bool: true or false depending on or rules
    """
    for elem in tree[1:]:
        if evaluate(elem, frame):
            return True
    return False

def andFunc(tree, frame):
    """and function that evaluates true if all elements are true and false otherwise with short circuiting

    Args:
        tree (list): tree that we are operating on
        frame (enviornment): current enviornment that we are operating in

    Returns:
        bool: boolean that represents correct value
    """
    for elem in tree[1:]:
        if not evaluate(elem, frame):
            return False
    return True

def notFunc(tree, frame):
    """Negates evaluation of expression after the not

    Args:
        tree (list): tree that we are operating on
        frame (enviornment): current enviornment that we are operating in

    Returns:
        bool: boolean that is negated value
    """
    if len(tree) != 2:
        raise SnekEvaluationError
    return not evaluate(tree[1], frame)

def ifFunc(tree, frame):
    """evaluates true arguement if true and false if false

    Args:
        tree (list): tree that we are operating on
        frame (enviornment): current enviornment that we are operating in

    Returns:
        any: value of calling if statement and evaluating correct expression
    """
    cond = tree[1]
    if evaluate(cond,frame):
        return evaluate(tree[2], frame)
    else:
        return evaluate(tree[3], frame)

def makeFunc(params,expr,funcFrame):
    return SnekFunction(params,expr,funcFrame)

def setFunc(tree,frame):
    return frame.setVar(tree[1],evaluate(tree[2],frame))

def letFunc(tree, frame):
    """adds multiple variables in the let fashion

    Args:
        tree (list): tree that we are operating on
        frame (enviornment): current enviornment that we are operating in

    Returns:
        any: val of evaluating each
    """
    
    newEnv = enviornment('newEnv',dict(),frame)
    for var in tree[1]:
        newEnv.addVar(var[0],evaluate(var[1],frame))
    
    return evaluate(tree[2],newEnv)

def delFunc(tree, frame):
    return frame.deleteVar(tree[1])

def beginFunc(tree, frame):
    """evalutes all arguements in tree and then returns the last one

    Args:
        tree (list): tree that we are operating on
        frame (enviornment): current enviornment that we are operating in

    Returns:
        any: value of evaluating all
    """
    for arg in tree[1:-1]:
        evaluate(arg, frame)
    return evaluate(tree[-1],frame)

def reduceFunc(tree, frame):
    """reduces list and returns value

    Args:
        tree (list): tree that we are operating on
        frame (enviornment): current enviornment that we are operating in

    Returns:
        int: value of applying function to start and all elements
    """
    func = tree[1]
    OGList = evaluate(tree[2],frame)
    curVal = evaluate(tree[3],frame)
    
    def reduceFuncCopy(OGList, frame, curVal):
        if OGList ==  None:
            return curVal
        if not isinstance(OGList, Pair) or not OGList.isList:
            raise SnekEvaluationError 
        newVal = evaluate([func] + [curVal] + [OGList.head], frame)
        return reduceFuncCopy(OGList.tail,frame,newVal)
        
    return reduceFuncCopy(OGList,frame,curVal)

def filterFunc(tree, frame):
    """applies filter to function and filters out elements that do not meet filter specs

    Args:
        tree (list): tree that we are operating on
        frame (enviornment): current enviornment that we are operating in

    Returns:
        Pair: start of linked list with filter applied
    """
    func = tree[1]
    OGList = evaluate(tree[2], frame)
    
    def filterFuncCopy(OGList, frame):
        if OGList ==  None:
            return None
        if not isinstance(OGList, Pair) or not OGList.isList:
            raise SnekEvaluationError
        if evaluate([func] + [OGList.head],frame):
            return Pair(evaluate(OGList.head, frame), filterFuncCopy(OGList.tail, frame))
        else:
            return filterFuncCopy(OGList.tail, frame)
    
    return filterFuncCopy(OGList,frame)
    
def mapFunc(tree, frame):
    """Maps elements based on original list and function given

    Args:
        tree (list): tree that we are operating on
        frame (enviornment): current enviornment that we are operating in

    Returns:
        Pair: head of new linked list with map applied
    """
    func = tree[1]
    OGList = evaluate(tree[2], frame)
    
    def mapFuncCopy(OGList, frame):
        if OGList ==  None:
            return None
        if not isinstance(OGList, Pair) or not OGList.isList:
            raise SnekEvaluationError
        return Pair(evaluate([func] + [OGList.head], frame), mapFuncCopy(OGList.tail, frame))
    return mapFuncCopy(OGList,frame)
    
def concat(tree, frame):
    """Concatenates multiple linked lists

    Args:
        tree (list): tree that we are operating on
        frame (enviornment): current enviornment that we are operating in

    Returns:
        Pair: new linked list that is other lists compiled
    """
    pairs = tree[1:]
    if not pairs:
        return None
    pairIndex = 0
    res = None
    while pairIndex < len(pairs):
        curPiar = evaluate(pairs[pairIndex], frame)
        if (isinstance(curPiar, Pair) and curPiar.isList) or curPiar == None:
            copy = makeListCopy(curPiar, frame)
            if res != None:
                res.getLastPair().tail = copy
            else: res = copy
            pairIndex += 1
        else: raise SnekEvaluationError
    return res
             
def makeListCopy(OGList, frame):
    """Makes copy of the list passed in and return in

    Args:
        tree (list): tree that we are operating on
        frame (enviornment): current enviornment that we are operating in

    Returns:
        Pair: new list that is a copy of the original
    """
    if OGList ==  None:
        return None
    return Pair(evaluate(OGList.head, frame), makeListCopy(OGList.tail, frame))

def nthElem(tree, frame):
    """Returns the nth element of a linked list

    Args:
        tree (list): tree that we are operating on
        frame (enviornment): current enviornment that we are operating in
        
    Returns:
        any: element at that index in the linked list
    """
    if len(tree) != 3 or not isinstance(evaluate(tree[1], frame), Pair):
        raise SnekEvaluationError
    try:
        curPair = evaluate(tree[1], frame)
        targetIndex = evaluate(tree[2],frame)
        if curPair.isList:
            curIndex = 0
            while curIndex < targetIndex:
                curPair = curPair.tail
                curIndex += 1
            return curPair.head
        else:
           if targetIndex == 0: return curPair.head
           else: raise SnekEvaluationError 
    except:
        raise SnekEvaluationError
    
def makeList(elems, frame):
    """creates a linked list containing elements passed in

    Args:
        elems (list): list of elements to operate on
        frame (enviornment): current enviornment that we are operating in

    Returns:
        Pair: head Pair of the linked list
    """
    if len(elems) == 0:
        return None
    if len(elems[1:]) == 0:
        return Pair(evaluate(elems[0], frame), None)
    return Pair(evaluate(elems[0], frame),makeList(elems[1:], frame))

def lengthFunc(tree, frame):
    """Returns length of the given linked list

    Args:
        tree (list): tree that we are operating on
        frame (enviornment): current enviornment that we are operating in

    Returns:
        int: length of inputted linked list
    """
    if evaluate(tree[1], frame) == None:
        return 0
    if len(tree) == 2 and isinstance(evaluate(tree[1], frame), Pair) and evaluate(tree[1], frame).isList:
        curPair = evaluate(tree[1], frame)
        count = 1
        while curPair.tail != None and isinstance(curPair.tail, Pair):
            count += 1
            curPair = curPair.tail
        return count
    raise SnekEvaluationError

snek_builtins = {
    "+": add,
    "-": sub,
    "*": product,
    "/": div,
    "#f": False,
    "#t": True,
    "=?": compAll(lambda x,y: x == y),
    ">": compAll(lambda x,y: x > y),
    ">=": compAll(lambda x,y: x >= y),
    "<": compAll(lambda x,y: x < y),
    "<=": compAll(lambda x,y: x <= y),
    'not': notFunc,
    "nil": None,
    'tail': tailFunc,
    'head': headFunc,
    'if': ifFunc,
    'and': andFunc,
    'or': orFunc,
    'define': defineFunc,
    'lambda': lambdaFunc,
    'pair': pairFunc,
    'list': listFunc,
    'length': lengthFunc,
    'nth': nthElem,
    'concat': concat,
    'map': mapFunc,
    'filter': filterFunc,
    'reduce': reduceFunc,
    'begin': beginFunc,
    'del': delFunc,
    'let': letFunc,
    'set!': setFunc
}

##############
# Evaluation #
##############

def makeNewEnviornment():
    """Creates not Environment Blank

    Returns:
        enviornment: blank enviornment
    """
    builtIns = enviornment('builtIns', snek_builtins, None)
    return enviornment('globalF', dict(), builtIns)

def evaluate(tree, frame = None):
    """
    Evaluate the given syntax tree according to the rules of the Snek
    language.

    Arguments:
        tree (type varies): a fully parsed expression, as the output from the
                            parse function
    """
    
    if frame == None:
        frame = makeNewEnviornment()
    
    if isinstance(tree, (int,float, Pair)):
        return tree
    elif isinstance(tree, str):
        return frame.getVar(tree)
    elif isinstance(tree, SnekFunction):
        return evaluate(tree.expr, frame)
    
    elif isinstance(tree, list):
        try:
            operator = tree[0]
        except: raise SnekEvaluationError
        
        #if need to evaluate expression inside of list
        if isinstance(operator, list):
            operatorFunc = evaluate(operator, frame)
        elif isinstance(operator, int):
            raise SnekEvaluationError
        else:
            operatorFunc = frame.getVar(operator)
        
        
        if not isinstance(operatorFunc, SnekFunction):
            return operatorFunc(tree,frame)
        else:
            newEnv = newFuncFrame(operator, operatorFunc, tree[1:], frame)
            return evaluate(operatorFunc, newEnv)

def newFuncFrame(operator, operatorFunc, elems, frame):
    """Creates a new frame for function calls and binds variables properly

    Args:
        operator (str): name of function
        operatorFunc (func): function that needs to be called in the new frame
        elems (list): list of elements that need to be passed into the function and binded
        frame (enviornment): frame of parent pointer

    Raises:
        SnekEvaluationError: if bad, raise this

    Returns:
        enviornment: enviorment that represents new frame with binded elems and correct parent pointer
    """
    evaledArgs = []
    for arg in elems:
        if not isinstance(arg, SnekFunction):
            # print(evaluate(arg,frame))
            evaledArgs.append(evaluate(arg,frame))
        else:
            evaledArgs.append(arg)
    newEnv = enviornment('frame'+str(operator), dict(),operatorFunc.frame)
    if len(evaledArgs) != len(operatorFunc.params):
        raise SnekEvaluationError
    for i in range(len(evaledArgs)):
        newEnv.addVar(operatorFunc.params[i],evaledArgs[i])
    return newEnv
            
def result_and_env(tree, frame = None):
    if frame == None:
        frame = makeNewEnviornment()  
    return evaluate(tree, frame), frame

def evaluate_file(fileName, frame = None):
    """Evalautes Code in a file in a given frame. Allows us to not need to manully input everything into REPL

    Args:
        fileName (str): path to the file we need to evaluate
        frame (enviornment, optional): frame in which to evaluate the file arguements. Defaults to None.

    Returns:
        any: result of evaluating lines of code in the file
    """
    if frame == None:
        frame = makeNewEnviornment()
    with open(fileName) as f:
        input = f.read()
    return evaluate(parse(tokenize(input)),frame)

def REPL(frame):
    """Standard REPL that handles exceptions and takes in user input to parse

    Args:
        frame (enviornment): frame in which to evaluate REPL arguements
    """
    uInput = ''
    while uInput != 'QUIT':
        uInput = str(input("in> " ))
        if uInput != 'QUIT':
            try:
                print('\tout> ',evaluate(parse(tokenize(uInput)), frame))
            except Exception as e:
                print('Snek Error here:', str(e))

if __name__ == "__main__":
     
    frame = makeNewEnviornment()
    evaluate_file('6.009/lab09/test_files/definitions.snek',frame)
    REPL(frame)
    
    
    # strA = '(begin (define (range start stop) (if (>= start stop) nil (pair start (range (+ start 1) stop)))) (define (all values) (if (=? (length values) 0) #t (and (head values) (all (tail values))))) (define (min a b) (if (< a b) a b)) (define (max a b) (if (> a b) a b)) (define (zip a b) (if (or (=? (length a) 0) (=? (length b) 0)) nil (pair (pair (head a) (head b)) (zip (tail a) (tail b))))) (define (set-nth ll index value) (if (=? index 0) (pair value (tail ll)) (pair (head ll) (set-nth (tail ll) (- index 1) value)))) (define (initialize-nd dimensions value) (if (=? (length dimensions) 0) value (map (lambda (_) (initialize-nd (tail dimensions) value)) (range 0 (head dimensions))))) (define (neighbors-nd dimensions coordinates) (if (=? (length dimensions) 0) (list (list)) (reduce concat (map (lambda (suffix) (map (lambda (prefix) (concat (list prefix) suffix)) (range (max 0 (- (head coordinates) 1)) (min (head dimensions) (+ (head coordinates) 2))))) (neighbors-nd (tail dimensions) (tail coordinates))) (list))))'
    # strA += '(define (get-nd board coordinates) (if (=? (length coordinates) 1) (nth board (head coordinates)) (get-nd (nth board (head coordinates)) (tail coordinates)))) (define (set-nd board coordinates value) (if (=? (length coordinates) 1) (set-nth board (head coordinates) value) (set-nth board (head coordinates) (set-nd (nth board (head coordinates)) (tail coordinates) value)))) (define (is-victory board mask dimensions) (if (=? (length dimensions) 0) (or mask (=? board -1)) (all (map (lambda (pair) (is-victory (head pair) (tail pair) (tail dimensions))) (zip board mask))))) (define (game-get-state game) ((nth game 0))) (define (game-set-state game state) ((nth game 1) state)) (define (game-get-board game) ((nth game 2)))'
    # strA += '(define (game-set-board game board) ((nth game 3) board)) (define (game-get-mask game) ((nth game 4))) (define (game-set-mask game mask) ((nth game 5) mask)) (define (game-get-dimensions game) ((nth game 6))) (define (new-game-nd dimensions bombs) (begin (define state 0) (define board (initialize-nd dimensions 0)) (define mask (initialize-nd dimensions #f)) (define self (list (lambda () state) (lambda (new_state) (set! state new_state)) (lambda () board) (lambda (new_board) (set! board new_board)) (lambda () mask) (lambda (new_mask) (set! mask new_mask)) (lambda () dimensions))) (map (lambda (bomb) (begin (game-set-board self (set-nd (game-get-board self) bomb -1)) (map (lambda (neighbor) (begin (define value (get-nd (game-get-board self) neighbor)) (if (>= value 0) (game-set-board self (set-nd (game-get-board self) neighbor (+ value 1))) nil))) (neighbors-nd dimensions bomb)))) bombs) self))'
    # strA += '(define (dig-nd game coordinates) (if (or (get-nd (game-get-mask game) coordinates) (not (=? (game-get-state game) 0))) 0 (if (=? (get-nd (game-get-board game) coordinates) -1) (begin (game-set-mask game (set-nd (game-get-mask game) coordinates #t)) (game-set-state game 2) 1) (begin (define (dig-helper coordinates) (if (get-nd (game-get-mask game) coordinates) 0 (begin (game-set-mask game (set-nd (game-get-mask game) coordinates #t)) (if (=? (get-nd (game-get-board game) coordinates) 0) (reduce + (map (lambda (neighbor) (dig-helper neighbor)) (neighbors-nd (game-get-dimensions game) coordinates)) 1) 1)))) (define count (dig-helper coordinates)) (if (is-victory (game-get-board game) (game-get-mask game) (game-get-dimensions game)) (game-set-state game 1) nil) count))))'
    # strA += ')'

    # evaluate(parse(tokenize(strA)), frame)
    # evaluate(parse(tokenize('(define game (new-game-nd (list 2 4) (list (list 0 0) (list 1 0) (list 1 1))))')),frame)