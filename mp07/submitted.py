'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import copy, queue

def standardize_variables(nonstandard_rules):
    '''
    @param nonstandard_rules (dict) - dict from ruleIDs to rules
        Each rule is a dict:
        rule['antecedents'] contains the rule antecedents (a list of propositions)
        rule['consequent'] contains the rule consequent (a proposition).
   
    @return standardized_rules (dict) - an exact copy of nonstandard_rules,
        except that the antecedents and consequent of every rule have been changed
        to replace the word "something" with some variable name that is
        unique to the rule, and not shared by any other rule.
    @return variables (list) - a list of the variable names that were created.
        This list should contain only the variables that were used in rules.
    '''
    variable_count = 0
    variables = []
    standardized_rules = dict()
    for id, rule in nonstandard_rules.items():
        if "something" not in rule['consequent']:
            no_something = True
            for antecedent in rule['antecedents']:
                if "something" in antecedent:
                    no_something = False
            if no_something:
                standardized_rules[id] = copy.deepcopy(rule)
                continue
        variable_count += 1
        new_rule = dict()
        new_rule['text'] = rule['text']
        new_antecedents = copy.deepcopy(rule['antecedents'])
        new_consequent = rule['consequent'][:]
        for i in range(len(new_antecedents)):
            for j in range(len(new_antecedents[i])):
                if new_antecedents[i][j] == "something":
                    new_antecedents[i][j] = hex(variable_count)
        for i in range(len(new_consequent)):
            if new_consequent[i] == "something":
                new_consequent[i] = hex(variable_count)
        variables.append(hex(variable_count))
        new_rule['antecedents'] = new_antecedents
        new_rule['consequent'] = new_consequent
        standardized_rules[id] = new_rule

    return standardized_rules, variables


def find(x, subs):
    if x not in subs:
        return x
    return find(subs[x], subs)


def unify(query, datum, variables):
    '''
    @param query: proposition that you're trying to match.
      The input query should not be modified by this function; consider deepcopy.
    @param datum: proposition against which you're trying to match the query.
      The input datum should not be modified by this function; consider deepcopy.
    @param variables: list of strings that should be considered variables.
      All other strings should be considered constants.
    
    Unification succeeds if (1) every variable x in the unified query is replaced by a 
    variable or constant from datum, which we call subs[x], and (2) for any variable y
    in datum that matches to a constant in query, which we call subs[y], then every 
    instance of y in the unified query should be replaced by subs[y].

    @return unification (list): unified query, or None if unification fails.
    @return subs (dict): mapping from variables to values, or None if unification fails.
       If unification is possible, then answer already has all copies of x replaced by
       subs[x], thus the only reason to return subs is to help the calling function
       to update other rules so that they obey the same substitutions.

    Examples:

    unify(['x', 'eats', 'y', False], ['a', 'eats', 'b', False], ['x','y','a','b'])
      unification = [ 'a', 'eats', 'b', False ]
      subs = { "x":"a", "y":"b" }
    unify(['bobcat','eats','y',True],['a','eats','squirrel',True], ['x','y','a','b'])
      unification = ['bobcat','eats','squirrel',True]
      subs = { 'a':'bobcat', 'y':'squirrel' }
    unify(['x','eats','x',True],['a','eats','a',True],['x','y','a','b'])
      unification = ['a','eats','a',True]
      subs = { 'x':'a' }
    unify(['x','eats','x',True],['a','eats','bobcat',True],['x','y','a','b'])
      unification = ['bobcat','eats','bobcat',True],
      subs = {'x':'a', 'a':'bobcat'}
      When the 'x':'a' substitution is detected, the query is changed to 
      ['a','eats','a',True].  Then, later, when the 'a':'bobcat' substitution is 
      detected, the query is changed to ['bobcat','eats','bobcat',True], which 
      is the value returned as the answer.
    unify(['a','eats','bobcat',True],['x','eats','x',True],['x','y','a','b'])
      unification = ['bobcat','eats','bobcat',True],
      subs = {'a':'x', 'x':'bobcat'}
      When the 'a':'x' substitution is detected, the query is changed to 
      ['x','eats','bobcat',True].  Then, later, when the 'x':'bobcat' substitution 
      is detected, the query is changed to ['bobcat','eats','bobcat',True], which is 
      the value returned as the answer.
    unify([...,True],[...,False],[...]) should always return None, None, regardless of the 
      rest of the contents of the query or datum.
    '''

    if query[-1] == True and datum[-1] == False:
        return None, None
    if len(query) != len(datum):
        return None, None

    unification = []
    subs = dict()

    for i in range(len(query)):
        if query[i] in variables and datum[i] in variables:
            subs[find(query[i], subs)] = datum[i]
        elif query[i] in variables:
            subs[find(query[i], subs)] = datum[i]
        elif datum[i] in variables:
            subs[find(datum[i], subs)] = query[i]
        elif query[i] != datum[i]:
            return None, None
    for word in query:
        unification.append(find(word, subs))

    return unification, subs

def apply(rule, goals, variables):
    '''
    @param rule: A rule that is being tested to see if it can be applied
      This function should not modify rule; consider deepcopy.
    @param goals: A list of propositions against which the rule's consequent will be tested
      This function should not modify goals; consider deepcopy.
    @param variables: list of strings that should be treated as variables

    Rule application succeeds if the rule's consequent can be unified with any one of the goals.
    
    @return applications: a list, possibly empty, of the rule applications that
       are possible against the present set of goals.
       Each rule application is a copy of the rule, but with both the antecedents 
       and the consequent modified using the variable substitutions that were
       necessary to unify it to one of the goals. Note that this might require 
       multiple sequential substitutions, e.g., converting ('x','eats','squirrel',False)
       based on subs=={'x':'a', 'a':'bobcat'} yields ('bobcat','eats','squirrel',False).
       The length of the applications list is 0 <= len(applications) <= len(goals).  
       If every one of the goals can be unified with the rule consequent, then 
       len(applications)==len(goals); if none of them can, then len(applications)=0.
    @return goalsets: a list of lists of new goals, where len(newgoals)==len(applications).
       goalsets[i] is a copy of goals (a list) in which the goal that unified with 
       applications[i]['consequent'] has been removed, and replaced by 
       the members of applications[0]['antecedents'].

    Example:
    rule={
      'antecedents':[['x','is','nice',True],['x','is','hungry',False]],
      'consequent':['x','eats','squirrel',False]
    }
    goals=[
      ['bobcat','eats','squirrel',False],
      ['bobcat','visits','squirrel',True],
      ['bald eagle','eats','squirrel',False]
    ]
    variables=['x','y','a','b']

    applications, newgoals = submitted.apply(rule, goals, variables)

    applications==[
      {
        'antecedents':[['bobcat','is','nice',True],['bobcat','is','hungry',False]],
        'consequent':['bobcat','eats','squirrel',False]
      },
      {
        'antecedents':[['bald eagle','is','nice',True],['bald eagle','is','hungry',False]],
        'consequent':['bald eagle','eats','squirrel',False]
      }
    ]
    newgoals==[
      [
        ['bobcat','visits','squirrel',True],
        ['bald eagle','eats','squirrel',False]
        ['bobcat','is','nice',True],
        ['bobcat','is','hungry',False]
      ],[
        ['bobcat','eats','squirrel',False]
        ['bobcat','visits','squirrel',True],
        ['bald eagle','is','nice',True],
        ['bald eagle','is','hungry',False]
      ]
    '''
    applications, goalsets = [], []
    for goal in goals:
        newgoal = list(goals[:])
        application = dict()
        application['antecedents'] = []
        #application['text'] = rule['text']
        _, subs = unify(goal, rule['consequent'], variables)
        if subs is None:
            continue
        newgoal.remove(goal)
        for antecedent in rule['antecedents']:
            antecedent_copy = antecedent[:]
            for i in range(len(antecedent_copy)):
                antecedent_copy[i] = find(antecedent_copy[i], subs)
            newgoal.append(tuple(antecedent_copy))
            application['antecedents'].append(antecedent_copy)
        consequent_copy = rule['consequent'][:]
        for i in range(len(consequent_copy)):
            consequent_copy[i] = find(consequent_copy[i], subs)
        application['consequent'] = consequent_copy
        applications.append(application)
        goalsets.append(tuple(newgoal))

    return applications, goalsets


def backward_chain(query, rules, variables):
    '''
    @param query: a proposition, you want to know if it is true
    @param rules: dict mapping from ruleIDs to rules

    @return proof (list): a list of rule applications
      that, when read in sequence, conclude by proving the truth of the query.
      If no proof of the query was found, you should return proof=None.
    '''
    proof = []
    q = queue.Queue()
    q.put((tuple(query),))
    parent = dict()
    app = dict()
    parent[(tuple(query),)] = None
    while q.qsize() > 0:
        goalset = q.get()
        for rule in rules.values():
            applications, newgoals = apply(rule, goalset, variables)
            for newgoal, application in zip(newgoals, applications):
                application['text'] = rule['text']
                if not newgoal:
                    parent["END"] = tuple(goalset)
                    app["END"] = application
                    break
                q.put(tuple(newgoal))
                parent[tuple(newgoal)] = tuple(goalset)
                app[tuple(newgoal)] = application
    if "END" not in parent:
        return None

    x = "END"
    while x != (tuple(query),):
        proof.append(app[x])
        x = parent[x]
    return proof

def test():
    # some tests
    standardized_rules, variables = standardize_variables({'triple1': {'text': 'The bald eagle chases the bear.', 'antecedents': [], 'consequent': ['bald eagle', 'chases', 'bear', True]}, 'triple2': {'text': 'The bald eagle is red.', 'antecedents': [], 'consequent': ['bald eagle', 'is', 'red', True]}, 'triple3': {'text': 'The bear visits the bald eagle.', 'antecedents': [], 'consequent': ['bear', 'visits', 'bald eagle', True]}, 'rule1': {'text': 'If something visits the bear then the bear sees the bald eagle.', 'antecedents': [['something', 'visits', 'bear', True]], 'consequent': ['bear', 'sees', 'bald eagle', True]}, 'rule2': {'text': 'If something sees the bear then the bear chases the bald eagle.', 'antecedents': [['something', 'sees', 'bear', True]], 'consequent': ['bear', 'chases', 'bald eagle', True]}, 'rule3': {'text': 'If something sees the bald eagle then the bald eagle sees the bear.', 'antecedents': [['something', 'sees', 'bald eagle', True]], 'consequent': ['bald eagle', 'sees', 'bear', True]}, 'rule4': {'text': 'If something visits the bald eagle then the bald eagle sees the bear.', 'antecedents': [['something', 'visits', 'bald eagle', True]], 'consequent': ['bald eagle', 'sees', 'bear', True]}, 'rule5': {'text': 'If something sees the bald eagle then the bald eagle chases the bear.', 'antecedents': [['something', 'sees', 'bald eagle', True]], 'consequent': ['bald eagle', 'chases', 'bear', True]}, 'rule6': {'text': 'If something sees the bald eagle and the bald eagle sees the bear then the bear chases the bald eagle.', 'antecedents': [['something', 'sees', 'bald eagle', True], ['bald eagle', 'sees', 'bear', True]], 'consequent': ['bear', 'chases', 'bald eagle', True]}})
    print(variables)

if __name__ == "__main__":
    test()