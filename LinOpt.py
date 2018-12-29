import pulp
import quadprog
import numpy

dealers = ['X', 'Y', 'Z']
variable_costs = {'X': 500, 'Y': 350, 'Z': 450}
fixed_costs = {'X': 4000, 'Y': 2000, 'Z': 6000}

# These are the 2 variables we are solving for
quantities = pulp.LpVariable.dicts('quantity', dealers, lowBound=0, cat=pulp.LpInteger)
is_orders = pulp.LpVariable.dicts('orders', dealers, cat=pulp.LpBinary)

if False:
    model = pulp.LpProblem('A cost minimization problem', pulp.LpMinimize)

    # Notice that we are multiplying two variables, which makes this a non-linear optimization problem
    # e.g f(x,y) = x*y is non-linear
    model += sum([variable_costs[i]*quantities[i] + fixed_costs[i]*is_orders[i] for i in dealers]), \
            "Minimize portfolio cost"
    model += sum([quantities[i] for i in dealers]) == 150, \
            "Total contracts required cannot be greater than 150"
    model += 30 <= quantities['X'] <= 100, 'X cannot sell less than 30 or more than 100'
    model += 30 <= quantities['Y'] <= 90, 'Y cannot sell less than 30 or more than 90'
    model += 30 <= quantities['Z'] <= 70, 'Z cannot sell less than 30 or more than 70'

    model.solve()
else:
    model = pulp.LpProblem('A cost minimization problem', pulp.LpMinimize)

    # Notice here that we are not multiplying variables, so the problem is still linear
    # f(x,y) = A*x + B*y with A,B constants
    model += sum(
        [variable_costs[i]*quantities[i] + fixed_costs[i]*is_orders[i] for i in dealers]
    ), 'Minimize portfolio cost'
    model += sum([quantities[i] for i in dealers]) == 150, 'Total contracts required cannot be more than 150'

    # Since we removed is_orders from the outer expression in the object function,
    # we need to add it to the constraints. Remember that is_orders is 0 or 1, therefore
    # is is flag stating if we will buy from a certain dealer or not.
    model += is_orders['X']*30 <= quantities['X'] <= is_orders['X']*100, 'Boundary of total volume of X'
    model += is_orders['Y']*30 <= quantities['Y'] <= is_orders['Y']*90, 'Boundary of total volume of Y'
    model += is_orders['Z']*30 <= quantities['Z'] <= is_orders['Z']*70, 'Boundary of total volume of Z'

    model.solve()

print ('Minimization results:')
for v in model.variables():
    print(v, v.varValue)


def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
    qp_G = .5 * (P + P.T) # make sure P is symmetric
    qp_a = -q
    if A is not None:
        qp_C = -numpy.vstack([A,G]).T
        qp_b = -numpy.hstack([b,h])
        meq = A.shape[0]
    else: # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]

