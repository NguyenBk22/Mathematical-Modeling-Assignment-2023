import numpy as np
from gamspy import Container,Variable, Equation, Model, Set, Sum, Parameter

def generate_random_demand(n):
    # Simulate random demand vector ω bin(10,0.5)
    demand = np.random.binomial(10, 0.5, n)
    return demand

def generate_random_data(n, m,MAX_VALUE):
    np.random.seed()
    # Simulate data vector b, l, q, s and matrix A
    l_arr = np.random.randint(10, 16, size=n)    # Lower bounds for variables y
    q_arr = np.random.randint(115, 121, size=n)   # Upper bounds for variables y
    d_arr1 = generate_random_demand(n)              #random D1
    d_arr2 = generate_random_demand(n)              #random D2
    A = np.random.randint(0, 6, size=(n, m))    # Coefficients matrix
    b_arr = np.zeros(m, dtype=int)
    s_arr = np.zeros(m, dtype=int)

    s_arr[:] = np.random.randint(1, 4, size=m) 
    b_arr[:] = np.random.randint(s_arr + 1, 8, size=m)

    return l_arr, q_arr, d_arr1, d_arr2, A, b_arr, s_arr
    


def build_MHH(ps,n,M):
    # Create a GAMSPy container
    l_arr, q_arr, d_arr1, d_arr2, A, b_arr, s_arr =generate_random_data(n,M,MAX_VALUE)
    for row in A:
        print("  ".join(f"{value:4}" for value in row)) 
    print("s:", s_arr)
    print("l:", l_arr)
    print("q:", q_arr)
    print("b:", b_arr)
    print("d1:", d_arr1)
    print("d2:", d_arr2)
    print("Matrix A : ")
    
    m = Container(delayed_execution=True)
    
    #Set
    Part = Set(
        m,
        name="Part",
        records=[f"Part {j}" for j in range(1, M + 1)]
        )
    Products = Set(
        m,
        name="Products",
        records=[f"Products {i}" for i in range(1, n + 1)]
        )
    
    
    #Parameter
    s = Parameter(m, name="s", domain=[Part], records=np.array(s_arr))
    b = Parameter(m, name="b", domain=[Part], records=np.array(b_arr))
    a = Parameter(m, name="a", domain=[Products, Part], records=np.array(A))
    l = Parameter(m, name="l", domain=[Products], records=np.array(l_arr))
    q = Parameter(m, name="q", domain=[Products], records=np.array(q_arr))
    d1 = Parameter(m, name="d1", domain=[Products], records=np.array(d_arr1))
    d2 = Parameter(m, name="d2", domain=[Products], records=np.array(d_arr2))
   

    # Define variables
    x = Variable(m, name="x", domain=[Part])
    y1 = Variable(m, name="y1", domain=[Part])
    y2 = Variable(m, name="y2", domain=[Part])
    z1 = Variable(m, name="z1", domain=[Products])
    z2 = Variable(m, name="z2", domain=[Products])
    result = Variable(m, name="result")

    # Define equations
    eq1 = Equation(m, name="eq1", domain=[Products])
    eq2 = Equation(m, name="eq2", domain=[Products])
    eq3 = Equation(m, name="eq3", domain=[Part])
    eq4 = Equation(m, name="eq4", domain=[Part])
    eq5 = Equation(m, name="eq5", domain=[Part])
    eq6 = Equation(m, name="eq6", domain=[Part])
    eq7 = Equation(m, name="eq7")
    
    eq1[Products] = 0 <= z1[Products] <= d1[Products]
    eq2[Products] = 0 <= z2[Products] <= d2[Products]
    eq3[Part] = x[Part] - Sum(Products, a[Products, Part] * z1[Products]) == y1[Part]
    eq4[Part] = x[Part] - Sum(Products, a[Products, Part] * z2[Products]) == y2[Part]
    eq5[Part] = y1[Part] >= 0
    eq6[Part] = y2[Part] >= 0
    
    eq7[...] = (
        Sum(Part, b[Part] * x[Part])
        + ps * (Sum(Products, (l[Products] - q[Products]) * z1[Products]) - Sum(Part, s[Part] * y1[Part]))
        + ps * (Sum(Products, (l[Products] - q[Products]) * z2[Products]) - Sum(Part, s[Part] * y2[Part]))
    ) == result

    # Define the model    
    PROBLEM1 = Model(
        m,
        name="PROBLEM1",
        equations=m.getEquations(),
        problem="lp",
        sense="min",
        objective=result,
    )       
    PROBLEM1.solve()   # Solve the model
    # Print results
    print("Objective Function Value: ",round(result.toValue(), 4))
     # return x, y, z
    print(x.records)
    print(y1.records)
    print(y2.records)
    print(z1.records)
    print(z2.records)
   

def main():
    ps = 1/2  # Scenario probability
    n=8 # Number of Product
    m = 5  # Number of Parts to be ordered before Productsion
    S = 2  # Number of scenarios
    
    build_MHH(ps,n,m)

    # # Generate random data
    # b, l, q, s, A = generate_random_data(n, m, S, ps)
    # demand = generate_random_demand(n)

    # Build and solve the model
    # x, y, z = build_model(n, m, S, ps, b, l, q, s, A, demand)

    # Print results
    # print("Objective Function Value: ", result(z.toValue(), 4))
    # print("Decision Variables x: ", [round(x[i].toValue(), 4) for i in range(n)])
    # print("Decision Variables y: ", [round(y[j].toValue(), 4) for j in range(m)])

if __name__ == "__main__":
    MAX_VALUE=50
    main()
    
    
