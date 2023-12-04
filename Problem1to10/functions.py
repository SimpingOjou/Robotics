import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def denavit():
    theta = [sp.symbols(f'theta{i}') for i in range(1, 5)]
    #dh = sp.Matrix([[theta[0], 50e-3, 0, sp.pi/2],
    #                [theta[1],  0,  93e-3, 0],
    #                [theta[2],  0,  93e-3, 0],
    #                [theta[3],  0,  50e-3, 0]])    
    dh = sp.Matrix([[theta[0], 50, 0, sp.pi/2],
                    [theta[1],  0,  93, 0],
                    [theta[2],  0,  93, 0],
                    [theta[3],  0,  50, 0]])
    
    #               theta_i     d_i   a_i   alpha_i
    #print("DH:")
    #sp.pretty_print(dh)
    #print("\n-----------------------\n")

    return dh, theta

def homogeneous():
    dh, theta = denavit()

    T = [sp.Matrix([
            [sp.cos(dh[i, 0]), -sp.sin(dh[i, 0])*sp.cos((dh[i, 3])), sp.sin(dh[i, 0])*sp.sin((dh[i, 3])), dh[i, 2]*sp.cos(dh[i, 0])],
            [sp.sin(dh[i, 0]), sp.cos(dh[i, 0])*sp.cos((dh[i, 3])), -sp.cos(dh[i, 0])*sp.sin((dh[i, 3])), dh[i, 2]*sp.sin(dh[i, 0])],
            [0, sp.sin(dh[i, 3]), sp.cos(dh[i, 3]), dh[i, 1]],
            [0, 0, 0, 1]
        ]) for i in range(len(theta))]

    T45 = sp.eye(4)
    T45[0, 3] = -15
    T45[1, 3] = 45
    T.append(T45)
    return T

def print_homogeneous(T):
    for i,matrix in enumerate(T):
        print(f'T{i}{i+1}')
        sp.pretty_print(matrix)
        print("\n-----------------------\n")

def inverse(x,y,z):
    dh, theta = denavit()

    x_c = x
    y_c = y
    z_c = z

    # theta 1
    dh[0, 0] = sp.atan2(y_c,x_c)

    # theta 3
    r = sp.sqrt(x_c**2 + y_c**2)
    s = z_c - dh[0, 1]
    c3 = (r**2 + s**2 - dh[1,2]**2 - dh[2,2]**2) / (2 * dh[1,2] * dh[2,2])
    dh[2, 0] = sp.atan2(sp.sqrt(1-c3**2), c3) # elbow up
    t3ed = sp.atan2(-sp.sqrt(1-c3**2), c3) # elbow down

    # theta 2
    dh[1, 0] = sp.atan2(s, r) - sp.atan2((dh[2,2]*sp.sin(dh[2, 0])), (dh[1,2] + dh[2,2]*sp.cos(dh[2, 0])))
    t2ed = sp.atan2(s, r) - sp.atan2((dh[2,2]*sp.sin(t3ed)), (dh[1,2] + dh[2,2]*sp.cos(t3ed)))
    # theta 4
    dh[3, 0] = - dh[1, 0] - dh[2, 0]
    t4ed = - t2ed - t3ed

    eu = [dh[0,0], dh[1,0], dh[2,0], dh[3,0]]
    ed = [dh[0,0], t2ed, t3ed, t4ed]
    
    return eu, ed

def jacobian(t1, t2, t3, t4):
    theta1, theta2, theta3, theta4 = sp.symbols('theta1 theta2 theta3 theta4')

    dh, angle = denavit()
    T = homogeneous()
    Trans = [sp.symbols(f'T{i}{i+1}') for i,j in enumerate(T)]

    for i,matrix in enumerate(T):
        Trans[i] = matrix

    # parameter extraction
    T01 = Trans[0]
    T02 = T01*Trans[1]
    T03 = T02*Trans[2]
    T04 = T03*Trans[3]
    T05 = T04*Trans[4]

    z0 = sp.Matrix([0, 0, 1])
    z1 = T01[0:3,2]
    z2 = T02[0:3,2]
    z3 = T03[0:3,2]
    z4 = T04[0:3,2]
    z5 = T05[0:3,2]

    O0 = sp.Matrix([0, 0, 0])
    O1 = T01[0:3,3]
    O2 = T02[0:3,3]
    O3 = T03[0:3,3]
    O4 = T04[0:3,3]
    O5 = T05[0:3,3]

    # J4 and J5
    J4v = sp.Matrix([[z0.cross(O4-O0).transpose()], [z1.cross(O4-O1).transpose()], [z2.cross(O4-O2).transpose()], [z3.cross(O4-O3).transpose()]]).transpose()
    J4w = sp.Matrix([z0.transpose(), z1.transpose(), z2.transpose(), z3.transpose()]).transpose()
    J4 = sp.Matrix.vstack(J4v, J4w)
    J4 = J4.subs({theta1: t1, theta2: t2, theta3: t3, theta4: t4})

    J5v = sp.Matrix([[z0.cross(O5-O0).transpose()], [z1.cross(O5-O1).transpose()], [z2.cross(O5-O2).transpose()], [z3.cross(O5-O3).transpose()]]).transpose()
    J5w = J4w
    J5 = sp.Matrix.vstack(J5v, J5w)
    J5 = J5.subs({theta1: t1, theta2: t2, theta3: t3, theta4: t4})

    return J4.applyfunc(lambda x: 0 if abs(x) < 1e-8 else x), J5.applyfunc(lambda x: 0 if abs(x) < 1e-8 else x)

def anal_jacobian():
    dh, angle = denavit()
    T = homogeneous()
    Trans = [sp.symbols(f'T{i}{i+1}') for i,j in enumerate(T)]

    for i,matrix in enumerate(T):
        Trans[i] = matrix

    # parameter extraction
    T01 = Trans[0]
    T02 = T01*Trans[1]
    T03 = T02*Trans[2]
    T04 = T03*Trans[3]
    T05 = T04*Trans[4]

    z0 = sp.Matrix([0, 0, 1])
    z1 = T01[0:3,2]
    z2 = T02[0:3,2]
    z3 = T03[0:3,2]
    z4 = T04[0:3,2]
    z5 = T05[0:3,2]

    O0 = sp.Matrix([0, 0, 0])
    O1 = T01[0:3,3]
    O2 = T02[0:3,3]
    O3 = T03[0:3,3]
    O4 = T04[0:3,3]
    O5 = T05[0:3,3]

    # J4 and J5
    J4v = sp.Matrix([[z0.cross(O4-O0).transpose()], [z1.cross(O4-O1).transpose()], [z2.cross(O4-O2).transpose()], [z3.cross(O4-O3).transpose()]]).transpose()
    J4w = sp.Matrix([z0.transpose(), z1.transpose(), z2.transpose(), z3.transpose()]).transpose()
    J4 = sp.Matrix.vstack(J4v, J4w)

    J5v = sp.Matrix([[z0.cross(O5-O0).transpose()], [z1.cross(O5-O1).transpose()], [z2.cross(O5-O2).transpose()], [z3.cross(O5-O3).transpose()]]).transpose()
    J5w = J4w
    J5 = sp.Matrix.vstack(J5v, J5w)

    return J4.applyfunc(sp.trigsimp), J5.applyfunc(sp.trigsimp)

def big_jacobian(t1, t2, t3, t4):
    theta1, theta2, theta3, theta4 = sp.symbols('theta1 theta2 theta3 theta4')

    dh, angle = denavit()
    T = homogeneous()
    Trans = [sp.symbols(f'T{i}{i+1}') for i,j in enumerate(T)]

    for i,matrix in enumerate(T):
        Trans[i] = matrix

    # parameter extraction
    T01 = Trans[0]
    T02 = T01*Trans[1]
    T03 = T02*Trans[2]
    T04 = T03*Trans[3]
    T05 = T04*Trans[4]

    z0 = sp.Matrix([0, 0, 1])
    z1 = T01[0:3,2]
    z2 = T02[0:3,2]
    z3 = T03[0:3,2]
    z4 = T04[0:3,2]
    z5 = T05[0:3,2]

    O0 = sp.Matrix([0, 0, 0])
    O1 = T01[0:3,3]
    O2 = T02[0:3,3]
    O3 = T03[0:3,3]
    O4 = T04[0:3,3]
    O5 = T05[0:3,3]

    # J1 to J5
    J4v = sp.Matrix([[z0.cross(O4-O0).transpose()], [z1.cross(O4-O1).transpose()], [z2.cross(O4-O2).transpose()], [z3.cross(O4-O3).transpose()]]).transpose()
    J4w = sp.Matrix([z0.transpose(), z1.transpose(), z2.transpose(), z3.transpose()]).transpose()
    J5v = sp.Matrix([[z0.cross(O5-O0).transpose()], [z1.cross(O5-O1).transpose()], [z2.cross(O5-O2).transpose()], [z3.cross(O5-O3).transpose()]]).transpose()
    J5w = J4w

    J1 = sp.Matrix.vstack(J4v[:, 0], J4w[:, 0])
    J1 = J1.subs({theta1: t1, theta2: t2, theta3: t3, theta4: t4})

    J2 = sp.Matrix.vstack(J4v[:, 0:2], J4w[:, 0:2])
    J2 = J2.subs({theta1: t1, theta2: t2, theta3: t3, theta4: t4})

    J3 = sp.Matrix.vstack(J4v[:, 0:3], J4w[:, 0:3])    
    J3 = J3.subs({theta1: t1, theta2: t2, theta3: t3, theta4: t4})

    J4 = sp.Matrix.vstack(J4v, J4w)
    J4 = J4.subs({theta1: t1, theta2: t2, theta3: t3, theta4: t4})

    J5 = sp.Matrix.vstack(J5v, J5w)
    J5 = J5.subs({theta1: t1, theta2: t2, theta3: t3, theta4: t4})

    J1 = J1.applyfunc(lambda x: 0 if abs(x) < 1e-8 else x)
    J2 = J2.applyfunc(lambda x: 0 if abs(x) < 1e-8 else x)
    J3 = J3.applyfunc(lambda x: 0 if abs(x) < 1e-8 else x)
    J4 = J4.applyfunc(lambda x: 0 if abs(x) < 1e-8 else x)
    J5 = J5.applyfunc(lambda x: 0 if abs(x) < 1e-8 else x)
    

    return J1, J2, J3, J4, J5

def anal_jacobian_10():
    theta = [sp.symbols(f'theta{i}') for i in range(1, 4)]
    dhcm = sp.Matrix([
        [theta[0], 30, 0, sp.pi/2],
        [theta[1],  0,  63, 0],
        [theta[2],  0,  63, 0],
    ])
    T = homogeneous()
    #    theta_i     d_i   a_i   alpha_i
    Tcm = [sp.Matrix([
            [1, 0, 0, 0],
             [0, 1, 0, -20],
             [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]),
        sp.Matrix([
            [1, 0, 0, 0],
             [0, 1, 0, -30],
             [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]),
        sp.Matrix([
            [1, 0, 0, 0],
             [0, 1, 0, -30],
             [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]),
        sp.Matrix([
            [1, 0, 0, -25],
             [0, 1, 0, 15],
             [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]),]
    
    # parameter extraction
    T02 = T[0]*T[1]
    T03 = T02*T[2]
    T04 = T03*T[3]
    T01cm = T[0]*Tcm[0]
    T02cm = T02*Tcm[1]
    T03cm = T03*Tcm[2]
    T04cm = T04*Tcm[3]

    z0 = sp.Matrix([0, 0, 1])
    z1 = T01cm[0:3,2]
    z2 = T02cm[0:3,2]
    z3 = T03cm[0:3,2]
    z4 = T04cm[0:3,2]

    O0 = sp.Matrix([0, 0, 0])
    O1 = T01cm[0:3,3]
    O2 = T02cm[0:3,3]
    O3 = T03cm[0:3,3]
    O4 = T04cm[0:3,3]

    # J1 to J5
    J4v = sp.Matrix([[z0.cross(O4-O0).transpose()], [z1.cross(O4-O1).transpose()], [z2.cross(O4-O2).transpose()], [z3.cross(O4-O3).transpose()]]).transpose()
    J4w = sp.Matrix([z0.transpose(), z1.transpose(), z2.transpose(), z3.transpose()]).transpose()

    J1 = sp.Matrix.vstack(J4v[:, 0], J4w[:, 0]).applyfunc(sp.trigsimp)
    J2 = sp.Matrix.vstack(J4v[:, 0:2], J4w[:, 0:2]).applyfunc(sp.trigsimp)
    J3 = sp.Matrix.vstack(J4v[:, 0:3], J4w[:, 0:3]).applyfunc(sp.trigsimp) 
    J4 = sp.Matrix.vstack(J4v, J4w).applyfunc(sp.trigsimp)

    return J1, J2, J3, J4

def calc_dq(J, v):
    J_anal = (J.T*J).inv()*J.T # pseudoinverse of an underactuated arm (joints<6)
    dq = J_anal * v 

    return dq.applyfunc(lambda x: 0 if abs(x) < 1e-8 else x)

def computeT04(t1, t2, t3, t4):
    theta1, theta2, theta3, theta4 = sp.symbols('theta1 theta2 theta3 theta4')

    T = homogeneous()
    T04 = T[0]*T[1]*T[2]*T[3]
    
    return T04.subs({theta1: t1, theta2: t2, theta3: t3, theta4: t4}).evalf(5)
    