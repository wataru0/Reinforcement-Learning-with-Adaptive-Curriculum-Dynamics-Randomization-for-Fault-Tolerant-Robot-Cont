import numpy as np

def my_mulQuat(p, q):
    qp = np.zeros(4)
    pw = p[0]
    px = p[1]
    py = p[2]
    pz = p[3]

    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]

    qp[0] = -qx*px-qy*py-qz*pz+qw*pw 
    qp[1] = qw*px-qz*py+qy*pz+qx*pw
    qp[2] = qz*px+qw*py-qx*pz+qy*pw
    qp[3] = -qy*px+qx*py+qw*pz+qz*pw
    return qp


if __name__=='__main__':
    main()