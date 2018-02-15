import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='g2o path', default='')

args = parser.parse_args()

PATH=args.path


sin=np.sin
cos=np.cos
pi=np.pi

def q2e(x, y, z, w):
    ysqr = y*y
    
    t0 = +2.0 * (w * x + y*z)
    t1 = +1.0 - 2.0 * (x*x + ysqr)
    X = math.degrees(math.atan2(t0, t1))
    
    t2 = +2.0 * (w*y - z*x)
    t2 =  1 if t2 > 1 else t2
    t2 = -1 if t2 < -1 else t2
    Y = math.degrees(math.asin(t2))
    
    t3 = +2.0 * (w * z + x*y)
    t4 = +1.0 - 2.0 * (ysqr + z*z)
    Z = math.degrees(math.atan2(t3, t4))
    
    
    return X, Y, Z 


def Rcam():
    rcam=Rz(-pi/2)@Rx(pi/2)

    return rcam

def Rx(x):
    
    x_m=np.array(([1,0,0],[0,cos(x),-sin(x)],[0,sin(x),cos(x)]))
    return x_m 

def Ry(y):

    y_m=np.array(([cos(y),0,sin(y)],[0,1,0],[-sin(y),0,cos(y)]))
    return y_m 


def Rz(z):
    

    z_m=np.array(([cos(z),-sin(z),0],[sin(z),cos(z),0],(0,0,1)))
    return z_m 

def rotation(x,y,z):
   
    z_m=np.array(([cos(z),-sin(z),0],[sin(z),cos(z),0],[0,0,1]))
    y_m=np.array(([cos(y),0,sin(y)],[0,1,0],[-sin(y),0,cos(y)]))
    x_m=np.array(([1,0,0],[0,cos(x),-sin(x)],[0,sin(x),cos(x)]))
    return np.dot(np.dot(z_m,y_m),x_m) 

def Rxdot(x):

    x_m=np.array(([0,0,0],[0,-sin(x),-cos(x)],[0,cos(x),-sin(x)]))
    return x_m 

def Rydot(y):
    
    y_m=np.array(([-sin(y),0,cos(y)],[0,0,0],[-cos(y),0,-sin(y)]))
    return y_m 

def Rzdot(z):
    
    z_m=np.array(([-sin(z),-cos(z),0],[cos(z),-sin(z),0],[0,0,0]))
    return z_m 

def J(pos,angles, lmark):
    pos=np.asarray(pos)
    lmark=np.asarray(lmark)
    j=np.zeros((3,6))
    j[:,:3]=-(Rcam()).T@(Rx(angles[0])@Ry(angles[1])@Rz(angles[2])).T
    j[:,3]=(Rcam()).T@(Rxdot(angles[0])@Ry(angles[1])@Rz(angles[2])).T@(lmark-pos)
    j[:,4]=(Rcam()).T@(Rx(angles[0])@Rydot(angles[1])@Rz(angles[2])).T@(lmark-pos)
    j[:,5]=(Rcam()).T@(Rx(angles[0])@Ry(angles[1])@Rzdot(angles[2])).T@(lmark-pos)
    return j

def Jproj(cam_p):
    x=cam_p[0]
    y=cam_p[1]
    z=cam_p[2]
    j_proj=np.array([[-1/z,0,(x/(z**2))],[0,-1/z,(y/(z**2))],[0,0,-1]])
    return j_proj

def h(pos, angles, landmark):
    pos=np.asarray(pos)
    landmark=np.asarray(landmark)
    z=(Rcam()).T@((Rx(angles[0])@Ry(angles[1])@Rz(angles[2])).T@(landmark-pos)-np.array([0,0,0.3]))
    return z

def camera(pos, angles, landmark):
    p_cam=h(pos, angles, landmark)
    x_cam=p_cam[0]
    y_cam=p_cam[1]
    z_cam=p_cam[2]
    x_cam/=-z_cam
    y_cam/=-z_cam
    out=np.array([x_cam+0.5, y_cam+0.5, -z_cam])
    return out


#state transiction jabocians
def A(r_state, inp):
    r_state=np.asarray(r_state)
    inp=np.asarray(inp)
    ang1=r_state[3]
    ang2=r_state[4]
    ang3=r_state[5]
    A_mat=np.eye(6)
    A_mat[0:3,3]=Rxdot(ang1)@Ry(ang2)@Rz(ang3)@inp[:3]
    A_mat[0:3,4]=Rx(ang1)@Rydot(ang2)@Rz(ang3)@inp[:3]
    A_mat[0:3,5]=Rx(ang1)@Ry(ang2)@Rzdot(ang3)@inp[:3]
    return A_mat

def B(r_state, inp):
    r_state=np.asarray(r_state)
    inp=np.asarray(inp)
    B_mat=np.eye(6)
    B_mat[:3,:3]=rotation(r_state[3],r_state[4],r_state[5])
    return B_mat

vertices={}

with open(PATH) as file:
    for line in file:
        if("VERTEX_TRACKXYZ") in line:
            parts=line.split(" ")
            vertices[int(parts[1])]=[float(parts[2]), float(parts[3]), float(parts[4])]


def observations(id):
    #Return a dictionary of observations
    obs={}
    with open(PATH) as file:
        for line in file:
            if("EDGE_PROJECT_DEPTH " + str(id)) in line:
                parts=line.split(" ")
                obs[int(parts[2])]=[float(parts[4]), float(parts[5]), float(parts[6])]
    return obs

def transition(id):
    #Returns the list of x y z and angles movement
    tr=[]
    with open(PATH) as file:
        for line in file:
            if("EDGE_SE3:QUAT " + str(id-1)) in line:
                parts=line.split(" ")
                eulers=q2e(float(parts[6]), float(parts[7]), float(parts[8]), float(parts[9]))
                tr=[float(parts[3]), float(parts[4]), float(parts[5]), eulers[0]*2*pi/360, eulers[1]*2*pi/360, eulers[2]*2*pi/360]
    return np.asarray(tr)

def ground_truths(show=True):
    #Return a dictionary of observations
    g_t={}
    
    with open(PATH) as file:
        for line in file:
            if("VERTEX_SE3:QUAT") in line:
                parts=line.split(" ")
                g_t[int(parts[1])]=[float(parts[2]), float(parts[3]), float(parts[4])]
                plt.scatter(float(parts[2]), float(parts[3]), color="red")
    if show:
        plt.show()
    return g_t

def no_correction(show=True):
    # shows the trajectoty with only encoder informations
    r_state=np.array([0.,0.,0.,0.,0.,0.])
    r_var=np.eye(6)*0.0000001
    index=1000

    num_obs_up=1

    for k in range(100):
        index+=1
        rotated_trans=rotation(r_state[3],r_state[4],r_state[5])@transition(index)[:3]
       # print("transition {} and rotated trans {}".format(transition(index), rotated_trans))
       # r_state+=transition(index)
        r_state[:3]+=rotated_trans
        r_state[3:6]+=transition(index)[3:6]
        r_var+=var_tr

        plt.scatter(r_state[0],r_state[1], color="green")
    if show:
        plt.show()

var_tr=np.eye(6)
var_tr[0,0]= 100 
var_tr[1,1]= 100
var_tr[2,2]= 100
var_tr[3,3]=10000
var_tr[4,4]=10000
var_tr[5,5]=10000

var_tr=np.linalg.inv(var_tr)

var_obs=np.eye(3)
var_obs[0,0]= 100
var_obs[1,1]= 100
var_obs[2,2]= 100


var_obs=np.linalg.inv(var_obs)


def min_row_col(a):
    #finds the values that are minimum of both col and row
    mins=[]
    for i in range(a.shape[0]):
        min_row=(i,np.argmin(a[i,:],axis=0))
    #    print("min row {}:".format(i), min_row[1])
        min_col=(np.argmin(a[:,min_row[1]]),min_row[1])
    #    print("min col of min row {}, {}:".format(i, min_row), min_col)
        if min_row==min_col:
            mins.append(min_row)
    return mins


def pred_assoc(obs, r_state):
    #Predict association based on euclidean distance.

    poss_obs=[]
    for key in vertices:
        poss_obs.append([key, camera(r_state[:3], r_state[3:6], vertices[key])])
    #    print(key, camera(r_state[:3], r_state[3:6], vertices[key]))
    pred2realkey={}
    
    for keyobs in obs:
        min_err=10000
        best_key=0
        for ver in range(len(poss_obs)):
            err=np.linalg.norm(obs[keyobs] - poss_obs[ver][1])
            #print(err)
            if err<min_err:
                min_err=err
                best_key=poss_obs[ver][0]
        print("minimum error in prediction ",min_err)
    #    print("best key ",best_key,"real key ", keyobs)
        if min_err<1. and not best_key in pred2realkey.values():
            pred2realkey[keyobs]=best_key
    
    return pred2realkey 



def pred_assoc_cost(obs, r_state):
    #Predicts association with gated and best friends heuristics based on association matrix.

    poss_obs=[]
    
    for key in vertices:
        poss_obs.append([key, camera(r_state[:3], r_state[3:6], vertices[key])])
    #    print(key, camera(r_state[:3], r_state[3:6], vertices[key]))
    pred2realkey={}
    cost_matrix=np.zeros(shape=(len(obs),len(poss_obs)))
    keys_matrix={}
   # print(cost_matrix.shape)
    
    a=0
    for keyobs in obs:
        b=0
        min_err=10000
        best_key=0
        for ver in range(len(poss_obs)):
            err=np.linalg.norm(obs[keyobs] - poss_obs[ver][1])
            cost_matrix[a,b]=err
            keys_matrix[a,b]=[keyobs, poss_obs[ver][0]]
            
            b+=1
        a+=1
            #print(err)
            
            
    
    for i in range(cost_matrix.shape[0]):
        min_row = np.argmin(cost_matrix[i,:],axis=0)
        min_col = np.argmin(cost_matrix[:,min_row])
        if min_row==min_col:
            pred2realkey[keys_matrix[min_row,min_col][0]]=keys_matrix[min_row,min_col][1]
    
    mins=min_row_col(cost_matrix)
  #  print(mins)
    for j in range(len(mins)):
        #print("Cost of the mins: ", cost_matrix[mins[j][0],mins[j][1]])
        if cost_matrix[mins[j][0],mins[j][1]]<0.7:
            pred2realkey[keys_matrix[mins[j][0],mins[j][1]][0]]=keys_matrix[mins[j][0],mins[j][1]][1]
           # print(cost_matrix[mins[j][0],mins[j][1]])
    
    return pred2realkey 


#Run the simulation.

VERBOSE=0

r_state=np.array([0.,0.,0.,0.,0.,0.])
r_var=np.eye(6)*0.000001
index=1000
#num_obs_up=1
unsup_steps=[np.array([0.,0.,0.,0.,0.,0.])]
est_states=[]

for k in range(100):
    # augment the index and extract the transition
    index+=1
    rotated_trans=rotation(r_state[3],r_state[4],r_state[5])@transition(index)[:3]
    if VERBOSE: print("transition {} and rotated trans {}".format(transition(index), rotated_trans))


    #update the robot state
    r_state[:3]+=rotated_trans
    r_state[3:6]+=transition(index)[3:6]
    r_var=A(r_state, transition(index))@r_var@A(r_state, transition(index)).T
    r_var+=B(r_state, transition(index))@var_tr@B(r_state, transition(index)).T  
    est_states.append(r_state)
    
    #extract the observations at this step
    obs=observations(index)
    if VERBOSE: print("Robot state in {}".format(index))
    if VERBOSE: print(r_state)
    if VERBOSE: print("Observations:")
    if VERBOSE: print(obs)

    hs=[]
    absolutes=[]
    zs=[] 

    #create the pairs observations - predicted obs
    pred_assoc_dict=pred_assoc_cost(obs, r_state)

    if VERBOSE: print("predicted associations cost", pred_assoc_dict)
    for key in pred_assoc_dict:
        real_key=key
        pred_key=pred_assoc_dict[key]
        sing_err=(obs[key]-camera(r_state[:3], r_state[3:6], vertices[pred_key]))
        if VERBOSE: print("modulo errore sing err {}".format(np.linalg.norm(sing_err)))
        if np.linalg.norm(sing_err)<10.:
            hs.append(obs[key])
            absolutes.append(vertices[pred_key])
            zs.append(camera(r_state[:3], r_state[3:6], vertices[pred_key]))
        else:
            if VERBOSE: print("obs esclusa!")
     
    hs=np.asarray(hs)
    absolutes=np.asarray(absolutes)
    zs=np.asarray(zs)
    errs=hs-zs #errors between z and h

    if VERBOSE: print("Modulo errore {}".format(np.linalg.norm(errs)))

    #create jacobian
    JTOT=np.zeros((len(hs)*3,6))

    for i in range(len(hs)):
        jjtot=Jproj(h(r_state[:3], r_state[3:6], absolutes[i]))@J(pos=r_state[:3], angles=r_state[3:6], lmark=absolutes[i])
        JTOT[3*i:3*i+3,:]=jjtot

    JTOT=np.asarray(JTOT)

    var_obs=np.eye(len(hs)*3)*100
    var_obs=np.linalg.inv(var_obs)

    #Kalman gain
    k_gain=(r_var@JTOT.T@(np.linalg.inv(var_obs+ JTOT@r_var@JTOT.T)))

    dx=k_gain@np.array(errs.ravel())

    #State and variance correction
    r_state+=dx
    r_var = (np.eye(6)-k_gain@JTOT)@r_var
    #print(r_var)

    if VERBOSE: print("Reestimated state: {}".format(r_state))

#no_correction(show=False)
#   ground_truths(show=False)
    plt.scatter(r_state[0],r_state[1], color="blue")
  #  plt.scatter(unsup_steps[-1][0],unsup_steps[-1][1], color="green")
  #  plt.savefig("/Users/normand/Desktop/prob_rob_steps/" + str(index) + ".jpg")
print("Estimated trajectory with EKF (blue) and ground truth (red)")
ground_truths(show=False)
plt.show()
print("Estimated without EKF correction (green) and ground truth (red)")
ground_truths(show=False)
no_correction(show=False)
plt.show()