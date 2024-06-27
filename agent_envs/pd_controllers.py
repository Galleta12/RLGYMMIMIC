import numpy as np
import mujoco as mj
from scipy.linalg import cho_solve, cho_factor
from rfc_utils.config import Config

def compute_acceleration(q_error,qdot,C,new_mass,KP,KD):
    
    # we add none on the indexing to add a dimension and avoid errors
    # add a dimension to vectors to fit matrix operation requirements
   
    C = C[:, None]
    q_error = q_error[:, None]
    qdot = qdot[:, None]
    
    # Calculate the proportional and damping forces
    prop_force = KP @ q_error
    damp_force = KD @ qdot
    
    # Combine forces for the equation
    combined_forces = -C - prop_force - damp_force       
    #facotor the mass, to solve the equation    
    chol_factor, lower = cho_factor(new_mass, overwrite_a=True, check_finite=False)
    
    # solve the equation
    #we can also solve it with this line of code, but I will use the cholesky way
    #qdot_dot = np.linalg.solve(new_mass, combined_forces)
    
    qdot_dot = cho_solve((chol_factor, lower), combined_forces, overwrite_b=True, check_finite=False)
    #return it back to one dim array
    return qdot_dot.squeeze()
    
#calculate the new mass,
def calculate_new_mass(M,KD,dt):

    new_mass = M + (KD * dt)

    return new_mass

#initialize corilis, mass and external forces
#and set kp and kd as diagonal matrices
def init_corolis_mass_external_diagonal(data,model,kp,kd):
    #get the centrifugal force
    #dim (nv)
    C = data.qfrc_bias.copy()
    nv = model.nv
    #set the mass
    M = np.zeros((nv, nv))
    mj.mj_fullM(model, M, data.qM)
    KP = np.diag(kp)
    KD = np.diag(kd)
    
    return C,M,KP,KD
def stable_pd_controller(data,model,target_pos,q,qdot,cfg:Config,dt):
    
    #calculate the next q erro, this is on the paper stable pd
    #this is have dim nu
    error_q_next =(q[7:] + (qdot[6:]*dt) )-target_pos
    # add 6 elements to math the dim of the mass and corolis
    error_pos = np.concatenate([np.zeros(6), error_q_next])
    #remember the size of the kp and kd is 28, and we want 34 to match
    #this is for the final humanoid model dim
    #the dofs nv size
    k_p = np.zeros(qdot.shape[0])
    k_d = np.zeros(qdot.shape[0])
    k_p[6:] = cfg.jkp
    k_d[6:] = cfg.jkd
    #save the angular error, that is the velocity itself
    angular_error = qdot
    #initialize the variables for getting the acceleration equation
    C,M,KP,KD=init_corolis_mass_external_diagonal(data,model,k_p,k_d)
    #get the mass inertia matrix with the added kd dy
    new_mass = calculate_new_mass(M,KD,dt)
    
    #calculate the predicted acceleration   
    qdot_dot = compute_acceleration(error_pos,angular_error,C,new_mass,KP,KD)
    # add the predicted error for the qdot, and then add than on the principal equation
    #so this is like the  next angular error on the stable pd paper
    
    #angular_error = angular_error + (qdot_dot*dt)
    
    #now get the torque avoiding the root
    angular_error += qdot_dot * dt
    tau = -cfg.jkp * error_pos[6:] -cfg.jkd * angular_error[6:]
    
    return tau
