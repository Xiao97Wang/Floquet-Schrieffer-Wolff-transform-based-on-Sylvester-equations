from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPOEnvironment
import tenpy.linalg.np_conserved as npc


from tenpy.networks.site import SpinHalfSite, SpinHalfFermionSite, SpinSite
from tenpy.models.model import CouplingModel, MPOModel, NearestNeighborModel, CouplingMPOModel, MultiCouplingModel
from tenpy.models.lattice import Site, Chain
from tenpy.tools.params import get_parameter, unused_parameters



import numpy as np
from tenpy.networks.site import SpinHalfFermionSite


import time

# The `MultiCouplingModel` class is deprecated and has been merged into the `CouplingModel`. No need to subclass the `MultiCouplingModel` anymore!

class FermiHubbardModel(CouplingMPOModel, CouplingModel):

    def __init__(self, model_params):
        CouplingMPOModel.__init__(self, model_params)

    def init_sites(self, model_params):
        cons_N = model_params.get('cons_N', None) 
        cons_Sz = model_params.get('cons_Sz', 'Sz') 
        site = SpinHalfFermionSite(cons_N=None, cons_Sz='Sz')
        return site


    def init_terms(self, model_params):
        t = model_params.get('t', 1.) 
        U = model_params.get('U', 0.)
        w = model_params.get('w', 10.)
        chem_pot = model_params.get('chem_pot', 0.)
        mag_field = model_params.get('mag_field', 0.)
        g = model_params.get('g', 0.)        


        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-chem_pot, u, 'Ntot')         
            self.add_onsite(U, u, 'NuNd')
            self.add_onsite(mag_field, u, 'Sz')
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            #add renormalised hopping
            t_re = t * ( 1 - g**2/w**2 )
            #no renormalisation in the undriven case
            t_re = t  
            self.add_coupling(-t_re, u1, 'Cdu', u2, 'Cu', dx)
            self.add_coupling(-t_re, u2, 'Cdu', u1, 'Cu', -dx)  # h.c.
            self.add_coupling(-t_re, u1, 'Cdd', u2, 'Cd', dx)
            self.add_coupling(-t_re, u2, 'Cdd', u1, 'Cd', -dx)  # h.c.
            #add nearest neighbour repulsion V
            #self.add_coupling(V, u1, 'Ntot', u2, 'Ntot', dx)
            
            #add driving induced interaction, which is the correlated hopping
            t_co = t * (g**2/w**2) * U * ( 1/(U-w) + 1/(U+w) ) 
            # in the undriven case, we don't have this correlated hopping term
            t_co = 0  
            
            # s=up 
            self.add_coupling(- t_co/2. , u1, 'Cdu Nd', u2, 'Cu', dx)
            self.add_coupling(- t_co/2. , u1, 'Cdu', u2, 'Cu Nd', dx)
            self.add_coupling(  t_co    , u1, 'Cdu Nd', u2, 'Cu Nd', dx)
            #     h.c.  which is equivalent to switching j and j+1
            self.add_coupling(- t_co/2. , u2, 'Cdu Nd', u1, 'Cu', -dx)
            self.add_coupling(- t_co/2. , u2, 'Cdu', u1, 'Cu Nd', -dx)
            self.add_coupling(  t_co    , u2, 'Cdu Nd', u1, 'Cu Nd', -dx)
            
            # s=down 
            self.add_coupling(- t_co/2. , u1, 'Cdd Nu', u2, 'Cd', dx)
            self.add_coupling(- t_co/2. , u1, 'Cdd', u2, 'Cd Nu', dx)
            self.add_coupling(  t_co    , u1, 'Cdd Nu', u2, 'Cd Nu', dx)
            #     h.c.  which is equivalent to switching j and j+1
            self.add_coupling(- t_co/2. , u2, 'Cdd Nu', u1, 'Cd', -dx)
            self.add_coupling(- t_co/2. , u2, 'Cdd', u1, 'Cd Nu', -dx)
            self.add_coupling(  t_co    , u2, 'Cdd Nu', u1, 'Cd Nu', -dx)
            
        #for dx1 in range(2,L):                      
        #    self.add_multi_coupling(-2.*glob, [('Cdu', [1], 0), ('Cu', [0], 0), ('Cdu', [1+dx1], 0), ('Cu', [dx1], 0)])  





class FermiHubbardModelEvo(CouplingMPOModel, CouplingModel):

    def __init__(self, model_params):
        CouplingMPOModel.__init__(self, model_params)

    def init_sites(self, model_params):
        cons_N = model_params.get('cons_N', None) 
        cons_Sz = model_params.get('cons_Sz', 'Sz') 
        site = SpinHalfFermionSite(cons_N=None, cons_Sz='Sz')
        return site


    def init_terms(self, model_params):
        t = model_params.get('t', 1.) 
        U = model_params.get('U', 0.)
        w = model_params.get('w', 10.)
        chem_pot = model_params.get('chem_pot', 0.)
        mag_field = model_params.get('mag_field', 0.)
        g = model_params.get('g', 0.)          
        time = model_params.get('time', 0.)
        
        drive_field = np.zeros(L)   # Here in the FSWT stroboscopic TEBD, we dont have this drive field
        
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-chem_pot, u, 'Ntot')         
            self.add_onsite(U, u, 'NuNd')
            self.add_onsite(mag_field, u, 'Sz')
            self.add_onsite(drive_field, u, 'Ntot')   # add time-dependent driving term
            
            
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            #add renormalised hopping
            t_re = t * ( 1 - g**2/w**2 )
            #Here in the FSWT stroboscopic TEBD, we have the bandwidth renormalisation
            self.add_coupling(-t_re, u1, 'Cdu', u2, 'Cu', dx)
            self.add_coupling(-t_re, u2, 'Cdu', u1, 'Cu', -dx)  # h.c.
            self.add_coupling(-t_re, u1, 'Cdd', u2, 'Cd', dx)
            self.add_coupling(-t_re, u2, 'Cdd', u1, 'Cd', -dx)  # h.c.
            
            # Here in the FSWT stroboscopic TEBD, we also have the correlated hopping term
            t_co = t * (g**2/w**2) * U * ( 1/(U-w) + 1/(U+w) ) 
            
            # s=up 
            self.add_coupling(- t_co/2. , u1, 'Cdu Nd', u2, 'Cu', dx)
            self.add_coupling(- t_co/2. , u1, 'Cdu', u2, 'Cu Nd', dx)
            self.add_coupling(  t_co    , u1, 'Cdu Nd', u2, 'Cu Nd', dx)
            #     h.c.  which is equivalent to switching j and j+1
            self.add_coupling(- t_co/2. , u2, 'Cdu Nd', u1, 'Cu', -dx)
            self.add_coupling(- t_co/2. , u2, 'Cdu', u1, 'Cu Nd', -dx)
            self.add_coupling(  t_co    , u2, 'Cdu Nd', u1, 'Cu Nd', -dx)
            
            # s=down 
            self.add_coupling(- t_co/2. , u1, 'Cdd Nu', u2, 'Cd', dx)
            self.add_coupling(- t_co/2. , u1, 'Cdd', u2, 'Cd Nu', dx)
            self.add_coupling(  t_co    , u1, 'Cdd Nu', u2, 'Cd Nu', dx)
            #     h.c.  which is equivalent to switching j and j+1
            self.add_coupling(- t_co/2. , u2, 'Cdd Nu', u1, 'Cd', -dx)
            self.add_coupling(- t_co/2. , u2, 'Cdd', u1, 'Cd Nu', -dx)
            self.add_coupling(  t_co    , u2, 'Cdd Nu', u1, 'Cd Nu', -dx)

        # Here in the FSWT stroboscopic TEBD, we have doublon-holon exchange as O(t^2) two-site term
        # t_co2 = t^2 g^2/w^3  (beta' - gamma')
        t_co2 = t**2 * (g**2/w**3) * U * ( 1/(U-w) - 1/(U+w) ) 
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling( 4 * t_co2 , u1, 'Cdu Cdd', u2, 'Cu Cd', dx)
            #     h.c.  which is equivalent to switching j and j+1
            self.add_coupling( 4 * t_co2 , u2, 'Cdd Cdu', u1, 'Cd Cu', -dx)
        
        # add 3-site terms which is order O(t^2)
        delta = U * ( 1/(U-w) + 1/(U+w) )
        
        # NNN hopping
        # s = up        # automatically add the H.c. part
        self.add_multi_coupling( 2*t_co2        , [ ('Cdu', [-1], 0), ('Cu', [1], 0), ('Nd', [0], 0) ] , plus_hc = True)
        self.add_multi_coupling( (delta-1)*t_co2, [ ('Cdu', [-1], 0), ('Cu', [1], 0), ('Nd', [-1], 0) ] , plus_hc = True)
        self.add_multi_coupling( (delta-1)*t_co2, [ ('Cdu', [-1], 0), ('Cu', [1], 0), ('Nd', [1], 0) ] , plus_hc = True)
        self.add_multi_coupling( -2*delta*t_co2 , [ ('Cdu', [-1], 0), ('Cu', [1], 0), ('Nd', [-1], 0), ('Nd', [1], 0) ] , plus_hc = True)
        self.add_multi_coupling( -2*delta*t_co2 , [ ('Cdu', [-1], 0), ('Cu', [1], 0), ('Nd', [0], 0), ('Nd', [-1], 0) ] , plus_hc = True)
        self.add_multi_coupling( -2*delta*t_co2 , [ ('Cdu', [-1], 0), ('Cu', [1], 0), ('Nd', [0], 0), ('Nd', [1], 0) ] , plus_hc = True)
        self.add_multi_coupling( 4*delta*t_co2  , [ ('Cdu', [-1], 0), ('Cu', [1], 0), ('Nd', [0], 0), ('Nd', [-1], 0), ('Nd', [1], 0) ] , plus_hc = True)
        # s = down      # automatically add the H.c. part
        self.add_multi_coupling( 2*t_co2        , [ ('Cdd', [-1], 0), ('Cd', [1], 0), ('Nu', [0], 0) ] , plus_hc = True)
        self.add_multi_coupling( (delta-1)*t_co2, [ ('Cdd', [-1], 0), ('Cd', [1], 0), ('Nu', [-1], 0) ] , plus_hc = True)
        self.add_multi_coupling( (delta-1)*t_co2, [ ('Cdd', [-1], 0), ('Cd', [1], 0), ('Nu', [1], 0) ] , plus_hc = True)
        self.add_multi_coupling( -2*delta*t_co2 , [ ('Cdd', [-1], 0), ('Cd', [1], 0), ('Nu', [-1], 0), ('Nu', [1], 0) ] , plus_hc = True)
        self.add_multi_coupling( -2*delta*t_co2 , [ ('Cdd', [-1], 0), ('Cd', [1], 0), ('Nu', [0], 0), ('Nu', [-1], 0) ] , plus_hc = True)
        self.add_multi_coupling( -2*delta*t_co2 , [ ('Cdd', [-1], 0), ('Cd', [1], 0), ('Nu', [0], 0), ('Nu', [1], 0) ] , plus_hc = True)
        self.add_multi_coupling( 4*delta*t_co2  , [ ('Cdd', [-1], 0), ('Cd', [1], 0), ('Nu', [0], 0), ('Nu', [-1], 0), ('Nu', [1], 0) ] , plus_hc = True)
        
        # two-electron hopping
        # s = up        # automatically add the H.c. part
        self.add_multi_coupling( 2*t_co2        , [ ('Cdu', [0], 0), ('Cdd', [-1], 0), ('Cd', [0], 0), ('Cu', [1], 0) ] , plus_hc = True)
        self.add_multi_coupling( -2*delta*t_co2 , [ ('Cdu', [0], 0), ('Cdd', [-1], 0), ('Cd', [0], 0), ('Cu', [1], 0), ('Nu', [-1], 0) ] , plus_hc = True)
        self.add_multi_coupling( -2*delta*t_co2 , [ ('Cdu', [0], 0), ('Cdd', [-1], 0), ('Cd', [0], 0), ('Cu', [1], 0), ('Nd', [1], 0) ] , plus_hc = True)
        self.add_multi_coupling( 4*delta*t_co2  , [ ('Cdu', [0], 0), ('Cdd', [-1], 0), ('Cd', [0], 0), ('Cu', [1], 0), ('Nu', [-1], 0), ('Nd', [1], 0) ] , plus_hc = True)
        # s = down      # automatically add the H.c. part
        self.add_multi_coupling( 2*t_co2        , [ ('Cdd', [0], 0), ('Cdu', [-1], 0), ('Cu', [0], 0), ('Cd', [1], 0) ] , plus_hc = True)
        self.add_multi_coupling( -2*delta*t_co2 , [ ('Cdd', [0], 0), ('Cdu', [-1], 0), ('Cu', [0], 0), ('Cd', [1], 0), ('Nd', [-1], 0) ] , plus_hc = True)
        self.add_multi_coupling( -2*delta*t_co2 , [ ('Cdd', [0], 0), ('Cdu', [-1], 0), ('Cu', [0], 0), ('Cd', [1], 0), ('Nu', [1], 0) ] , plus_hc = True)
        self.add_multi_coupling( 4*delta*t_co2  , [ ('Cdd', [0], 0), ('Cdu', [-1], 0), ('Cu', [0], 0), ('Cd', [1], 0), ('Nd', [-1], 0), ('Nu', [1], 0) ] , plus_hc = True)
        
        # the tuple ('Op' , [dx] , u ) appears repeatedly above, which represents an operator 
        # in our simple lattice, the unit_cell index u = 0
        # but dx =[1] means site j+1 ,  dx =[-1] means site j-1 ,  dx =[0] means site j
        







class FermiHubbardChain(FermiHubbardModel, NearestNeighborModel):    

    def __init__(self, model_params):
        model_params.setdefault('lattice', "Chain")
        CouplingMPOModel.__init__(self, model_params)

class FermiHubbardChainEvo(FermiHubbardModelEvo, CouplingMPOModel):    

    def __init__(self, model_params):
        model_params.setdefault('lattice', "Chain")
        CouplingMPOModel.__init__(self, model_params)

def Hubbard_Hamiltonian(U, t, chem_pot, mag_field, w, g, cons_N, cons_Sz, L):
    if(cons_N == False and cons_Sz == False):
        model_params = dict(L=L, U=U, t=t, chem_pot=chem_pot, mag_field = mag_field, w = w, g = g, bc_MPS='finite')
    elif(cons_N != False and cons_Sz == False):
        model_params = dict(L=L, U=U, t=t, chem_pot=chem_pot, mag_field = mag_field, w = w, g = g, cons_N=cons_N, bc_MPS='finite')
    elif(cons_N == False and cons_Sz !=False):
        model_params = dict(L=L, U=U, t=t, chem_pot=chem_pot, mag_field = mag_field, w = w, g = g, cons_Sz=cons_Sz, bc_MPS='finite')
    else:
        model_params = dict(L=L, U=U, t=t, chem_pot=chem_pot, mag_field = mag_field, w = w, g = g, cons_N = cons_N, cons_Sz=cons_Sz, bc_MPS='finite')

    return FermiHubbardChain(model_params)

def Hubbard_Hamiltonian_Evo(U, t, chem_pot, mag_field, w, g, cons_N, cons_Sz, L, time):
    if(cons_N == False and cons_Sz == False):
        model_params = dict(L=L, U=U, t=t, chem_pot=chem_pot, mag_field = mag_field, w = w, g = g, time = time, bc_MPS='finite')
    elif(cons_N != False and cons_Sz == False):
        model_params = dict(L=L, U=U, t=t, chem_pot=chem_pot, mag_field = mag_field, w = w, g = g, time = time, cons_N=cons_N, bc_MPS='finite')
    elif(cons_N == False and cons_Sz !=False):
        model_params = dict(L=L, U=U, t=t, chem_pot=chem_pot, mag_field = mag_field, w = w, g = g, time = time, cons_Sz=cons_Sz, bc_MPS='finite')
    else:
        model_params = dict(L=L, U=U, t=t, chem_pot=chem_pot, mag_field = mag_field, w = w, g = g, time = time, cons_N = cons_N, cons_Sz=cons_Sz, bc_MPS='finite')

    return FermiHubbardChainEvo(model_params)

from tenpy.algorithms.dmrg import run as run_DMRG
from tenpy.algorithms import tdvp

def perform_DMRG(psi, Ham, chi_max):
    dmrg_params = {'trunc_params': {'chi_max': chi_max, 'svd_min': 1.e-10, 'trunc_cut': None}, 'mixer': True}
    t0 = time.time()
    info = run_DMRG(psi, Ham, dmrg_params)
    print("DMRG finished after", time.time() - t0, "s")

    return psi





U = 28
L=10
w=16  
g= w / 4.
t=1.0   
mu= -1.0      # the chemical potential is used for setting the total electron number, which then remains conserved under drive
chi_dmrg = 300          # MPS bond dimension for DMRG and also for the TEBD
mag_field = np.zeros(L)

TDVP_Simulation_Time = 60

import argparse


mu_tot = U/2 + mu   # there is a U/2 bias on mu
chem_field = np.zeros(L)+ mu_tot     

site = SpinHalfFermionSite(cons_N='N', cons_Sz='Sz')
Ham1 =  Hubbard_Hamiltonian(U, t, chem_field, mag_field, w, g, False, True, L)    # grand-canonical
MPO_Ham1 = Ham1.calc_H_MPO()

stateString = ["up", "down"] * int(L//2)
if L % 2:      # if the site-number L is odd
    stateString = stateString + ["up"]

psi_1 = MPS.from_product_state(Ham1.lat.mps_sites(), stateString , "finite"  )  # Half-filled   # the "finite" argument is necessary
#outcome_1=perform_DMRG(psi_1, Ham1, chi_dmrg)
outcome_1= MPS.from_product_state(Ham1.lat.mps_sites(),  [ "full" , "empty"] * (L//2) , "finite"  )   

import copy
psi_ini = copy.deepcopy(outcome_1)

data=open("FSWT_finiteTDVP_EvoPlot_L"+str(L)+"_chi"+str(chi_dmrg)+"_mu"+str(mu)+"_U"+str(U)+"_omega"+str(w)+"_g"+str(g)+".txt",'a+')    #  change to your output directory

print("undriven doublon density nD=", np.average(outcome_1.expectation_value("NuNd")),  ", occupancy n="  , np.average(outcome_1.expectation_value("Ntot")) , file=data)

#apply two-site finite TDVP, because TEBD is not working for 3-site coupling, which appears in O(t^2) Floquet Hamiltonian

EVOdt = 0.001   # should be much smaller than 1/w

tdvp_params = {
    'N_steps': 1,
    'dt': EVOdt,
    'trunc_params': {'chi_max': chi_dmrg, 'svd_min': 1.e-10}
}

Evolved_time =0.

while Evolved_time < TDVP_Simulation_Time:
    HamEvo =  Hubbard_Hamiltonian_Evo(U, t, chem_field, mag_field, w, g, False, True, L, Evolved_time) 
    MPO_HamEvo = HamEvo.calc_H_MPO()
    eng = tdvp.TwoSiteTDVPEngine(outcome_1, HamEvo, tdvp_params)   # use the original time-dependent Hamiltonian
    eng.run()  
    print("FSWT stroboscopic TDVP nD=", np.average(outcome_1.expectation_value("NuNd")), ", occupancy n=", np.average(outcome_1.expectation_value("Ntot")),", at time = ", Evolved_time)
    print(Evolved_time,  np.abs( outcome_1.overlap(psi_ini) )**2 , file=data)   # Return Rate
    Evolved_time = Evolved_time + EVOdt


data.close()

# to run the code, type the following
# python Xiao_FSWT_TimeEvo_finite.py