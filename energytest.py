import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "1" 

import psi4
import numpy as np

import cartesian as cart

psi4.core.set_num_threads(8)
d={}

def energy(ch,hh):
    
    molecule='''C 0.000000 0.000000 0.000000
                 O 1.210000 0.000000 0.000000
                 H -0.555000 0.961288 0.000000
                 H -0.558924 -0.959011 0.000000
            '''
    molecule=psi4.geometry(molecule)


    psi4.set_options({'geom_maxiter': 1500, 'g_convergence': 'GAU_LOOSE','guess': 'sad'})
    
 
    #energy_pre,wfn_hf= psi4.optimize('hf/3-21g', molecule=molecule,engine='optking', return_wfn=True)

    constraints = [
        {'type': 'bond', 'indices': [0, 2], 'value': ch},  
        {'type': 'bond', 'indices': [2, 3], 'value': hh}   
    ]


    try:
        energy, wfn = psi4.optimize('hf/6-31g', molecule=molecule, return_wfn=True)
        
    except psi4.OptimizationConvergenceError as ex:
        psi4.set_options({'guess':'read'})
        last_geom = ex.wfn.molecule().geometry()
        coords_np = np.asarray(last_geom)
        mol = psi4.geometry(create_geometry_string(coords_np))
        energy, wfn = psi4.optimize('hf/6-31G', molecule=mol, return_wfn=True)



    psi4.set_options({
        'geom_maxiter': 50,#1000
        'g_convergence': 'GAU_LOOSE',#gauloose
        'scf_type': 'df',
        'diis':True,
        'damping_percentage': 10.0,
        'level_shift':0.8,
        'dft_spherical_points':302,
        'dft_radial_points':75,
        'guess': 'sad',
        'scf__maxiter':10000,#10000
        'optking__opt_coordinates': 'BOTH',
        'optking__max_force_g_convergence': 2e-4,
        'optking__rms_force_g_convergence': 5e-4,
        'optking__max_disp_g_convergence': 1e-2,
        'optking__rms_disp_g_convergence': 2e-3,
        'optking__max_energy_g_convergence': 1e-5
        
    })

    try:
        energy, wfn = psi4.optimize('m06-2x/6-31++G', molecule=molecule,engine='optking',optimizer_keywords={'constraints': constraints}, return_wfn=True)
        
    except psi4.OptimizationConvergenceError as ex:
        last_geom = ex.wfn.molecule().geometry()
        coords_np = np.asarray(last_geom)
        mol = psi4.geometry(create_geometry_string(coords_np))
        energy, wfn = psi4.optimize('m06-2x/6-31++G', molecule=molecule,engine='optking',optimizer_keywords={'constraints': constraints}, return_wfn=True)
    
    return(energy)

def pes():
    for i in np.linspace(0.7,1.4,17):
        for j in np.linspace(0.6,2.2,17):
            en=energy(float(i),float(j))
            d[(i,j)]=en
            print(d)
    return d

print(pes())




