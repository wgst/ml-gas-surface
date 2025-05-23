# from schnetpack.md import neighborlist_md, calculators
from ase import io
# from ase.build import bulk
import time
from mace.calculators import MACECalculator
# import schnetpack as spk
# import random
# from ase.md.npt import NPT
from ase.md.langevin import Langevin
from ase.units import s
from ase import units
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution,Stationary
from time import perf_counter
import matplotlib.pyplot as plt
# from pathlib import Path
from ase.io import read,Trajectory

# Print statements
def print_dyn():
    imd = dyn.get_number_of_steps()
    etot  = atoms.get_total_energy()
    ekin = atoms.get_kinetic_energy()
    temp_K = atoms.get_temperature()
    stress = atoms.get_stress(include_ideal_gas=True)/units.GPa
    stress_ave = (stress[0]+stress[1]+stress[2])/3.0
    elapsed_time = perf_counter() - start_time
    print(f"  {imd: >3}   {etot:.3f}    {ekin:.3f}    {temp_K:.2f}    {stress_ave:.2f}  {stress[0]:.2f}  {stress[1]:.2f}  {stress[2]:.2f}  {stress[3]:.2f}  {stress[4]:.2f}  {stress[5]:.2f}    {elapsed_time:.3f}")


###INPUT PARAMS
surface = 'cu111'

model_path="/home/m/msrmnk2/1_calculations/1_ml/2_training/2_h2cu/31_MACE/10_mace_0_2_0/5_4_ad_float32_final2_FINAL_opt_co5_no_outliers/1/MACE_model_swa.model"

calculator = MACECalculator(model_path=model_path, device="cpu", default_dtype="float32")


# Set up a crystal
atoms = read(f'../../../../../0_input_structures/{surface}_3_66_ase.in')
atoms.calc = calculator
io.write('input.xyz', atoms)
print("atoms: ",atoms)

# input parameters
time_step    = 1.0*units.fs    # fsec
temperature = 300    # Kelvin
num_md_steps = 140000*6 # 20000*6
num_interval = 10

output_folder = 'output'
traj_filename = f'{output_folder}/nvt_{temperature}K.traj'
log_filename = f'{output_folder}/nvt_{temperature}K.log'

MaxwellBoltzmannDistribution(atoms, temperature_K=temperature,force_temp=True)
Stationary(atoms) # Sets the center-of-mass momentum to zero.

dyn = Langevin(
    atoms,
    timestep=time_step,
    temperature_K=temperature,  # temperature in K
    friction=0.2 / units.fs,
    logfile = log_filename,
    trajectory = traj_filename,
    loginterval=num_interval
)

print_interval = num_interval # if calc_type == "EMT" else num_interval
dyn.attach(print_dyn, interval=print_interval)
dyn.attach(MDLogger(dyn, atoms, log_filename, header=True, stress=True, peratom=True, mode="a"), interval=num_interval)

# Now run the dynamics
start_time = perf_counter()
print(f"    imd     Etot(eV)    T(K)    stress(mean,xx,yy,zz,yz,xz,xy)(GPa)  elapsed_time(sec)")
dyn.run(num_md_steps)
