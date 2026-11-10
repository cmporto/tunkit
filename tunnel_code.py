import numpy as np
from dataclasses import replace
# from matplotlib import pyplot as plt
# import math
# import os
# import sys

# from scipy.interpolate import InterpolatedUnivariateSpline as interpolate


# from scipy.interpolate import splrep, PPoly


# import scipy.special as special

# from scipy.optimize import OptimizeWarning
# from sklearn.metrics import r2_score







# importing the subclasses
from tunneling_calc.io import gaussian
from tunneling_calc.io import orca
from tunneling_calc.io import cluster
# from tunneling_calc.core import constants as const
# from tunneling_calc.core import elements as elem
# from tunneling_calc.core import geometry as geom
# from tunneling_calc.path import projections
from tunneling_calc.path import reaction_path
from tunneling_calc.path import build
from tunneling_calc.neb import neb_io
from tunneling_calc.solvers import wkb
from tunneling_calc.solvers import analytic
from tunneling_calc.neb import runner
# from tunneling_calc.io.gaussian import GaussianIOMixin
# from tunneling_calc.io.cluster import ClusterIOMixin
# from tunneling_calc.io.orca import OrcaIOMixin 
# from tunneling_calc.io.ee_inputs import EEInputsIOMixin
# from tunneling_calc.path.build import PathBuildingMixin
# from tunneling_calc.path.projections import ProjectionMixin
# from tunneling_calc.path.diagnostics import DiagIOMixin
# from tunneling_calc.neb.band import BandNebMixin
# from tunneling_calc.neb.neb_io import IoNebMixin
# from tunneling_calc.neb.optimizers import OptimizersNebMixin
# from tunneling_calc.neb.forces import ForcesNebMixin
# from tunneling_calc.solvers.analytic import AnalyticSolverMixin
# from tunneling_calc.solvers.wkb import WKBSolverMixin
# from tunneling_calc.solvers.general import GenSolverMixin




class WKB_Tunneling_Analysis:
# (
    # ClusterIOMixin,
    # PathBuildingMixin,
    # ProjectionMixin,
    # OrcaIOMixin,
    # EEInputsIOMixin,
    # BandNebMixin,
    # IoNebMixin,
    # OptimizersNebMixin,
    # ForcesNebMixin,
    # WKBSolverMixin,
    # AnalyticSolverMixin,
    # GenSolverMixin
# ):
    '''Class to coordinate the preparation and calculation of WKB tunneling analyses.
    '''
    
    def __init__(self, 
                 mep_file_name = '',
                 xyz_file_name = '',
                 reactant_file_name = '',
                 mep_method = 'irc'):

        # Define the initial path
        self.path = reaction_path.ReactionPath()

        # Defining the class variables
        self.mep_file_name      = mep_file_name
        self.xyz_file_name      = xyz_file_name
        self.reactant_file_name = reactant_file_name
        self.mep_method         = mep_method # 'irc' or 'neb'
        self.header             = ''
        self.path_to_folder     = ''
        self.irc_step           = 0
        self.turning_point      = 0
        self.relevant_vib       = 0
        self.proj_made          = False
        self.kinetic_rate       = False
 

        # # Define the arrays to store the parameters
        # self.raw_coord             = []
        # self.raw_coord_ang         = []
        # self.centered_coord        = []
        # self.aligned_coord         = []
        # self.center_of_mass        = []
        # self.energies              = []
        # self.proj_energies         = []
        # self.mass_scaled_coord     = []
        # self.displacement_per_step = []
        # self.reaction_path         = []
        # self.proj_reaction_path    = []
        # self.V_minus_E             = np.array([])
        # self.masses                = []
        # self.atomic_numbers        = []
        # self.rotational_matrices   = []
        # self.transl_vectors        = []
        # self.alignement_rmsd       = []
        # self.hessians              = []
        # self.mw_hessians           = []
        # self.gradients             = []
        # self.mw_gradients          = []
        # self.gradient_rms          = []
        # self.normal_modes          = []
        # self.path_frequencies      = []
        # self.m_vec                 = []
        # self.h_vec                 = []
        # self.b_vec                 = []
        # self.L_vec                 = []
        # self.aps_energy_levels     = [] 
        # self.dwp_sym_eigenvectors  = []
        # self.dwp_sym_degrees       = []
        # self.projected_corrections = []
        # self.proj_coordinates      = []
        # self.tangents              = []
        # self.projected_frequencies = []
        # self.cartesian_disp_non_mw = []






    def extract_irc_gaussian(self,
                             irc_file_name='',
                             **kwargs):
        if not irc_file_name:
            irc_file_name = self.mep_file_name

        self.path = gaussian.extract_irc_gaussian(self.path, irc_file_name, **kwargs)

        return self.path








    def extract_projection_energies(self, *args, **kwargs):
        self.path = gaussian.extract_projection_energies(self.path, *args, **kwargs)
        return self.path




    
 


    def build_reaction_path(self, *args, **kwargs):
        self.path = build.build_reaction_path(self.path, *args, **kwargs)
        return self.path








    def clip_reaction_path(self, clip_start=0, clip_end=0):
        self.path = self.path.clipped(clip_start, clip_end)
        return self.path








    def reverse_path(self):
        self.path = self.path.reversed()
        return self.path








    def extract_coord_energy_xyz(self,
                             xyz_file_name='',
                             **kwargs):
        if not xyz_file_name:
            xyz_file_name = self.mep_file_name

        self.path = orca.extract_coord_energy_xyz(self.path, xyz_file_name, **kwargs)

        return self.path








    def extract_projection_energies_neb(self,
                             path_to_folder='',
                             **kwargs):
        if not path_to_folder and not self.mep_file_name:
            print('The path to the projections folder must be entered!')
        
        if not path_to_folder:
            path_to_folder = self.mep_file_name

        self.path = orca.extract_projection_energies_neb(self.path, path_to_folder=path_to_folder, **kwargs)

        return self.path








    def load_neb_path(self, source, **kwargs):

        self.path = neb_io.load_neb_path(self.path, source, **kwargs)

        return self.path








    def load_cluster_results(self, invert_curve=False, **kwargs):
        
        self.path = cluster.load_cluster_results(self.path, **kwargs)

        if invert_curve:
                    self.path = self.path.reversed()
        
        return self.path








    def calc_tunneling_wkb(self, **kwargs):

        self.result =  wkb.calc_tunneling_wkb(self.path, **kwargs)

        return self.result








    def calc_tunneling_wkb_sym(self, **kwargs):

        self.result =  wkb.calc_tunneling_wkb_sym(self.path, **kwargs)

        return self.result








    def calc_tunneling_wkb_asym(self, **kwargs):

        self.result =  wkb.calc_tunneling_wkb_asym(self.path, **kwargs)

        return self.result








    def calc_symmetric_energy_levels(self, **kwargs):

        self.result =  analytic.calc_symmetric_energy_levels(self.path, **kwargs)

        return self.result







    def calc_asymmetric_energy_levels(self, **kwargs):

        self.result =  analytic.calc_asymmetric_energy_levels(self.path, **kwargs)

        return self.result








    def calculate_matrix_tunneling_rate(self, **kwargs):

        self.mtx_result =  analytic.calculate_matrix_tunneling_rate(self.result, **kwargs)

        return self.mtx_result








    def generate_projection_payload(self, **kwargs):

        cluster.generate_projection_payload(self.path,**kwargs)








    def run_neb(self, **kwargs):

        self.neb_result = runner.run_neb(self.path, **kwargs)
        self.path = self.neb_result.path

        return self.neb_result