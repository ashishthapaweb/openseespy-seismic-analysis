# Function to build nodes
import openseespy.opensees as ops
from settings import *
import numpy as np
from response_spectra import soil_type_acc_coeff

colSecId = 1
beamSecId = 2
transfType = 'Linear'
colTransTag = 1
beamTransTag = 2
colIntTag = 1
beamIntTag = 2


class Model:
    def __init__(self, numBayX, numBayY, bayWidthX, bayWidthY, numFloor, storyHeight):
        self.periods = []
        ops.wipe()
        ops.model('basic', '-ndm', 3, '-ndf', 6)
        self.eigenValues = None
        self.floor_mass = {}
        self.overall_mass = 0 * kg
        self.beam_massX = None
        self.col_massX = None
        self.numBayX = numBayX
        self.numBayY = numBayY
        self.bayWidthX = bayWidthX
        self.bayWidthY = bayWidthY
        self.numFloor = numFloor
        self.storyHeight = storyHeight
        self.nodes = {}  # dictionary containing floor and nodes tags for nodes in that floor
        self.cm_node_ids = []
        self.columns = {}  # dictionary containing floor numbers and number of columns in that floor
        self.beams = {}
        self.eleTag = int

    def make_nodes(self):
        nodeTag = 1
        zLoc = 0.
        for k in range(0, self.numFloor + 1):
            xLoc = 0.
            self.nodes[k] = []
            for i in range(0, self.numBayX + 1):
                yLoc = 0.
                for j in range(0, self.numBayY + 1):
                    coord = [xLoc, yLoc, zLoc]
                    ops.node(nodeTag, *coord)

                    if k == 0:
                        fully_fixed = [1, 1, 1, 1, 1, 1]
                        ops.fix(nodeTag, *fully_fixed)

                    yLoc += self.bayWidthY
                    self.nodes[k].append(nodeTag)
                    nodeTag = nodeTag + 1
                xLoc += self.bayWidthX
            zLoc += self.storyHeight
        return self.nodes

    # Column
    # def build_columns(self, d1, d2):
    #     A = d1 * d2
    #     Ixx = d1 * (d2 ** 3) / 12
    #     Iyy = (d1 ** 3) * d2 / 12
    #     J = Ixx + Iyy
    #     Yc = 25 * kN / m ** 3  # Unit weight of concrete
    #     self.col_massX = A * Yc  # Mass per unit length of column
    #     ops.section("Elastic", colSecId, E, A, Ixx, Iyy, G, J)
    #     ops.beamIntegration('Legendre', 1, 1, 2)
    #     ops.geomTransf(transfType, colTransTag, 1, 0, 0)
    #
    #     self.eleTag = 1
    #     nodeTag1 = 1
    #     for k in range(0, self.numFloor):
    #         self.columns[k + 1] = []
    #         for i in range(0, self.numBayX + 1):
    #             for j in range(0, self.numBayY + 1):
    #                 nodeTag2 = nodeTag1 + (self.numBayX + 1) * (self.numBayY + 1)
    #                 ops.element('forceBeamColumn', self.eleTag, *[nodeTag1, nodeTag2], colTransTag, colIntTag)
    #                 self.columns[k + 1].append(self.eleTag)  # Add column to the record
    #                 self.eleTag += 1
    #                 nodeTag1 += 1
    #
    #     return self.columns

    def build_columns(self, d1, d2):
        A = d1 * d2
        Avy = 5 / 6 * A  # Effective shear area in Y direction
        Avz = 5 / 6 * A  # Effective shear area in Z direction
        Izz = d1 * (d2 ** 3) / 12
        Iyy = (d1 ** 3) * d2 / 12
        J = Izz + Iyy
        Yc = 25 * kN / m ** 3  # Unit weight of concrete
        self.col_massX = A * Yc  # Mass per unit length of column
        # ops.section("Elastic", colSecId, E, A, Ixx, Iyy, G, J)
        # ops.beamIntegration('Legendre', 1, 1, 2)
        ops.geomTransf(transfType, colTransTag, 1, 0, 0)

        self.eleTag = 1
        nodeTag1 = 1
        for k in range(0, self.numFloor):
            self.columns[k + 1] = []
            for i in range(0, self.numBayX + 1):
                for j in range(0, self.numBayY + 1):
                    nodeTag2 = nodeTag1 + (self.numBayX + 1) * (self.numBayY + 1)
                    ops.element('elasticTimoshenkoBeam', self.eleTag, nodeTag1, nodeTag2, E, G, A, J, Iyy, Izz, Avy,
                                Avz, colTransTag)
                    # ops.element('forceBeamColumn', self.eleTag, *[nodeTag1, nodeTag2], colTransTag, colIntTag)
                    self.columns[k + 1].append(self.eleTag)  # Add column to the record
                    self.eleTag += 1
                    nodeTag1 += 1

        return self.columns

    # def build_beams(self, b, d):
    #     A = b * d
    #     self.beam_massX = A * 25 * kN / m ** 3
    #     Iz = b * (d ** 3) / 12
    #     Iy = d * (b ** 3) / 12
    #     J = Iz + Iy
    #
    #     ops.section('Elastic', beamSecId, E, A, Iz, Iy, G, J)
    #     ops.beamIntegration('Legendre', beamIntTag, beamSecId, 2)
    #
    #     # Beams in One Direction
    #
    #     ops.geomTransf('Linear', 2, 0, 1, 0)  # 0, 1, 0
    #
    #     nodeTag1 = 1 + (self.numBayX + 1) * (self.numBayY + 1)
    #
    #     for j in range(1, self.numFloor + 1):
    #         self.beams[j] = []
    #         for i in range(0, self.numBayX):
    #             for k in range(0, self.numBayY + 1):
    #                 nodeTag2 = nodeTag1 + (self.numBayY + 1)
    #                 ops.element('forceBeamColumn', self.eleTag, nodeTag1, nodeTag2, 2, 2)
    #                 self.beams[j].append(self.eleTag)
    #                 self.eleTag += 1
    #                 nodeTag1 += 1
    #         nodeTag1 += (self.numBayY + 1)
    #
    #     # Beam in other direction
    #
    #     ops.geomTransf('Linear', 3, 1, 0, 0)
    #     nodeTag1 = 1 + (self.numBayX + 1) * (self.numBayY + 1)
    #
    #     for j in range(1, self.numFloor + 1):
    #         for i in range(0, self.numBayX + 1):
    #             for k in range(0, self.numBayY):
    #                 nodeTag2 = nodeTag1 + 1
    #                 ops.element('forceBeamColumn', self.eleTag, nodeTag1, nodeTag2, 3, 2)
    #                 self.beams[j].append(self.eleTag)
    #                 self.eleTag += 1
    #                 nodeTag1 += 1
    #             nodeTag1 += 1
    #
    #     return self.beams

    def build_beams(self, b, d):
        A = b * d
        Avy = 5 / 6 * A  # Effective shear area in Y direction
        Avz = 5 / 6 * A  # Effective shear area in Z direction
        self.beam_massX = A * 25 * kN / m ** 3
        Izz = b * (d ** 3) / 12
        Iyy = d * (b ** 3) / 12
        J = Izz + Iyy

        # ops.section('Elastic', beamSecId, E, A, Iz, Iy, G, J)
        # ops.beamIntegration('Legendre', beamIntTag, beamSecId, 2)

        # Beams in One Direction

        ops.geomTransf('Linear', 2, 0, 1, 0)  # 0, 1, 0

        nodeTag1 = 1 + (self.numBayX + 1) * (self.numBayY + 1)

        for j in range(1, self.numFloor + 1):
            self.beams[j] = []
            for i in range(0, self.numBayX):
                for k in range(0, self.numBayY + 1):
                    nodeTag2 = nodeTag1 + (self.numBayY + 1)
                    # ops.element('forceBeamColumn', self.eleTag, nodeTag1, nodeTag2, 2, 2)
                    ops.element('elasticTimoshenkoBeam', self.eleTag, nodeTag1, nodeTag2, E, G, A, J, Iyy, Izz, Avy,
                                Avz, 2)
                    self.beams[j].append(self.eleTag)
                    self.eleTag += 1
                    nodeTag1 += 1
            nodeTag1 += (self.numBayY + 1)

        # Beam in other direction

        ops.geomTransf('Linear', 3, 1, 0, 0)
        nodeTag1 = 1 + (self.numBayX + 1) * (self.numBayY + 1)

        for j in range(1, self.numFloor + 1):
            for i in range(0, self.numBayX + 1):
                for k in range(0, self.numBayY):
                    nodeTag2 = nodeTag1 + 1
                    # ops.element('forceBeamColumn', self.eleTag, nodeTag1, nodeTag2, 3, 2)
                    ops.element('elasticTimoshenkoBeam', self.eleTag, nodeTag1, nodeTag2, E, G, A, J, Iyy, Izz, Avy,
                                Avz, 3)
                    self.beams[j].append(self.eleTag)
                    self.eleTag += 1
                    nodeTag1 += 1
                nodeTag1 += 1

        return self.beams

    def apply_rigid_diaphragm_constraint(self):
        for floor in range(1, self.numFloor + 1):
            zLevel = floor * self.storyHeight

            cm_coordinates = (self.bayWidthX * self.numBayX / 2, self.bayWidthY * self.numBayY / 2, zLevel)

            cm_node_id = floor + 10000
            self.cm_node_ids.append(cm_node_id)

            ops.node(cm_node_id, *cm_coordinates)

            # Select the master node (typically the first node in the list)
            ops.fix(cm_node_id, 0, 0, 1, 1, 1, 0)

            # Apply the rigid diaphragm constraint for all nodes in the floor
            ops.rigidDiaphragm(3, cm_node_id, *self.nodes[floor])

        return print("Rigid diaphragm constraint is applied successfully.")

    def distribute_the_self_weight(self):
        ops.timeSeries('Linear', 1)
        ops.pattern('Plain', 1, 1)
        total_beam_length = self.bayWidthX * self.numBayX * (self.numBayY + 1) + self.bayWidthY * self.numBayY * (
                self.numBayX + 1)
        total_beam_wt = self.beam_massX * total_beam_length
        total_col_length = (self.numBayX + 1) * (self.numBayY + 1) * self.storyHeight
        total_col_wt = self.col_massX * total_col_length
        slab_depth = 125 * mm
        q = 25 * (kN / m ** 3) * slab_depth + 2 ** (kN / m ** 2)  # Dead load + Live load
        total_slab_wt = (q * self.bayWidthX * self.numBayX * self.bayWidthY * self.numBayY)

        for floor, nodes in self.nodes.items():
            if floor < self.numFloor:
                total_wt = total_beam_wt + total_col_wt + total_slab_wt
                for node in nodes:
                    ops.load(node, 0, 0, - total_wt / len(nodes), 0, 0, 0)
            else:
                total_wt = total_beam_wt + total_col_wt + total_slab_wt

                for node in nodes:
                    ops.load(node, 0, 0, - total_wt / len(nodes), 0, 0, 0)

    def distribute_the_mass(self):
        """
        Correct approach is to:
            - assign the total floor mass to the horizontal DOFs of the primary rigid diaphragm node,
            - assign the rotational mass about the vertical DOF
        """
        self.overall_mass = 0

        total_beam_length = self.bayWidthX * self.numBayX * (self.numBayY + 1) + self.bayWidthY * self.numBayY * (
                self.numBayX + 1)
        total_beam_mass = self.beam_massX * total_beam_length / g
        total_col_length = (self.numBayX + 1) * (self.numBayY + 1) * self.storyHeight
        total_col_mass = self.col_massX * total_col_length / g
        slab_depth = 127 * mm
        q = 25 * (kN / m ** 3) * slab_depth + 2 ** (kN / m ** 2)
        total_slab_mass = (q * self.bayWidthX * self.numBayX * self.bayWidthY * self.numBayY) / g

        for floor, cm_node_id in enumerate(self.cm_node_ids, start=1):

            if floor < self.numFloor:
                total_mass = total_beam_mass + total_col_mass + total_slab_mass
                self.floor_mass[floor] = total_mass
                # Rotational mass about vertical axis
                mz = total_mass * ((self.bayWidthX * self.numBayX) ** 2 + (self.bayWidthY * self.numBayY) ** 2) / 12
                ops.mass(cm_node_id, total_mass, total_mass, 0, 0, 0, mz)
                self.overall_mass += total_mass
            else:
                total_mass = total_beam_mass + total_col_mass / 2 + total_slab_mass
                self.floor_mass[floor] = total_mass
                # Rotational mass about vertical axis
                mz = total_mass * ((self.bayWidthX * self.numBayX) ** 2 + (self.bayWidthY * self.numBayY) ** 2) / 12
                ops.mass(cm_node_id, total_mass, total_mass, 0, 0, 0, mz)
                self.overall_mass += total_mass

        print(
            f"The mass distribution is applied successfully. The total mass of the building is {self.overall_mass: .3f} tons.")

    def eigen_values(self, numOfModes):
        print("\nInitializing modal analysis...")
        self.eigenValues = ops.eigen('-fullGenLapack', numOfModes)
        self.periods = [(2 * np.pi) / np.sqrt(value) if value > 0 else float('inf') for value in self.eigenValues]
        # Print table header
        print(f"\n{'Mode number': <15}{'Period': <15}")
        print("-" * 30)
        # Print each mode number and corresponding period
        for i, period in enumerate(self.periods, start=1):
            print(f"{i: <15}{period: <15.6f}")

    def run_static_analysis(self, Z, I, R):
        print("\n Initializing analysis using Equivalent Static Method...\n")
        # Implementation of Equivalent Static Method
        Tn = soil_type_acc_coeff("ESM")[0]
        Sa = soil_type_acc_coeff("ESM")[1]
        T_1 = self.periods[0]
        Sa_here = np.interp(T_1, Tn, Sa)
        print(f"The value of Sa/g used here is {Sa_here: .2f}.")
        # Total Seismic Weights
        seismic_weight = sum(self.floor_mass.values()) * g
        # Base shear
        Ah = (Z / 2 * I / R * Sa_here)
        Vb = Ah * seismic_weight

        h = [i * self.storyHeight for i in range(1, self.numFloor + 1)]
        w = [value * g for key, value in self.floor_mass.items()]

        # Compute lateral forces

        k = 1.0  # Exponent for distribution (k=1 for T ≤ 0.5 sec; k=2 for T ≥ 2.5 sec)
        F = []
        sum_w_hk = sum(wi * hi ** k for wi, hi in zip(w, h))
        for wi, hi in zip(w, h):
            Fi = Vb * (wi * hi ** k) / sum_w_hk
            F.append(Fi)

        # Apply forces to the floor nodes
        for i, node in enumerate(self.cm_node_ids):
            ops.load(node, F[i], 0, 0, 0, 0, 0)

        print(f'Base shear V = {Vb: .2f} kN successfully distributed to each floor.')

        # Run static analysis
        ops.system('BandGeneral')
        ops.numberer('RCM')
        ops.constraints('Transformation')
        ops.integrator('LoadControl', 1.0)
        ops.algorithm('Linear')
        ops.analysis('Static')
        status_code = ops.analyze(1)
        print(f'Analysis complete with status code {status_code}. \n')

    def calc_max_isdr(self):
        isdr = []
        for i in range(self.numFloor):
            drift = ops.nodeDisp(self.cm_node_ids[i], 1) - ops.nodeDisp(self.cm_node_ids[i - 1], 1)
            drift_ratio = drift / self.storyHeight
            isdr.append(drift_ratio)

        return max(isdr)

    def print_results(self):
        # Get Displacement at Roof Node
        disp = ops.nodeDisp(self.cm_node_ids[-1], 1)  # X-direction displacement at node 14
        print(f"Roof Displacement: {disp * 1000:.4f} mm")

        # Check for base reactions and validate with base shear
        ops.reactions()
        Rx = sum([ops.nodeReaction(node, 1) for node in self.nodes[0]])
        print(f"Base Reaction: {Rx: .2f} kN")

        # Compute maximum inter-storey drift ratio

        max_isdr = self.calc_max_isdr()
        print(f"Maximum ISDR is {max_isdr: .6f}.")

    def perform_rsa(self, Z, I, R, num_modes):
        print("\nRunning Response Spectrum Analysis (RSA)...\n")
        Tn = soil_type_acc_coeff("RSA")[0]
        Sa = soil_type_acc_coeff("RSA")[1]
        a = [each * g * Z / 2 * I / R for each in Sa]
        ops.modalProperties()
        ops.timeSeries("Path", 2, "-time", *Tn, "-values", *a)
        disp = []
        isdr = []
        for i in range(len(self.eigenValues)):
            ops.responseSpectrumAnalysis(1, '-Tn', *Tn, '-Sa', *a, '-mode', i + 1)
            roof_disp = ops.nodeDisp(self.cm_node_ids[-1], 1) * 1000
            drift_ratio = self.calc_max_isdr()
            disp.append(roof_disp)
            isdr.append(drift_ratio)

        res_disp = np.sqrt(np.sum(np.array(disp) ** 2))
        res_isdr = np.sqrt(np.sum(np.array(isdr) ** 2))

        print(f'Maximum Roof Displacement (SRSS) = {res_disp:.2f} mm')
        print(f'Maximum ISDR (SRSS) = {res_isdr:.6f}')
