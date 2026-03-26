"""
The script automatically extracts data from .xyz files in a folder,
reorient the molecules to project them onto the figure plane with the π-system
aligned horizontally.
Bond lengths are then printed onto the structures, and the resulting images are saved as .png files. 
Place it within a folder containing the .xyz files and run from PowerShell (e.g. Anaconda) by:
    python xyz2BL.py [parameters]
    
usage: xyz2BL.py [-h] [--bond-len] [--pi-sys] [--fs FS]

Generate a pic with bond distances.

options:
  -h, --help  show this help message and exit
  --bond-len  Show bond lengths. Default: Yes
  --pi-sys    Only bond lengths in the pi polycyclic system system will be shown. Default: no
  --fs FS     Font size (default: 10.0).
"""


import os
import re
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import itertools

import matplotlib.colors as mcolors

def make_bond_cmap_cc(
    anchors=(#(1.20, "#ff00ff"),   # pink (triple)
             (1.34, "#ff0000"),   # red (double)
             (1.37, "#ddaa00"),   # orange (double)
             (1.395,"#009900"),   # green (aromatic)
             (1.43,"#00bbbb"),   # cyan (int)
             (1.47, "#0000ff")),  # blue (single)
    name="cc_bondlen_cmap"
):
    """
    Returns a matplotlib colormap mapping bond length (Å) to color
    by linear interpolation between anchor points.
    """
    # sort by length
    anchors = sorted(anchors, key=lambda x: x[0])
    Lmin, Lmax = anchors[0][0], anchors[-1][0]

    # positions must be 0..1
    positions = [(L - Lmin) / (Lmax - Lmin) for L, _ in anchors]
    colors = [c for _, c in anchors]

    cmap = mcolors.LinearSegmentedColormap.from_list(
        name, list(zip(positions, colors))
    )
    return cmap, Lmin, Lmax

def make_bond_cmap_HOMA(
    anchors=(
             (0.00, "#009900"),   # green (aromatic)
             (0.00194, "#ddaa00"),   # orange 
             (0.00388,"#ff0000"),  # red (single/double)
             (0.00582,"#bc00bb"),   # pink (int)
             (0.00766, "#0000ff")),  # blue (single)
    name="cc_bondlen_cmap"
):
    """
    Returns a matplotlib colormap mapping bond length (Å) to color
    by linear interpolation between anchor points.
    """
    # sort by length
    anchors = sorted(anchors, key=lambda x: x[0])
    Lmin, Lmax = anchors[0][0], anchors[-1][0]
    

    # positions must be 0..1
    positions = [(L - Lmin) / (Lmax - Lmin) for L, _ in anchors]
    colors = [c for _, c in anchors]

    cmap = mcolors.LinearSegmentedColormap.from_list(
        name, list(zip(positions, colors))
    )
    return cmap, Lmin, Lmax
  
def make_bond_cmap_cc_diff(
    anchors=((-0.030, "#440011"),   # red (double)
             (-0.012, "#ff0000"), 
             (-0.006, "#ff4444"),   # orange (double)
             (0.00,"#eeeeee"),#(0.00,"#009900"),   # green (aromatic)
             (0.006,"#1155ff"),   # cyan (int)
             (0.012, "#0000ff"),  # blue (single)
             (0.030, "#660099")),  # blue (single)
    name="cc_bondlen_cmap"
):
    """
    Returns a matplotlib colormap mapping bond length (Å) to color
    by linear interpolation between anchor points.
    """
    # sort by length
    anchors = sorted(anchors, key=lambda x: x[0])
    Lmin, Lmax = anchors[0][0], anchors[-1][0]

    # positions must be 0..1
    positions = [(L - Lmin) / (Lmax - Lmin) for L, _ in anchors]
    colors = [c for _, c in anchors]

    cmap = mcolors.LinearSegmentedColormap.from_list(
        name, list(zip(positions, colors))
    )
    return cmap, Lmin, Lmax
  
def save_bond_length_colorbar(
    cmap,
    Lmin,
    Lmax,
    filename="cc_bondlength_scale.png",
    label="C–C bond length (Å)",
    figsize=(1.2, 4.5),
    dpi=300,
):
    """
    Save a standalone vertical color bar for bond-length coloring.
    """
    fig, ax = plt.subplots(figsize=figsize)

    norm = mcolors.Normalize(vmin=Lmin, vmax=Lmax)
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax
    )

    cb.set_label(label, rotation=90, labelpad=10)
    cb.ax.tick_params(labelsize=9)

    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def bond_color_from_length_grad_cc(length_ang, cmap, Lmin, Lmax, clamp=True, as_hex=True):
    """
    Map a C–C bond length (Å) to a continuous color from cmap.
    """
    if length_ang is None:
        return "#7f7f7f"  # unknown/gray

    L = float(length_ang)
    if clamp:
        if L < Lmin: L = Lmin
        if L > Lmax: L = Lmax

    t = (L - Lmin) / (Lmax - Lmin)
    rgba = cmap(t)

    return mcolors.to_hex(rgba) if as_hex else rgba

def save_bond_length_colorbar_with_refs(
    cmap,
    Lmin,
    Lmax,
    refs=(1.34, 1.395, 1.47), # (1.20, 1.34, 1.395, 1.47),
    ref_labels=("C=C", "aromatic", "C–C"), #("C≡C", "C=C", "aromatic", "C–C"),
    filename="cc_bondlength_scale_refs.png",
    label="C–C bond length (Å)",
    figsize=(1.4, 5.0),
    dpi=300,
    coltype=1
):
    fig, ax = plt.subplots(figsize=figsize)

    norm = mcolors.Normalize(vmin=Lmin, vmax=Lmax)
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax
    )

    cb.set_label(label, rotation=90, labelpad=10)

    cb.set_ticks(refs)
    if coltype == 1:
        cb.set_ticklabels([f"{r:.2f} ({lbl})" for r, lbl in zip(refs, ref_labels)])
    else:
        cb.set_ticklabels(ref_labels)

    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def extract_coordinates(log_file, section_title):
    """ Extract atomic coordinates from Gaussian log file. """
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    start_idx = None
    for i, line in enumerate(lines):
        if section_title in line:
            start_idx = i + 5  # Skip header lines
            break

    if start_idx is None:
        return None

    coords = []
    for line in lines[start_idx:]:
        if '------' in line:  # End of section
            break
        parts = line.split()
        coords.append(tuple(map(float, parts[3:6])))  # Extract x, y, z

    return coords

def estimate_dipole_moment(at_coordinates, at_types):
    # Rough partial charges dictionary
    partial_charges = {
        "C":  -0.075,
        "H":  +0.117,
        "O":  -0.463,
        "N":  +0.109,
        "S":  +0.0, #avg was +0.6, but there are two types of S: negative -S- and positive in SO2
        "P":  +0.464,
        "Si": +0.2,
        "F":  -0.3,
        "Cl": -0.2,
        "Br": -0.2,
        "B":  +0.3,
    }

    meaned_coord = at_coordinates - np.mean(at_coordinates, axis=0)
    dipole_vector = np.zeros(3)

    for at, coord in zip(at_types, meaned_coord):
        q = 4.8032 * partial_charges.get(at, 0.0)  # Default to 0 if unknown atom
        dipole_vector += q * coord
    
    est_dipole = np.linalg.norm(dipole_vector)
    print(f"Estimated dipole: {est_dipole:.2f} D")
    if est_dipole < 0.618:
        dipole_vector = [0,0,0]
        for at, coord in zip(at_types, meaned_coord):
            q = 5.0
            if at != "C" and at != "H":
                print(f"Freaky atom: {at}")
                dipole_vector = q * coord 
                break

    return dipole_vector    

# Function to parse coordinates from the input orientation section
def parse_coordinates(filepath):

    
    atom_pattern = re.compile(
    r"^\s*([A-Z][a-z]?)\s+(-?\d+(?:\.\d*)?)\s+(-?\d+(?:\.\d*)?)\s+(-?\d+(?:\.\d*)?)\s*$"
    )
    
    dipole_pattern = re.compile(
    r"\s+X=\s+(-?\d+\.\d+)\s+Y=\s+(-?\d+\.\d+)\s+Z=\s+(-?\d+\.\d+)"    
    )
    
    at_coordinates = []
    at_types = []
    dipole_moments = []
    with open(filepath, 'r') as file:
        for line in file:
            match = atom_pattern.match(line)
            if match:
                at, x, y, z = match.groups()
                if at != 'Bq':
                    at_coordinates.append(np.array([float(x), float(y), float(z)]))
                    at_types.append(at)
            else:
                match = dipole_pattern.match(line)
                if match:
                    x, y, z = map(float, match.groups())
                    dipole_moments.append([x,y,z])
                        
    stnd_cords = extract_coordinates(filepath, "Standard orientation:")
    
    
    if stnd_cords:
        st_len = len(stnd_cords)
        print(f"Standard coords: {st_len}")
        if st_len == len(at_coordinates):
            at_coordinates = stnd_cords
                    
    #numa = len(bq_coordinates)
    numa_ac= len(at_coordinates)
    numa_dp = len(dipole_moments)
    
    if numa_dp == 0:
        dipole_moments.append(estimate_dipole_moment(at_coordinates, at_types))
    
    
    print(f"Atomic coords: {numa_ac}, Dipole moments: {numa_dp}")  
    return np.array(at_coordinates), at_types, dipole_moments


def construct_graph(atoms, coords, bond_threshold=1.9):
    """Construct a molecular graph from atomic coordinates."""
    G = nx.Graph()
    num_atoms = len(atoms)
    for i in range(num_atoms):
        if atoms[i] != 'H':
            G.add_node(i, element=atoms[i], pos=coords[i])
            for j in range(i + 1, num_atoms):
                if atoms[j] != 'H':
                    dist = np.linalg.norm(coords[i] - coords[j])
                    if dist <= bond_threshold:  # Adjust bond length tolerance
                        G.add_edge(i, j)
    return G


def find_non_hydrogen_rings(graph, atoms):
    """Identify rings formed only by non-hydrogen atoms."""
    all_rings = nx.cycle_basis(graph)
    non_hydrogen_rings = [
        ring for ring in all_rings if all(atoms[i] != 'H' for i in ring)
    ]
    return non_hydrogen_rings


def filter_fused_ring_systems(graph, rings):
    """Group rings into fused systems."""
    fused_graph = nx.Graph()
    for i, ring1 in enumerate(rings):
        for j, ring2 in enumerate(rings):
            if i < j:
                shared_edges = [
                    tuple(sorted((ring1[k], ring1[(k + 1) % len(ring1)])))
                    for k in range(len(ring1))
                ]
                if any(
                    tuple(sorted((ring2[k], ring2[(k + 1) % len(ring2)]))) in shared_edges
                    for k in range(len(ring2))
                ):
                    fused_graph.add_edge(i, j)
    connected_components = list(nx.connected_components(fused_graph))
    print(f"Con comp: {len(connected_components)}")
    if len(connected_components) > 0:
        
        longest_system = max(
            connected_components, key=lambda comp: len(comp)
        )  # Longest fused system
        return [rings[i] for i in longest_system]
    else:
        return rings


def rotation_matrix_from_axis_angle(axis, angle):
    """
    Generate a rotation matrix for a given axis and angle using Rodrigues' formula.
    """
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    identity = np.eye(3)
    return identity + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def align_molecule_to_xy_and_x(principal_coords,coords,dipoles):
    """
    Align the molecule's principal coordinates:
    - Align the mean plane of the largest polycyclic system to the XY plane.
    - Align the principal axis of the molecule to the X axis.
    """

    # Step 1: Calculate the mean plane normal using SVD
    centroid_mean = np.mean(principal_coords, axis=0)
    centered_coords = principal_coords - centroid_mean
    _, _, vh = np.linalg.svd(centered_coords, full_matrices=False)
    normal_vector = vh[-1]  # Normal to the mean plane (smallest singular vector)

    centroid_mean_all = np.mean(coords, axis=0)
    centered_coords_all = coords - centroid_mean_all
    
    # Ensure the normal vector points upwards (positive z-direction)
    if normal_vector[2] < 0:
        normal_vector = -normal_vector

    # Step 2: Rotate the molecule to align the mean plane with the XY plane
    z_axis = np.array([0, 0, 1])
    axis_of_rotation = np.cross(normal_vector, z_axis)
    angle = np.arccos(np.dot(normal_vector, z_axis))
    if np.linalg.norm(axis_of_rotation) > 1e-6:  # Avoid division by zero
        axis_of_rotation /= np.linalg.norm(axis_of_rotation)
        rotation_matrix = rotation_matrix_from_axis_angle(axis_of_rotation, angle)
        aligned_coords = centered_coords @ rotation_matrix.T
        aligned_coords_all = centered_coords_all @ rotation_matrix.T
        aligned_dipoles = dipoles @ rotation_matrix.T
    else:
        aligned_coords = centered_coords  # Already aligned with z-axis
        aligned_coords_all = centered_coords_all  # Already aligned with z-axis
        aligned_dipoles = dipoles

    # Step 3: Align the principal axis to the X axis
    cov_matrix = np.cov(aligned_coords.T)
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    principal_axis = eigvecs[:, -1]  # Principal eigenvector (largest eigenvalue)

    # Ensure the principal axis points in the positive x-direction
    if principal_axis[0] < 0:
        principal_axis = -principal_axis

    x_axis = np.array([1, 0, 0])
    axis_of_rotation = np.cross(principal_axis, x_axis)
    angle = np.arccos(np.dot(principal_axis, x_axis))
    if np.linalg.norm(axis_of_rotation) > 1e-6:
        axis_of_rotation /= np.linalg.norm(axis_of_rotation)
        rotation_matrix = rotation_matrix_from_axis_angle(axis_of_rotation, angle)
        aligned_coords = aligned_coords @ rotation_matrix.T
        aligned_coords_all = aligned_coords_all @ rotation_matrix.T
        aligned_dipoles = aligned_dipoles @ rotation_matrix.T

    return aligned_coords_all, aligned_dipoles


       
# Main script

# get arguments
parser = argparse.ArgumentParser(description="Generate a pic with bond distances.")

parser.add_argument("--BL-off", dest="bond_len_off", action="store_true", help="Hide bond lengths. Default: No")
#parser.add_argument("--color-BL", dest="bond_color", action="store_true", help="Color bonds by length. Default: No")
parser.add_argument("--color-BL", dest="bond_color", type=int, default=0, help="Color bonds by length. Specify: 0 - off (default), 1 - RGB, 2 - HOMA")
parser.add_argument("--pi-sys", dest="bl_pisys", action="store_true", help="Only bond lengths in the pi polycyclic system system will be shown. Default: no")
parser.add_argument("--spec-at", dest="sp_at", type=str, default="", help="Always show distance between these two atoms, e.g.: N-N . Default: none")
parser.add_argument("--fs", type=float, default=12.0, help="Font size (default: 12.0).")
parser.add_argument("--save-rot", dest="sv_rot", action="store_true", help="Save the rotatet coordinates in a xyz subfolder. Default: no")
parser.add_argument("--or-atm", dest="atm", type=str, default="", help="Specify atom to be shown in bottom left corner (default: none)")
parser.add_argument("--res", type=float, default=90.0, help="Image resolution in pixels per angstrom (default: 90).")
parser.add_argument("--pol-diff", dest="pol_diff", action="store_true", help="Find pairs in different solvents and display changes in bond lengths. Default: no")
parser.add_argument("--sv1", type=str, default="Hex", help="Enter solvent 1 abbr (default: Hex)")
parser.add_argument("--sv2", type=str, default="MeOH", help="Enter solvent 2 abbr (default: MeOH)")
parser.add_argument("--perc", dest="perc", action="store_true", help="Display bond length change in percent. Default: mÅ")


parser.set_defaults(bond_len_off=False)
parser.set_defaults(show_dipole=False)
parser.set_defaults(bl_pisys=False)
parser.set_defaults(sv_rot=False)
parser.set_defaults(bond_color=False)
parser.set_defaults(pol_diff=False)
parser.set_defaults(perc=False)

args = parser.parse_args()

solvent1 = args.sv1
solvent2 = args.sv2

b_min = 2.0
b_max = 0.0

cc_min = 2.0
cc_max = 0.0
blx_max = -2.0
blx_min = 2.0
cc_b_stat = 0.0 
cc_b_n = 0
cc_pi_stat = 0.0
cc_pi_n = 0

pix_per_angstrom = args.res
input_folder = "."  # Use current folder
#if not args.bond_len_off:
atom_scale = 1.00
lw_scale = 2.0
#else:
#    atom_scale = 1.0

special_atm = args.atm
if special_atm:
    print(f"Orient by atom: {special_atm}")    

sp_bnd_str = args.sp_at    
special_bonds = sp_bnd_str.split() 
print(f"Special bonds: {sp_bnd_str}")
bn_no = 0
spec_bond_pairs = []
for bond in special_bonds:
    bn_no += 1
    pair = bond.split("-")
    spec_bond_pairs.append(pair)
    print(f"Special bond {bn_no}: {bond}")


output_folder = "xyzs"
if args.sv_rot:
    os.makedirs(output_folder, exist_ok=True)

#color map
bl_col_type = args.bond_color
unit = "Å"
if bl_col_type == 1:
    if not args.pol_diff:
        unit = "Å"
        cmap, Lmin, Lmax = make_bond_cmap_cc()
        
        save_bond_length_colorbar_with_refs(
            cmap, Lmin, Lmax,
            filename="cc_bondlength_scale1.png"
        )
    else:
        cmap, Lmin, Lmax = make_bond_cmap_cc_diff()
        unit = "mÅ" if not args.perc else "%"
        save_bond_length_colorbar_with_refs(
            cmap, Lmin, Lmax,
            filename="cc_bondlength_diff_scale1.png",
            label=f"bond length change ({unit})",
            refs=(-0.030, 0.0, 0.030),
            ref_labels=("-30.0 (shorter)", "0.0 (equal)", "+30.0 (longer)"),
            coltype = 2
        )
elif bl_col_type == 2:
    if not args.pol_diff:
        cmap, Lmin, Lmax = make_bond_cmap_HOMA()
        unit = "Å"
        save_bond_length_colorbar_with_refs(
            cmap, Lmin, Lmax,
            filename="cc_bondlength_scale2.png",
            refs=(0.00, 0.00388, 0.00776),
            ref_labels=("1.39 (aromatic)", "1.33 / 1.45 (C=C/C-C)", "1.26 / 1.51 (non-aromatic)"),
            coltype = 2
        )
    else:
        unit = "mÅ" if not args.perc else "%"
        cmap, Lmin, Lmax = make_bond_cmap_HOMA(
            anchors=((-0.0035, "#8800ff"),  # violet (single)
                     (-0.0010, "#0000ff"),  # blue (single)
                     #(-0.0005, "#00ddbb"),   # cyan (int)
                     (0.0000,"#dddddd"),   # green (aromatic)
                     #(0.0005,"#ddbb00"),   # orange (double)
                     (0.0010, "#ff0000"), # red (double)
                     (0.0020, "#bc00bb"))   # pink (double)
            
            )
        

        
        save_bond_length_colorbar_with_refs(
            cmap, Lmin, Lmax,
            filename="cc_bondlength_diff_scale2.png",
            refs=(-0.0035, 0.000, 0.0020),
            ref_labels=("-0.0035 (aromatize)", "0.0 (no change)", "0.0020 (des-aromatize)"),
            coltype = 2,
            label="HOMA contributor change",
        )
        

file_end = "_BLx.png" if not args.bond_len_off else "_x.png"
if args.bond_color:
    
    file_end = f"_BLxc{bl_col_type}.png" if not args.bond_len_off else f"_xc{bl_col_type}.png"


summary_lines = []
lp = 0 
# for all files in the folder
tasks = []
filelist = []

files_data = []

for filename in os.listdir(input_folder):
    if filename.endswith(".xyz"):
        
        filelist.append(filename)

        
        base_name = os.path.splitext(filename)[0]
        parts = base_name.split("-")
        molecule_name = parts[0]
        solvent_abbr = parts[1] if len(parts) > 1 else "Vac"  # Default solvent is DCM 
        #s_or_t = "t" if (filename.endswith("tNMP.log") or filename.find("-tNMP")!=-1) else "s"
        files_data.append({"file": filename, "cpd": molecule_name, "slv": solvent_abbr })
        
        
if not args.pol_diff:
    for i, filename in enumerate(filelist):
        tasks.append((i,-1))
            

    
mol_pairs = []
diff_type = []

if args.pol_diff:

    for i, file1 in enumerate(files_data):
 
       for j, file2 in enumerate(files_data):
           if i >= j:
               continue
           if file1['cpd'] == file2['cpd']:
               
              if (file1['slv'] == solvent2) and (file2['slv'] == solvent1):
                   mol_pairs.append((j,i))
                   diff_type.append("pol")
                   print(f"Pair: {file2['file']} and {file1['file']}")
              elif (file2['slv'] == solvent2) and (file1['slv'] == solvent1):
                   mol_pairs.append((i,j))
                   diff_type.append("pol")
                   print(f"Pair: {file1['file']} and {file2['file']}")
    
    if len(mol_pairs)>0:
        tasks = mol_pairs

unit_d = unit
if unit != "%":
    unit_d = f" {unit}"

for task in tasks:
    #if filename.endswith(".xyz"):
        b_min_ex = 5.0
        b_max_ex = 0.0
        filename = files_data[task[0]]['file']
        
        
        filepath = os.path.join(input_folder, filename)
        file_png = filepath
        
        if args.pol_diff:
            result_file_name = f"{files_data[task[0]]['cpd']}-dif{solvent1}-{solvent2}{file_end}"
            #print(f"out: {result_file_name}")
            file_png = os.path.join(input_folder, result_file_name)
            
            filename2 = files_data[task[1]]['file']
            filepath2 = os.path.join(input_folder, filename2)

        
        
        file_png = file_png.replace(".xyz", file_end ) # if not args.pol_diff else 
        print(f"Opened: {filename}")
        if args.sv_rot:
            output_xyz_file = os.path.join(output_folder, filename)
            print(f"Output xyz: {output_xyz_file}")
        
        lp += 1
        summary_line = f"{lp}.\t{filename}"
        
        # Parse the coordinates and shielding data
        at_coords, at_types, dipole_moments = parse_coordinates(filepath)
        if args.pol_diff:
            at_coords2, at_types2, dipole_moments2 = parse_coordinates(filepath2)
        
        # Construct the molecular graph
        graph = construct_graph(at_types, at_coords, bond_threshold=1.92)
        
        # Find rings in the molecular graph
        rings = find_non_hydrogen_rings(graph, at_types)
        fused_ring_systems = filter_fused_ring_systems(graph, rings)
        
        if not fused_ring_systems:
            print("No suitable fused ring system detected.")
            
        else:
        
            # Consider only the longest fused system
            longest_fused_system = fused_ring_systems  
            
            # Calculate mean plane and rotate molecule
            system_coords = np.concatenate([at_coords[ring] for ring in longest_fused_system])
            
            rotated_coords, rotated_dipoles = align_molecule_to_xy_and_x(system_coords,at_coords,dipole_moments) # rotate_to_xy_plane(coords, mean_normal)
            # list of pi-system or fused system atoms:
            fused_atoms = set(np.concatenate(longest_fused_system))     
            n_pi_sys = len(fused_atoms)
            print(f"{n_pi_sys} atoms in pi system.")
            if args.pol_diff:
                system_coords2 = np.concatenate([at_coords2[ring] for ring in longest_fused_system])
                
                rotated_coords2, rotated_dipoles2 = align_molecule_to_xy_and_x(system_coords2,at_coords2,dipole_moments2) # rotate_to_xy_plane(coords, mean_normal)
                # list of pi-system or fused system atoms:

        
        
# project atom corrds onto 2D x,y plane
        at_proj_coords = []

        if not args.pol_diff:
            for at,atty in zip(rotated_coords,at_types):
                x_pr = at[0]
                y_pr = at[1]
                at_proj_coords.append((atty,x_pr,y_pr))
        else:
            for at,at2,atty in zip(rotated_coords,rotated_coords2,at_types):
                x_pr = 0.5 * (at[0] + at2[0])
                y_pr = 0.5 * (at[1] + at2[1])
                at_proj_coords.append((atty,x_pr,y_pr))
           

        x_pi_avg = 0.0
        y_pi_avg = 0.0
        n_avg = 0.0            
        for i,(at,x,y) in enumerate(at_proj_coords):
            if i in fused_atoms:
                x_pi_avg += x
                y_pi_avg += y
                n_avg += 1
        
        if n_avg > 0.1:
            x_pi_avg /= n_avg
            y_pi_avg /= n_avg                
            print(f"Avg pisys x,y: {x_pi_avg:.2f},{y_pi_avg:.2f}; n= {n_avg:.1f}")
            
            
# find bonds   
        bond_info=[]
        bonds_pi_sys = 0
        #coords_sv = []
        xyz_avg = [0.0,0.0,0.0]
        xyz_n = 0
        for i, (x1, y1, z1) in enumerate(rotated_coords):
            for j, (x2, y2, z2) in enumerate(rotated_coords):
                if i >= j:
                    continue  # Avoid double calculation
                

                
                distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
                
                if args.pol_diff:
                    xb1, yb1, zb1 = rotated_coords2[i]
                    xb2, yb2, zb2 = rotated_coords2[j]
                    distance2 = np.sqrt((xb2 - xb1) ** 2 + (yb2 - yb1) ** 2 + (zb2 - zb1) ** 2)
                else:
                    distance2 = distance
                
                max_distance = 1.3 if "H" in (at_types[i], at_types[j]) else 1.9
                heavy_atoms = False if "H" in (at_types[i], at_types[j]) else True
                b_type = f"{at_types[i]}-{at_types[j]}"
                special_bond = False
                if len(spec_bond_pairs)>0:
                    for sp_pair in spec_bond_pairs:
                        if len(sp_pair)==2:      
                            special_bond = True if (at_types[i] == sp_pair[0] and at_types[j] == sp_pair[1]) or (at_types[i] == sp_pair[1] and at_types[j] == sp_pair[0]) else False
                            if special_bond:
                                break
                

                if distance <= max_distance or special_bond:
                    is_in_fused_system = i in fused_atoms or j in fused_atoms # does the bond connect any atoms belonging to the pi system?
                    bond_info.append((i,j,distance,heavy_atoms,is_in_fused_system,special_bond, distance2, b_type))
                    
                    if is_in_fused_system:
                        bonds_pi_sys += 1
                
            #coords_sv.append([x1,y1,z1])
            if not args.bl_pisys or i in fused_atoms:
                #xyz_avg += [x1,y1,z1]
                xyz_avg[0] += x1
                xyz_avg[1] += y1
                xyz_avg[2] += z1
                xyz_n += 1 
            
        xyz_avg_r = [0.0,0.0,0.0]
        xyz_n_r = 0 
        for i, (x1,y1,z1) in enumerate(at_coords):
            if not args.bl_pisys or i in fused_atoms: 
                xyz_avg_r[0] += x1
                xyz_avg_r[1] += y1
                xyz_avg_r[2] += z1
                xyz_n_r += 1
                    
                
        if xyz_n > 0:
            xyz_avg[0] /= float(xyz_n)
            xyz_avg[1] /= float(xyz_n)
            xyz_avg[2] /= float(xyz_n)
            
        if xyz_n_r > 0:
            xyz_avg_r[0] /= float(xyz_n_r)
            xyz_avg_r[1] /= float(xyz_n_r)
            xyz_avg_r[2] /= float(xyz_n_r)
            
            print(f"Avg x,y,z: {xyz_avg_r[0]:.3f}, {xyz_avg_r[1]:.3f}, {xyz_avg_r[2]:.3f}, n = {xyz_n_r}")
                          
             
        number_bonds = len(bond_info)    
        print(f"Bonds found: {number_bonds} total, {bonds_pi_sys} in pi system")


        
        atom_colors = {
            "C": "#666666",  # Grey
            "H": "#FFFFFF",  # White
            "O": "#FF0000",  # Red
            "N": "#0000FF",  # Blue
            "S": "#FFFF00",  # Yellow
            "P": "#FFA500",  # Orange
            "Si": "#00AAAA", # Cyan
            "F": "#00FFAA",  # Green
            "Cl": "#00FF00",  # Green
            "Br": "#8B0000",  # Dark red
            "B": "#FFB6C1",  # Peach
        }

        default_color = "#FFC0CB"  # Pink for any other element

        atom_sizes = {
            "C": 120,
            "H": 60,
            "O": 160,
            "N": 140,
            "S": 180,
            "P": 180,
            "Si": 180,
            "F": 120,
            "Cl": 140,
            "Br": 160,
            "B": 100,
        }
        
        colors = ["green", "red", "blue", "purple", "orange", "brown"] #dipole colors
        color_cycle = itertools.cycle(colors)
        
        atoms, x_coords, y_coords = zip(*at_proj_coords)
        dipole_scale = -0.2082  # scale the dipole to point from positive to negative charge and reflect real magnitude of charge separation (1 Debye ~ 4.80 angstrom)
        
# Compute dipole projection onto the xy-plane
        dipoles_xy = np.array(rotated_dipoles)[:, :2] * dipole_scale # Take only x and y components
        dipoles_lengths = []
        for dipole_m in rotated_dipoles:
            dipoles_lengths.append(np.linalg.norm(dipole_m))
        
        
# orientacja!
        x_flip = 1.0
        y_flip = 1.0
        if not special_atm:

            dx, dy = dipoles_xy[0]
            if dipoles_lengths[0] > 0.25:
                if dx < 0:
                    x_flip = -x_flip
                if dy < 0:
                    y_flip = -y_flip
        else:
            for at,x,y in at_proj_coords:
                if at == special_atm:
                    dx = x - x_pi_avg
                    dy = y - y_pi_avg
                    dr = (dx * dx + dy * dy) ** 0.5
                    if dr > 0.1:
                        if dx > 0.0:
                            x_flip = -x_flip
                        if dy > 0.0:
                            y_flip = -y_flip
                    
                    break
                
        if args.sv_rot and not args.pol_diff:
            # Write the output
            with open(output_xyz_file, "w") as f:


                for atom, coord in zip(at_types, rotated_coords):

                    
                    x = (coord[0] - xyz_avg[0]) * x_flip
                    y = (coord[1] - xyz_avg[1]) * y_flip
                    z = (coord[2] - xyz_avg[2]) * x_flip * y_flip
                                             
                    f.write(f"{atom} {x:.8f} {y:.8f} {z:.8f}\n")        
        
# Compute the center of mass for visualization reference
        x_center = np.mean(x_coords)
        y_center = np.mean(y_coords)
        #print(f"Avg x,y: {x_center:.2f},{y_center:.2f}")
        
        #size and scales
        xspan = np.max(x_coords) - np.min(x_coords)
        yspan = np.max(y_coords) - np.min(y_coords)
        #xspan = x_coords.max() - x_coords.min()
        
        border_w = 30
        image_width = int(float(pix_per_angstrom * xspan))
        scaling_factor = float(float(pix_per_angstrom) / 90.0)
        image_height = int(float(image_width) * yspan / xspan)
        image_width += border_w
        image_height += border_w
        print(f"X span: {xspan:.2f} Å, Y span: {yspan:.2f} Å")
        print(f"Est. W/H {image_width}/{image_height} pix, Scale: {scaling_factor:.2f}")
# Create the plot

        dpi = 150
        desired_w_px = int(image_width)    
        desired_h_px = int(image_height)   
        fig, ax = plt.subplots(figsize=(desired_w_px/dpi, desired_h_px/dpi))


# Draw bonds

        for bond in bond_info:
            _, x1, y1 = at_proj_coords[bond[0]]
            _, x2, y2 = at_proj_coords[bond[1]]
            
            l_style = '-'
            
            if not args.bond_color: # or not bond[3]:
                
                l_color="black"
                lw = 1.00
            elif args.bond_color:
                if bl_col_type == 1:
                    if not args.pol_diff:
                        blx = bond[2]
                    elif not args.perc:
                        blx = bond[6] - bond[2]
                    else:
                        blx = 2.0 * (bond[6] - bond[2])/(bond[6] + bond[2])
                    
                    l_color= bond_color_from_length_grad_cc(blx , cmap, Lmin, Lmax) 
                    lw = 3.00
                    if blx < blx_min:
                        blx_min = blx
                    elif blx > blx_max:
                        blx_max = blx
                elif bl_col_type == 2:
                    if bond[7] == "C-C":
                        
                        if not args.pol_diff:
                            blx = bond[2]
                            blx = blx - 1.388
                            blx = blx * blx
                            if blx < blx_min:
                                blx_min = blx
                            elif blx > blx_max:
                                blx_max = blx
                        else:
                            blx = (bond[6] - 1.388) ** 2 - (bond[2] - 1.388) ** 2
                            if blx < blx_min:
                                blx_min = blx
                            elif blx > blx_max:
                                blx_max = blx
                        
                        l_color= bond_color_from_length_grad_cc(blx , cmap, Lmin, Lmax) 
                        lw = 3.00
                        
                    else:
                        l_color="black"
                        lw = 1.00

                    

            
            
            if bond[5]:
                lw = 1.50
                l_color = "#000099"  
                l_style = '--'  
                summary_line = '\t'.join([summary_line,f"{at_types[bond[0]]}-{at_types[bond[1]]}:",f"{bond[2]:.3f}"])                
                     
            plt.plot([x_flip * x1, x_flip * x2], [y_flip * y1, y_flip * y2], color=l_color, linewidth=lw*scaling_factor * lw_scale, zorder=5,linestyle=l_style) 
            
                        
            if not args.bond_len_off: # Show bond lengths
                

                            
                if not args.pol_diff:
                    db = bond[2]
                    b_len = f"{bond[2]:.3f}" 
                    if db < b_min_ex:
                        b_min_ex = db
                    elif db > b_max_ex:
                        b_max_ex = db
                        
                    if bond[7] == "C-C":
                        if db < cc_min:
                            cc_min = db
                        elif db > cc_max:
                            cc_max = db
                        cc_b_stat += db
                        cc_b_n += 1 
                        if bond[4]:
                            cc_pi_stat += db
                            cc_pi_n += 1     
                        
                        
                else:
                    db = 1000.0* (bond[6]-bond[2]) if not args.perc else 200.0 * (bond[6] - bond[2])/(bond[6] + bond[2])
                    
                    if db < b_min_ex:
                        b_min_ex = db
                    elif db > b_max_ex:
                        b_max_ex = db
                    if bond[7] == "C-C":
                        if db < cc_min:
                            cc_min = db
                        elif db > cc_max:
                            cc_max = db
                    
                    if not args.perc:
                        b_len = f"−{-db:.1f}" if db < 0 else f"+{db:.1f}"
                    else:
                        b_len = f"−{-db:.2f}" if db < 0 else f"+{db:.2f}"
                    
                
                
                    
                fstyle = 'ultralight' if not bond[5] else 'bold'
                fcolor = "black" if not bond[5] else l_color
                dx = x2 - x1
                dy = y2 - y1
                bond_center_x = 0.5 * (x1 + x2)
                bond_center_y = 0.5 * (y1 + y2)
                
                label_x = x_flip * bond_center_x 
                label_y = y_flip * bond_center_y 
                
                if (np.sqrt(dx * dx + dy * dy) > 0.75 and bond[3]) or bond[5]:                                           
                    if not args.bl_pisys or bond[4]:                        
                        angle = np.degrees(np.arctan2(x_flip *dy,y_flip * dx))  # Convert to degrees
                        if angle < -90:
                            angle += 180
                        elif angle > 90:
                            angle -= 180
                        text = ax.text(label_x, label_y, b_len, fontsize=args.fs*scaling_factor, ha='center', va='bottom',color=fcolor, rotation=angle, rotation_mode='anchor',  zorder=30, fontweight=fstyle)
                     
# Plot atoms

        for atom, x, y in at_proj_coords:
            lw = 1.0
            color = atom_colors.get(atom, "black")  # Default to black if atom type is not in the dictionary
            size = atom_sizes.get(atom, 30) * atom_scale        # Default size of 30 if atom type is not in the dictionary
            ax.scatter(x_flip * x, y_flip * y, color=color, s=size*scaling_factor, linewidth=lw*scaling_factor * lw_scale, label=atom, edgecolor='black',  zorder=10)
            #ax.text(x, y, atom, fontsize=12, ha='center', va='center') # Atom symbols?
            
            
        # Get current limits
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        

            
        #add legend with units
        if not args.bond_len_off: 

            label_x = np.max(x_coords) if x_flip > 0 else -np.min(x_coords)
            label_y = np.min(y_coords) - 0.25 if y_flip > 0 else -np.max(y_coords) - 0.25
            
            
            b_len = f"Unit: {unit}, min/max: {b_min_ex:.3f}/{b_max_ex:.3f}{unit_d}"
            fstyle = 'ultralight' 
            fcolor = "black" 
            
            text = ax.text(label_x, label_y, b_len, fontsize=args.fs*scaling_factor, ha='right', va='top',color=fcolor,   zorder=40, fontweight=fstyle)
        
            if b_min_ex < b_min:
                b_min = b_min_ex
            elif b_max_ex > b_max:
                b_max = b_max_ex
        
        # Extend axis limits to include dipole arrows
        for (dx, dy) in dipoles_xy:
            tip_x = x_flip * x_center + x_flip * dx
            tip_y = y_flip * y_center + y_flip * dy  # account for vertical offset
            x_min = min(x_min, tip_x)
            x_max = max(x_max, tip_x)
            y_min = min(y_min, tip_y)
            y_max = max(y_max, tip_y)
        
        # Set new limits with margin
        margin = 0.0  # Optional margin
        ax.set_xlim(x_min - margin, x_max + margin)       
        ax.set_ylim(y_min - margin, y_max + margin)
# Formatting
        ax.set_aspect('equal')
        ax.relim()
        ax.autoscale_view()
        ax.set_xticks([])
        ax.set_yticks([])
        
        for spine in ax.spines.values():
            spine.set_visible(False)
     
        ax.set_xlabel("")
        ax.set_ylabel("")            
        
        plt.subplots_adjust(left=0.001, right=0.999, top=0.999, bottom=0.001)
        
# Save the figure
        plt.savefig(file_png, dpi=dpi, bbox_inches='tight', pad_inches=0)
        
        print(f"{summary_line}")
        print(f"Saved {file_png}\n")
        plt.close()
        
        summary_lines.append(summary_line)

if len(summary_lines)>0:
    outpath = "xyz2BL_summary.txt"
    with open(outpath, "w") as out:
        #out.write("Lp.\t" + "\t".join(cols) + "\n")
        for i, r in enumerate(summary_lines):
            out.write(r + "\n")

print(f"Shortest/longest bonds (diffs): {b_min:.3f}/{b_max:.3f}{unit_d}")
print(f"Shortest/longest C-C bonds (diffs): {cc_min:.3f}/{cc_max:.3f}{unit_d}")
print(f"Smallest/largest HOMA contributor (diffs): {blx_min:.4f}/{blx_max:.4f}")     
if cc_pi_n > 0 and cc_b_n >0:
    cc_pi_stat /= float(cc_pi_n)  
    cc_b_stat /= float(cc_b_n)       
    print(f"Average C-C: {cc_b_stat:.3f} Å (n={cc_b_n}), in pi system: {cc_pi_stat:.3f} Å (n={cc_pi_n})") 
    
    
    