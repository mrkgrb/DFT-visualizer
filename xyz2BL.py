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
    longest_system = max(
        connected_components, key=lambda comp: len(comp)
    )  # Longest fused system
    return [rings[i] for i in longest_system]


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

parser.add_argument("--bond-len", dest="bond_len", action="store_true", help="Show bond lengths. Default: Yes")
parser.add_argument("--pi-sys", dest="bl_pisys", action="store_true", help="Only bond lengths in the pi polycyclic system system will be shown. Default: no")
parser.add_argument("--fs", type=float, default=10.0, help="Font size (default: 10.0).")
parser.set_defaults(bond_len=True)
parser.set_defaults(show_dipole=False)
parser.set_defaults(bl_pisys=False)

args = parser.parse_args()

input_folder = "."  # Use current folder
if args.bond_len:
    atom_scale = 0.33

else:
    atom_scale = 1.0
    
# for all files in the folder
for filename in os.listdir(input_folder):
    if filename.endswith(".xyz"):
        filepath = os.path.join(input_folder, filename)
        file_png=filepath
        file_png = file_png.replace(".xyz", "_BLx.png")
        print(f"Opened: {filename}")
        # Parse the coordinates and shielding data
        at_coords, at_types, dipole_moments = parse_coordinates(filepath)
       
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
        
# project atom corrds onto 2D x,y plane
        at_proj_coords = []
        
        for at,atty in zip(rotated_coords,at_types):
            x_pr = at[0]
            y_pr = at[1]
            at_proj_coords.append((atty,x_pr,y_pr))
            
# find bonds   
        bond_info=[]
        bonds_pi_sys = 0
        
        for i, (x1, y1, z1) in enumerate(rotated_coords):
            for j, (x2, y2, z2) in enumerate(rotated_coords):
                if i >= j:
                    continue  # Avoid double calculation

                distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
                max_distance = 1.3 if "H" in (at_types[i], at_types[j]) else 1.9
                heavy_atoms = False if "H" in (at_types[i], at_types[j]) else True

                if distance <= max_distance:
                    is_in_fused_system = i in fused_atoms or j in fused_atoms # does the bond connect any atoms belonging to the pi system?
                    bond_info.append((i,j,distance,heavy_atoms,is_in_fused_system))
                    if is_in_fused_system:
                        bonds_pi_sys += 1
                        
                    
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
            "F": "#00FF00",  # Green
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
        
        
        x_flip = 1.0
        y_flip = 1.0
        dx, dy = dipoles_xy[0]
        if dipoles_lengths[0] > 0.25:
            if dx < 0:
                x_flip = -x_flip
            if dy < 0:
                y_flip = -y_flip
                
        
        
# Compute the center of mass for visualization reference
        x_center = np.mean(x_coords)
        y_center = np.mean(y_coords)


# Create the plot
        fig, ax = plt.subplots(figsize=(6, 6))


# Draw bonds
        for bond in bond_info:
            _, x1, y1 = at_proj_coords[bond[0]]
            _, x2, y2 = at_proj_coords[bond[1]]
                                     
            plt.plot([x_flip * x1, x_flip * x2], [y_flip * y1, y_flip * y2], color="black", linewidth=1.0, zorder=5) 
                        
            if args.bond_len: # Show bond lengths
                
                b_len = f"{bond[2]:.3f}"
                dx = x2 - x1
                dy = y2 - y1
                bond_center_x = 0.5 * (x1 + x2)
                bond_center_y = 0.5 * (y1 + y2)
                
                label_x = x_flip * bond_center_x 
                label_y = y_flip * bond_center_y 
                
                if np.sqrt(dx * dx + dy * dy) > 0.75 and bond[3]:                                           
                    if not args.bl_pisys or bond[4]:                        
                        angle = np.degrees(np.arctan2(x_flip *dy,y_flip * dx))  # Convert to degrees
                        if angle < -90:
                            angle += 180
                        elif angle > 90:
                            angle -= 180
                        text = ax.text(label_x, label_y, b_len, fontsize=args.fs, ha='center', va='bottom',color="black", rotation=angle, rotation_mode='anchor',  zorder=30, fontweight='ultralight')
                     
# Plot atoms

        for atom, x, y in at_proj_coords:
            color = atom_colors.get(atom, "black")  # Default to black if atom type is not in the dictionary
            size = atom_sizes.get(atom, 30) * atom_scale        # Default size of 30 if atom type is not in the dictionary
            ax.scatter(x_flip * x, y_flip * y, color=color, s=size, label=atom, edgecolor='black',  zorder=10)
            #ax.text(x, y, atom, fontsize=12, ha='center', va='center') # Atom symbols?
            
            
        # Get current limits
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        
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
        plt.savefig(file_png, dpi=300, bbox_inches='tight', pad_inches=0)
        print(f"Saved {file_png}\n")
        plt.close()

        
        
