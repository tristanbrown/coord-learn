from ccdc import io
import pandas as pd
import numpy as np
import time

np.random.seed(901)
csd_reader = io.MoleculeReader('CSD')

class Mol():
    """
    A wrapper class for csd molecule objects. 
    """
    def __init__(self, index):
        self._molecule = self.get_mol(index)
    
    def __getattr__(self, attr):
        """Wraps this class object around a CSD molecule object."""
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self._molecule, attr)
    
    def get_mol(self, index):
        """Acquires a molecule object from the CSD, using either the string 
        label for the structure, or its numerical index."""
        try: 
            return csd_reader[index]
        except TypeError:
            return csd_reader.molecule(index)
    
    def remove_unlocated(self):
        """Removes all atoms in a molecule that are missing coordinates."""
        for atom in self.atoms:
            if atom.coordinates is None:
                self.remove_atom(atom)
    
    def center(self):
        """Centers the molecule, eliminating unlocated atoms."""
        try:
            self.remove_unlocated()
            self.translate([round(-1 * a, 4)
                            for a in self.centre_of_geometry()])
            return True
        except ValueError:
            return False
        
    @property
    def xyz(self):
        """Returns a dataframe of the molecule's atomic coordinates in
        a format similar to .xyz files.
        """
        atoms = self.atoms
        coords = [[atom.atomic_symbol] + 
                    self.coord_format(atom.coordinates)
                        for atom in atoms]
            #use atom.label to get the symbol and index number together
        return pd.DataFrame(coords, columns=['Element', 'x', 'y', 'z'])
    
    def coord_format(self, coord):
        """Rounds coordinates or deals with missing data."""
        try:
            return [round(a, 4) for a in coord]
        except TypeError:
            return [None, None, None]
    
    def element_distances(self, elem):
        """Takes an element type and returns a dict of dataframes, representing
        the same molecule centered on each instance of that element.
        The molecule is represented entirely in terms of radii from the central
        atom, and the atoms are ordered by these distances, smallest to largest.
        Each key in the dictionary is the label of the centered atom. 
        """
        labels = self.find_elements(elem)
        return {atom: self.relative_distances(atom) for atom in labels}
        
    
    def find_elements(self, elem):
        """Returns a list of the atom labels of a given element type."""
        atoms = self.atoms
        return [atom.label for atom in atoms if atom.atomic_symbol == elem]
    
    def element_count(self, elem):
        """Returns the count of a specific type of element in the molecule."""
        return len(self.find_elements(elem))
    
    def relative_distances(self, label):
        """Takes an atom label and gives a sorted dataframe containing the
        distance of every element in the molecule to that atom."""
        atoms = self.atoms
        central = self.atom(label)
        distances = [[atom.atomic_symbol]
                 + [self.atom_distance(atom, central)] for atom in self.atoms] 
        unsortedframe = pd.DataFrame(distances, columns=['Element', 'r'])
        return unsortedframe.sort_values('r').reset_index(drop=True)
    
    def atom_distance(self, atom1, atom2):
        """Returns the distance of the atom from the given point in space."""
        point1 = atom1.coordinates
        point2 = atom2.coordinates
        ssd = [(x1 - x2)**2 for x1, x2 in zip(point1, point2)]
        return round((sum(ssd)**(0.5)), 4)
    
    def coordination_num(self, atomlabel):
        return len(self.atom(atomlabel).neighbours)

                
class Molset():
    """
    __init__:
    Takes either a list of structure identifier symbols, or an integer value n 
    representing the first n entries in the CSD, and generates a dictionary
    containing a subset of the molecule objects in the CSD. 
    
    """
    def __init__(self, ids=[], elem=None, num_nearest=20, max=5000, version=1):
        self.elem = elem
        self.V = version
        self.ids = ids
        self.max = max
        self.mols = self.populate_mols()
        self.Periodic_Table = pd.read_csv('element_data.csv',
                                        delimiter=',', header=0, index_col=0)
        self.prepare_data(self.elem, num_nearest)
        
        # self.center_all()
        # self.xyzset = self.populate_xyz()
        # self.xyzset = self.centered_xyz()
    
    def populate_mols(self):
        """Populates self.mols using a list of string identifiers, or a list of
        numerical indices. If instead a number n is given directly, n fully-3D
        entries are chosen from the CSD to populate self.mols."""
        try:
            mols = {}
            for id in self.ids:
                amol = Mol(id)
                mols[amol.identifier] = amol
        except TypeError:
            self.count = self.ids
            mols = self.random_populate(self.count, self.elem)
        return mols
    
    def random_populate(self, count, elem):
        """Returns a dict of Mol objects populated at random. The number of
        objects is either the count, or if an element is given, by the total 
        number of instances of that element in the set of molecules."""
        time1 = time.time()
        mols = {}
        csd_size = len(csd_reader)
        checked = 0
        while len(mols) < count and checked < self.max:
            id = np.random.randint(0, csd_size)
            checked += 1
            amol = Mol(id)
            label = amol.identifier
            if amol.all_atoms_have_sites:
                if elem is None:
                    amol.normalise_labels()
                    mols[label] = amol
                else:
                    n_atoms = amol.element_count(elem)
                    if n_atoms > 0:
                        amol.normalise_labels()
                        mols[label] = amol
                        count = count - n_atoms + 1
        time2 = time.time()
        
        if checked == self.max:
            self.count = len(mols)
        
        print("""Spent %.1f sec checking %d CSD entries, finding %d samples of 
                the element, %s."""
                    % (round(time2 - time1, 2), checked, self.count, elem))
        return mols
    
    def center_all(self):
        """Use to re-center all Mols."""
        return [mol.center() for id, mol in self.mols.items()]
            
    
    def populate_xyz(self):
        return {id: self.mols[id].xyz for id in self.mols}
    
    def centered_xyz(self):
        return {id: mol.xyz for id, mol in self.mols.items() 
                    if mol.center()}
    
    def prepare_data(self, element, n_closest=20):
        """Uses self.mols to create a training set of input samples (self.X) and 
        target values (self.y). 
        
        element: The given element to consider.
        n_closest: the number of closest atoms considered in each sample.
        
        self.X: array-like, shape = [n_samples, n_features]
            A dict of Mol.relative_distances dataframes is extracted from each
            item in self.mols. Each of these dataframes generates a sample for
            self.X.
        
        self.y: array-like, shape = [n_samples]
            Each key to a dataframe used for self.X is used as an atom label to
            acquire (from the Mol) the number of atoms bonded to the central
            atom. These are the target values.
        """
        _X = []
        _y = []
        for id, mol in self.mols.items():
            frameview = mol.element_distances(element)
            for atomlabel, centering in frameview.items():
                _y.append(mol.coordination_num(atomlabel))
                _X.append(self.create_sample(centering, n_closest))
        self.X = np.array(_X)
        self.y = np.array(_y)
        try:
            self.X.resize((self.count,60))
            self.y.resize((self.count))
        except:
            pass
    
    def create_sample(self, frameview, size):
        """Takes a dataframe containing elements and their distances from the 
        central atom, 'frameview.' This frameview is truncated to the given size
        and a new flat list is created of each atom in their periodic table
        representation: 
        (Element, r) = (n, group, r)
        where n is the row and 'group' refers to the column in the periodic
        table. This is accomplished using the lookup table in elementdata.py.
        
        EXAMPLE:
        C 0
        O 1.86  --> [2, 14, 0, 2, 16, 1.86]
        
        The sample length is 3N, where N is the number of atoms.  
        If 3N < 60, the sample is padded with zeros such that len(sample) = 60.
        """
        sample = []
        smallframe = frameview.head(size)
        for row in smallframe.itertuples():
            sample.extend(self.Periodic_Table.loc[row[1],'n':'group']) 
            sample += [row[2]]
        sample += [0] * (3 * size - len(sample))
        return sample
    
