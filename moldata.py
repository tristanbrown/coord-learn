from ccdc import io
import pandas as pd
import numpy as np

np.random.seed(901)
csd_reader = io.EntryReader('CSD')

def mol_reader(n):
    """Calls the nth entry in the CSD as a molecule object."""
    entry = csd_reader[n]
    return csd_reader.molecule(entry.identifier)

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
            return csd_reader.molecule(index)
        except NotImplementedError:
            return mol_reader(index)
    
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
    
    def relative_distances(self, label):
        """Takes an atom label and gives a sorted dataframe containing the
        distance of every element in the molecule to that atom."""
        atoms = self.atoms
        central = self.atom(label)
        distances = [[atom.atomic_symbol]
                 + [self.atom_distance(atom, central)] for atom in self.atoms] 
        return pd.DataFrame(distances,
                                columns=['Element', 'r']).sort_values('r')#.head(20)
    
    def atom_distance(self, atom1, atom2):
        """Returns the distance of the atom from the given point in space."""
        point1 = atom1.coordinates
        point2 = atom2.coordinates
        ssd = [(x1 - x2)**2 for x1, x2 in zip(point1, point2)]
        return round((sum(ssd)**(0.5)), 4)
        
        

A = Mol('AABHTZ')
print(A)
print(A.xyz())

B = Mol(1)
print(B)
print(B.xyz())
print(A.all_atoms_have_sites)
print(B.all_atoms_have_sites)
B.center()
print(B.xyz())
print(B.all_atoms_have_sites)
print(B.find_elements('C'))
print(B.find_elements('N'))
print(B.atom('N1').coordinates)
print(B.atom('N1').neighbours)
print([len(B.atom(label).neighbours) for label in B.find_elements('O')])
print(B.element_distances('N'))

                
class Molset():
    """
    __init__:
    Takes either a list of structure identifier symbols, or an integer value n 
    representing the first n entries in the CSD, and generates a dictionary
    containing a subset of the molecule objects in the CSD. 
    
    """
    def __init__(self, ids=[]):
        self.mols = self.populate_mols(ids)
        self.center_all()
        self.xyzset = self.populate_xyz()
        #self.xyzset = self.centered_xyz()
    
    def populate_mols(self, ids):
        """Populates self.mols using a list of string identifiers, or a list of
        numerical indices. If instead a number n is given directly, n fully-3D
        entries are chosen from the CSD to populate self.mols."""
        try:
            mols = {}
            for id in ids:
                amol = Mol(id)
                mols[amol.identifier] = amol
        except TypeError:
            mols = self.random_populate(ids)
        return mols
    
    def random_populate(self, count):
        mols = {}
        while len(mols) < count:
            id = np.random.randint(0, len(csd_reader))
            amol = Mol(id)
            if amol.all_atoms_have_sites:
                mols[amol.identifier] = amol
        return mols
    
    def center_all(self):
        """Use to re-center all Mols."""
        return [mol.center() for id, mol in self.mols.items()]
            
    
    def populate_xyz(self):
        return {id: self.mols[id].xyz() for id in self.mols}
    
    def centered_xyz(self):
        {id: mol.xyz() for id, mol in self.mols.items() if mol.center()}
    
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
        self.X = []
        self.y = []
        for id, mol in self.mols.items():
            distances = mol.element_distances(element)
            for atomlabel, centering in distances.items():
                self.y.append(atomlabel)
                self.X.append(centering)
    
        
# examples = [csd_reader[i].identifier for i in range(11)]
# print(examples)
                        
# trainset = Molset(['AABHTZ', 'ABEBUF'])
# print(trainset.mols)
# print(trainset.xyzset)
trainset2 = Molset([10])
print(trainset2.xyzset)
trainset3 = Molset(10)
print(trainset3.xyzset)
print(len(trainset3.xyzset))
# trainset3.prepare_data('C', 20)
# print(trainset3.X)
# print(trainset3.y)



################################################################################
# #Timing Tests
# import time
# import timeit

# testlist = [1, 2, 3, 4, 5]

# def array_from_list(alist):
    # deeplist = [[[x, x**2], [x**3, x**4]] for x in alist]
    # return np.array(deeplist).reshape(-1, 2)

# print(array_from_list(testlist))
# time1 = timeit.timeit('array_from_list([1, 2, 3, 4, 5])',
                        # "from __main__ import array_from_list", number=10000)


# def array_from_arrays(alist):
    # container = np.array([0, 0])
    # for x in alist:
        # container = np.vstack((container, np.array([[x, x**2], [x**3, x**4]])))
    # return container[1:]

# print(array_from_arrays(testlist))
# time2 = timeit.timeit('array_from_arrays([1, 2, 3, 4, 5])', 
                        # "from __main__ import array_from_arrays", number=10000)

# def array_from_listappend(alist):
    # deeplist = []
    # for x in alist:
        # deeplist.append([x, x**2])
        # deeplist.append([x**3, x**4])
    # return np.array(deeplist)

# time3 = timeit.timeit('array_from_listappend([1, 2, 3, 4, 5])', 
                    # "from __main__ import array_from_listappend", number=10000)

# print(time1)
# print(time2)
# print(time3)

# start = time.time()
# trainset10 = Molset(10)
# end = time.time()
# time10 = end - start

# start = time.time()
# trainset100 = Molset(100)
# end = time.time()
# time100 = end - start

# start = time.time()
# trainset1000 = Molset(1000)
# end = time.time()
# time1000 = end - start

# print(time10)
# print(time100)
# print(time1000)
