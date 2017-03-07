from ccdc import io
import pandas as pd
import time

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
    
    def check_3d(self):
        """Checks if all 3D coordinates are available, and returns true if they
        are, false if they are not."""
        for atom in self.atoms:
            if atom.coordinates is None:
                return False
        return True
    
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
        except ValueError:
            pass
    
    def xyz(self):
        """Returns a dataframe of the molecule's atomic coordinates in
        a format similar to .xyz files.
        """
        atoms = self.atoms
        size = len(atoms)
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

A = Mol('AABHTZ')
print(A)
print(A.xyz())

B = Mol(1)
print(B)
print(B.xyz())
print(B.check_3d())
B.center()
print(B.xyz())
print(B.check_3d())
                
class Molset():
    """
    __init__:
    Takes either a list of structure identifier symbols, or an integer value n 
    representing the first n entries in the CSD, and generates a dictionary
    containing a subset of the molecule objects in the CSD. 
    
    """
    def __init__(self, index=[]):
        self.mols = self.populate_mols(index)
        self.center_all()
        self.xyzset = self.populate_xyz()
    
    def populate_mols(self, index):
        try: 
            return {label: csd_reader.molecule(label) for label in index}
        except TypeError:
            return {csd_reader[m].identifier: 
                            mol_reader(m) for m in range(index)}
    
    def xyz(self, mol):
        """Takes a molecule object and returns a dataframe of its atomic coordinates in
        a format similar to .xyz files.
        """
        atoms = mol.atoms
        size = len(atoms)
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
    
    def populate_xyz(self):
        return {id: self.xyz(self.mols[id]) for id in self.mols}
    
    def remove_unlocated(self, mol):
        """Removes all atoms in a molecule that are missing coordinates."""
        for atom in mol.atoms:
            if atom.coordinates is None:
                mol.remove_atom(atom)
    
    def remove_all_unlocated(self):
        for label in self.mols:
            self.remove_unlocated(self.mols[label])
        self.xyzset = self.populate_xyz()
    
    def center_all(self):
        for label in self.mols:
            try:
                mol = self.mols[label]
                moltrunc = mol.copy()
                self.remove_unlocated(moltrunc)
                mol.translate([round(-1 * a, 4)
                                for a in moltrunc.centre_of_geometry()])
            except ValueError:
                pass
    

        
# examples = [csd_reader[i].identifier for i in range(11)]
# print(examples)
                        
# trainset = Molset(['AABHTZ', 'ABEBUF'])
# print(trainset.mols)
# print(trainset.xyzset)
# trainset2 = Molset(10)
# print(trainset2.xyzset)



# #Timing Tests
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
