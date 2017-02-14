from ccdc import io
import pandas as pd
import time

csd_reader = io.EntryReader('CSD')

def mol_reader(n):
    """Calls the nth entry in the CSD as a molecule object."""
    entry = csd_reader[n]
    return csd_reader.molecule(entry.identifier)
    
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

        
examples = [csd_reader[i].identifier for i in range(11)]
print(examples)
                        
trainset = Molset(['AABHTZ', 'ABEBUF'])
print(trainset.mols)
print(trainset.xyzset)
trainset2 = Molset(10)
print(trainset2.xyzset)


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
