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
    
    xyz:
    Takes a molecule object and returns a dataframe of its atomic coordinates in
    a format similar to .xyz files.
    """
    def __init__(self, index=[]):
        self.mols = self.populatemols(index)
        self.xyzset = self.populatexyz()
    
    def populatemols(self, index):
        try: 
            return {label: csd_reader.molecule(label) for label in index}
        except TypeError:
            return {csd_reader[m].identifier: 
                            mol_reader(m) for m in range(index)}
    
    def xyz(self, mol):
        atoms = mol.atoms
        size = len(atoms)
        coords = [[atom.atomic_symbol] + 
                    self.coordformat(atom.coordinates)
                        for atom in atoms]
            #use atom.label to get the symbol and index number together
        return pd.DataFrame(coords, columns=['Element', 'x', 'y', 'z'])
    
    def coordformat(self, coord):
        try:
            return [round(a, 4) for a in coord]
        except TypeError:
            return [None, None, None]
    
    def populatexyz(self):
        return {id: self.xyz(self.mols[id]) for id in self.mols}
    
    def center(self):
        for label in self.mols:
            #try:
                mol = self.mols[label]
                mol.translate([round(-1 * a, 4)
                                for a in mol.centre_of_geometry()])
            # except RuntimeError:
                # pass
        self.xyzset = self.populatexyz()

        
examples = [csd_reader[i].identifier for i in range(11)]
print(examples)
                        
trainset = Molset(['AABHTZ', 'ABEBUF'])
print(trainset.mols)
print(trainset.xyzset)
trainset.center()
print(trainset.xyzset)
start = time.time()
trainset2 = Molset(10)
end = time.time()
time10 = end - start
print(trainset2.xyzset)
trainset2.center()
print(trainset2.xyzset)

# start = time.time()
# trainset2 = Molset(100)
# end = time.time()
# time100 = end - start

# start = time.time()
# trainset2 = Molset(1000)
# end = time.time()
# time1000 = end - start

# print(time10)
# print(time100)
# print(time1000)

