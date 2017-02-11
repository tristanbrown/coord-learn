from ccdc import io
import pandas as pd

csd_reader = io.EntryReader('CSD')

class Molecule():
    """
    """
    def __init__(self, entry):
        self.entry = entry

    def xyz(self):
        mol = csd_reader.molecule(self.entry)
        size = len(mol.atoms)
        coords = []
        for i in range(size):
            atom = mol.atoms[i]
            coords.append([atom.atomic_symbol] + 
                [round(a, 4) for a in atom.coordinates])
            #use atom.label to get the symbol and index number together
        return pd.DataFrame(coords, columns=['Element', 'x', 'y', 'z'])

abebuf = Molecule('ABEBUF')
print(abebuf.xyz())
print(len(csd_reader))

class Molset():
    """
    """
    def __init__(self):
        self.xyzset = {}
        
    def populatexyz(self, count):
        self.xyzset = {csd_reader[m].identifier: 
                        Molecule(csd_reader[m]).xyz() for m in range(count)}

trainset = Molset()
trainset.populatexyz(5)
print(trainset.xyzset)