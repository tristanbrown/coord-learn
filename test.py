import pandas as pd

print(pd.__file__)

from ccdc import io

csd_reader = io.EntryReader('CSD')
entry_abebuf = csd_reader.entry('ABEBUF')
cryst_abebuf = csd_reader.crystal('ABEBUF')
mol_abebuf = csd_reader.molecule('ABEBUF')

print(round(mol_abebuf.molecular_weight, 3))

reader_formats = io.MoleculeReader.known_formats.keys()
reader_formats.sort()
for format in reader_formats:
    print format

first_molecule = csd_reader[0]
print first_molecule.identifier
ababub = csd_reader.entry('ABABUB')
mol = ababub.molecule
print len(mol.atoms)
print mol.formula

for i in range(11):
    mol = csd_reader[i]
    print mol.identifier

mol = csd_reader.molecule('ABEBUF')
size = len(mol.atoms)

def listcoords(entry):
    mol = csd_reader.molecule(entry)
    size = len(mol.atoms)
    structure = []
    for i in range(size):
        atom = mol.atoms[i]
        structure.append([atom.atomic_symbol] + [round(a, 4) for a in atom.coordinates])
        #use atom.label to get the symbol and index number together
    return pd.DataFrame(structure, columns=['Element', 'X', 'Y', 'Z'])

print(listcoords('ABEBUF'))