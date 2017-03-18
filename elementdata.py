from __future__ import division
import pandas as pd
#REWRITE THIS AS AN ATOM CLASS

periodic = pd.read_csv('element_data.csv', delimiter=',', header=0, index_col=0)
print(periodic)

# class Element():
    # """Lookup table for different representations of an element."""
    
    # def __init__(self, symbol):
        # self.symbol = symbol


# Element_Table_Periodic = {
    # 'H': (1, 1),
    # 'D': (1, 1),
    # 'He': (1, 18),
    # 'Li': (2, 1),
    # 'Be': (2, 2),
    # 'B': (2, 13),
    # 'C': (2, 14),
    # 'N': (2, 15),
    # 'O': (2, 16),
    # 'F': (2, 17),
    # 'Ne': (2, 18),
    # 'Na': (3, 1),
    # 'Mg': (3, 12),
    # 'Al': (3, 13),
    # 'Si': (3, 14),
    # 'P': (3, 15),
    # 'S': (3, 16),
    # 'Cl': (3, 17),
    # 'Ar': (3, 18),
    # 'K': (4, 1),
    # 'Ca': (4, 2),
    # 'Sc': (4, 3),
    # 'Ti': (4, 4),
    # 'V': (4, 5),
    # 'Cr': (4, 6),
    # 'Mn': (4, 7),
    # 'Fe': (4, 8),
    # 'Co': (4, 9),
    # 'Ni': (4, 10),
    # 'Cu': (4, 11),
    # 'Zn': (4, 12),
    # 'Ga': (4, 13),
    # 'Ge': (4, 14),
    # 'As': (4, 15),
    # 'Se': (4, 16),
    # 'Br': (4, 17),
    # 'Kr': (4, 18),
    # 'Rb': (5, 1),
    # 'Sr': (5, 2),
    # 'Y': (5, 3),
    # 'Zr': (5, 4),
    # 'Nb': (5, 5),
    # 'Mo': (5, 6),
    # 'Tc': (5, 7),
    # 'Ru': (5, 8),
    # 'Rh': (5, 9),
    # 'Pd': (5, 10),
    # 'Ag': (5, 11),
    # 'Cd': (5, 12),
    # 'In': (5, 13),
    # 'Sn': (5, 14),
    # 'Sb': (5, 15),
    # 'Te': (5, 16),
    # 'I': (5, 17),
    # 'Xe': (5, 18),
    # 'Cs': (6, 1),
    # 'Ba': (6, 2),
    # 'La': (6, 3),
    # #Lanthanides added programmatically below.
    # 'Hf': (6, 4),
    # 'Ta': (6, 5),
    # 'W': (6, 6),
    # 'Re': (6, 7),
    # 'Os': (6, 8),
    # 'Ir': (6, 9),
    # 'Pt': (6, 10),
    # 'Au': (6, 11),
    # 'Hg': (6, 12),
    # 'Tl': (6, 13),
    # 'Pb': (6, 14),
    # 'Bi': (6, 15),
    # 'Po': (6, 16),
    # 'At': (6, 17),
    # 'Rn': (6, 18),
    # 'Fr': (7, 1),
    # 'Ra': (7, 2),
    # 'Ac': (7, 3)
    # #Actinides added programmatically below.
    # }

# # Giving fractional values to 
# lanth = ['Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 
            # 'Tm', 'Yb', 'Lu']
# act = ['Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
            # 'No', 'Lr']
            
# for i in range(14):
    # n = i + 1
    # Element_Table_Periodic[lanth[i]] = (6, 3 + n / 15)
    # Element_Table_Periodic[act[i]] = (7, 3 + n / 15)

# # def periodic():
    # # """Takes an atomic symbol and returns the representation of that element
    # # as its row and column in the periodic table."""
    # # return pd.read_csv('element_data.csv', delimiter=',', header=0, index_col=0)