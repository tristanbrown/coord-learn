from moldata import *

## Tests of Mol

A = Mol('AABHTZ')

# print(A)
# print(A.xyz)

# B = Mol(1)
# print(B)
# print(B.xyz)
# print(A.all_atoms_have_sites)
# print(B.all_atoms_have_sites)
# B.center()
# print(B.xyz)
# print(B.all_atoms_have_sites)
# print(B.find_elements('C'))
# print(B.find_elements('N'))
# print(B.atom('N1').coordinates)
# print(B.atom('N1').neighbours)
# print([len(B.atom(label).neighbours) for label in B.find_elements('O')])
# print(B.element_distances('N'))
# print(A.element_count1('C'))
# print(A.element_count2('C'))

## Tests of Molset
    
# print(elementdata.Element_Table_Periodic)       
# examples = [csd_reader[i].identifier for i in range(11)]
# print(examples)
                        
# trainset = Molset(['AABHTZ', 'ABEBUF'])
# print(trainset.mols)
# print(trainset.xyzset)
# trainset2 = Molset([10])
# print(trainset2.xyzset)
# trainset3 = Molset(100, 'N', 20)
# print(trainset3.xyzset)
# print(len(trainset3.xyzset))
#trainset3.prepare_data('N', 20)
# print(trainset3.X)
# print(trainset3.y)
# print([(len(trainset3.X), len(trainset3.X[0])), len(trainset3.y)])

# print(trainset3.X.shape[1])


################################################################################
# #Timing Tests
# import time
# import timeit
# import cProfile

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
# trainset10 = Molset(100, 'Cu')
# end = time.time()
# time1 = end - start


# start = time.time()
# trainset100 = Molset(100, 'C', 2)
# end = time.time()
# time2 = end - start


# trainset1000 = Molset(1000)
# start = time.time()
# trainset1000.prepare_data('N', 20)
# end = time.time()
# time1000 = end - start

# print(time1)
# print(time2)
# print(time1000)

# cProfile.run('Molset(1000)')
# cProfile.run("trainset3.prepare_data('N', 20)")