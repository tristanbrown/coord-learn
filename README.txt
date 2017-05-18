README for coord-learn

Installation and Dependencies
    Install CSD Software.
    Python dependencies are listed in (and, if using Anaconda, installed using)
        the environment.yml file.

Usage
    The _harvest#.py scripts are examples of how to search the CSD for entries
        containing the specified elements, and save those entries' labels as
        sample sets.
            CSD -> entries as .npy files ("samples")
        
    _train04.py is an example of how to use the harvested entries to generate
        data sets and train a Perceptron to identify atom coordination numbers.
        This Perceptron is then evaluated on a test data set, determining the
        accuracy of the Perceptron on each element. 
            samples -> .csv file containing the accuracies and # samples for 
            each element
        
    The _plotacc**.py scripts are examples of how to plot the accuracy data
        generated for each element. 