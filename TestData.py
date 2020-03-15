import wfdb

record = wfdb.rdrecord('/Users/ashwini/Downloads/heartdisease-data/training-a/a0001') 
wfdb.plot_wfdb(record=record, title='Record a103l from Physionet Challenge 2015') 
display(record.__dict__)