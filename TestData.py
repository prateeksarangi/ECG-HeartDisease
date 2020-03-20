import wfdb

record = wfdb.rdrecord('/Users/ashwini/ECG-HeartDisease/data/patient001/s0010_re') 
wfdb.plot_wfdb(record=record, title='Record a103l from Physionet Challenge 2015') 
display(record.__dict__)