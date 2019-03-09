data =  importdata('file_class.csv')
col1= data(:,1)
col2= data(:,2)

covari = cov(col1,col2)