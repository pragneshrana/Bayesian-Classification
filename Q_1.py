import pandas as pd 
import numpy as np 
import matplotlib as mlt 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#Loading Dataset
dataset = pd.read_csv('Dataset_1_Team_41.csv')
column = ['X_1','X_2','Class_value']

# #Plotting data points
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# z = dataset['Class_label']
# x1 = dataset['x_1']
# x2 = dataset['x_2']
# ax.scatter(x1, x2, z, c='r', marker='o')
# ax.set_xlabel('X1')
# ax.set_ylabel('X2')
# ax.set_zlabel('Y')
# plt.show()

#Number of class in the dataset
Data_class = dataset.iloc[:,-1].unique() #class in dataset 
Data_class = np.sort(Data_class)
no_of_class = len(Data_class)       #number of class
no_of_feature = dataset.shape[1]-1      #-1 as last column is outcome

#Number of elemets in same class and minimum elements in the calss
class_sample_count  = pd.DataFrame(dataset.Class_label.value_counts())
min_data_in_class = class_sample_count.min() #Class of each class is not same

#Dataset sepearation of calss based on training set 
def dataset_seperation(dataset,Data_class):
    class_dataset= pd.DataFrame([])
    for i in range(len(dataset)):
        if (dataset.iloc[:,-1][i]==Data_class):
            class_dataset = class_dataset.append(dataset.iloc[i,:])
    return class_dataset

#Splitting the dataset
def split_dataframe(df, p=[0.7,0.15,0.15]):
    train, validate, test = np.split(df.sample(frac=1), [int(.7*min_data_in_class), int(.85*min_data_in_class)]) #split by [0-.7,.7-.85,.85-1]
    return train,validate,test

##################################################
# #####Generating Seperate Dataset ###############
##################################################

# list of datsset names                          
class_dataset_name = []
training_class_dataset_name=[]
validation_class_dataset_name=[]
testing_class_dataset_name=[]
Training_set = pd.DataFrame([])
Testing_set = pd.DataFrame([])
Validate_set = pd.DataFrame([])

for i in range(no_of_class):
    #Creating dataset 
    class_dataset_name.append('class_dataset'+ '_'+ str(Data_class[i]))
    training_class_dataset_name.append('training_class_dataset'+ '_'+ str(Data_class[i]))
    validation_class_dataset_name.append('testing_class_dataset'+ '_'+ str(Data_class[i]))
    testing_class_dataset_name.append('validation_class_dataset'+ '_'+ str(Data_class[i]))
    class_dataset_name[i]=pd.DataFrame([])
    #class wise dataframe
    data = dataset_seperation(dataset,Data_class[i])
    class_dataset_name[i]= pd.DataFrame(data)
    #Training Testing validation 
    training_class_dataset_name[i],validation_class_dataset_name[i],testing_class_dataset_name[i] = split_dataframe(class_dataset_name[i])
    training_class_dataset_name[i].to_csv("training_class_dataset_name[i]_"+str(i)+".csv")    
    Training_set = Training_set.append(training_class_dataset_name[i])
    Validate_set = Validate_set.append(validation_class_dataset_name[i])
    Testing_set = Testing_set.append(testing_class_dataset_name[i])

Training_set.to_csv("Training_set.csv")
Validate_set.to_csv("Validate_set.csv")
Testing_set.to_csv("Testing_set.csv")

# print('Testing_set: ', Testing_set.shape,Validate_set.shape,Training_set.shape)


#######################
# Prior of each class #
#######################

Total_sample = len(Training_set)
print('Total_sample: ', Total_sample)
class_prior = []

for i in range(no_of_class):
    class_prior.append(len(training_class_dataset_name[i])/Total_sample)
print('class_prior: ', class_prior)


#################################
# covariance Matrix calculation #
#################################
def cov_calcualtion(data_cov):
   
    no_of_feature = data_cov.shape[1]
    n=2 #as pasing two values for correlation 
    print('n: ', n)

    data=pd.DataFrame([])
    covariance_matrix = np.zeros((no_of_feature,no_of_feature))

    #For(sigma(x,y))
    for p in range(no_of_feature):
        for q in range(p,no_of_feature):    #p as matrix is semi-positive and symm 

            data[0]=data_cov.iloc[:,p]
            data[1]=data_cov.iloc[:,q]  

            #mean vector
            mean = []
            for i in range(n):
                mean.append(data.iloc[:,i].mean())
            
            #minus from mean
            data_minus_mean = []    #DMM
            for i in range(len(data)):
                X = data.iloc[i,:]
                data_minus_mean.append(X-mean)
            
            #multiplication of two column
            multiply_DMM = []
            for i in range(len(data)):
                multi= 1
                for j in range (n):
                    multi = data_minus_mean[i][j]*multi
                multiply_DMM.append(multi)

            #addition
            summation_feature = []
            covari = []
            for i in range(n):
                summation = 0
                for j in range (len(data)):
                    summation = summation + multiply_DMM[j]
                summation_feature.append(summation)
                covari.append(summation_feature[i]/(len(data)-1))

            #Adding ans to matrix
            covariance_matrix[p,q] = covari[0]
            covariance_matrix[q,p] = covari[1]

    return covariance_matrix

########################################
# Data set mean calculation _classwise#
########################################
def mean_of_class(data_set):
    mean = []
    n = data_set.shape[1]        #no_of_feature
    
    for i in range(n):
        mean.append(data_set.iloc[:,i].mean())
    return mean


######################################
# Posterior probability of each class#
######################################

def likelihood(data_likelhood):
    #Number of class in the data_likelhood
    no_of_feature = data_likelhood.shape[1]-1      #-1 as last column is outcome
    n=no_of_feature
    data_likelhood = data_likelhood.drop(columns=['Class_label']) #Dropped result colm

    #covariacne called
    covariance_matrix = np.matrix(cov_calcualtion(data_likelhood))
    inv_covariance_matrix = np.linalg.inv(covariance_matrix)
    print('covariance_matrix: ', covariance_matrix)

    #Multivariate gaussain distribution

    cov_matrix_det = np.linalg.det(covariance_matrix)
    density_function_vector=[]
    
    #mean vector
    mean = mean_of_class(data_likelhood)

    for i in range(len(data_likelhood)):
        X=np.array(data_likelhood.iloc[i,:])
        a = (X-mean).reshape(n,1)
        density_function = (1/( (( 2*np.pi)**(n/2) )* (np.sqrt(cov_matrix_det))) )* (np.exp((-1/2) * np.transpose(a) * inv_covariance_matrix * a ))
        density_function_vector.append(float(density_function))
    return density_function_vector
    
#Likelyhood calculation

class_likelyhood = []
for i in range(no_of_class):
    density_function_vector = likelihood(training_class_dataset_name[i])
    training_class_dataset_name[i]["Density_fucntion"] = density_function_vector
    #maximum out of all vector 
    max_likelyhood = max(density_function_vector)
    class_likelyhood.append(max_likelyhood)
    print('class_likelyhood: ', class_likelyhood)
    
    
#Calculationm of Evidence
Evidence = 0 #as all are of same size
for i in range(no_of_class):
    Evidence = Evidence + class_likelyhood[i]

print('Evidence: ', Evidence)
class_posterior = []
print('class_prior: ', class_prior)

#posterior 
for i in range(no_of_class):
    class_posterior.append((class_likelyhood[i]*class_prior[i])/ Evidence)
    print('class_posterior: ', class_posterior)

#DataValidation

# def DataValidation(feture_data_point):
#     #Number of class in the data_likelhood
#     no_of_feature = data_likelhood.shape[1]-1      #-1 as last column is outcome
#     n=no_of_feature
#     data_likelhood = data_likelhood.drop(columns=['Class_label']) #Dropped result colm

#     #covariacne called
#     covariance_matrix = np.matrix(cov_calcualtion(data_likelhood))
#     inv_covariance_matrix = np.linalg.inv(covariance_matrix)

#     #Multivariate gaussain distribution

#     cov_matrix_det = np.linalg.det(covariance_matrix)
#     density_function_vector=[]
#      #mean vector
#     mean = []
#     for i in range(n):
#         mean.append(data_likelhood.iloc[:,i].mean())

#     for i in range(len(data_likelhood)):
#         X=np.array(data_likelhood.iloc[i,:])
#         a = (X-mean).reshape(n,1)
#         density_function = (1/( (( 2*np.pi)**(n/2) )* (np.sqrt(cov_matrix_det))) )* (np.exp((-1/2) * np.transpose(a) * inv_covariance_matrix * a ))
#         density_function_vector.append(float(density_function))
#     return density_function_vector

# for i in range(len(dataset)):
#     dataset_validation = 




    






