import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets


def regression_module(dataset,data):
    ###Model from Q_6 
    def split_dataframe(df, p=[0.7, 0.20, 0.10]):
    
        '''
        It will split the dataset into training, validation and testing set based on the given fractin value
        of p. By default p=[0.7,0.15,0.15]
        '''

        train, validate, test = np.split(df.sample(frac=1), [int(
            p[0]*len(df)), int((p[0]+p[1])*len(df))])  # split by [0-.7,.7-.85,.85-1]
        return train, validate, test

    ##################################################
    # #####Generating Seperate Dataset ###############
    ##################################################

    # list of datsset names


    Training_set = pd.DataFrame([])
    Testing_set = pd.DataFrame([])
    Validate_set = pd.DataFrame([])


    # Training Testing validation BY SPLIT FUNCTION
    Training_set, Validate_set, Testing_set = split_dataframe(dataset)


    #Polynomial regression 
    n_power = [2]

    #Array generation method
    def array_gen(entry,power):
        '''
        Generated the each row of the 
        matrix so you can append and 
        generate full matrix
        '''
        entry_row = []
        for i in range(power):
            entry_row.append(entry**i)
        entry_row = np.array(entry_row)
        return entry_row

    #Regularized Parameeter
    regularized_parameter = [-100,-1,-.5,0.01,0.05,0.1,1,10,100,1000]
    for k in range (len(regularized_parameter)):
        #Matrix  Genration in the form of dataset 
        for i in range (len(n_power)):
            Error_store = []
            # for m in range(1000):
            Training_set = Training_set.reset_index(drop=True)   
            Testing_set = Testing_set.reset_index(drop=True)    

            data_len = int(len(Training_set))
            no_of_columns = n_power[i]+1
            #Traing Matrix
            A = []
            for j in range (len(Training_set)): #-1 
                print('Training_set: ', Training_set)
                training_mat_row = array_gen(Training_set['x'][j],n_power[i])
                A.append(training_mat_row)
            A = np.matrix(A)
            print('A: ', A)

            
            #Solving Matrix and calculating weights 
            A_Trans = (np.transpose(A))
            A_T_A = np.matmul(A_Trans,A)
            print('A_T_A: ', A_T_A)
            I = np.identity(no_of_columns-1)
            regularizer =I.dot(2*regularized_parameter[k])
            A_T_A_inv = np.linalg.inv(np.add(A_T_A,regularizer))
            A_T_A_inv_A_T = np.matmul(A_T_A_inv,A_Trans)
            weights = A_T_A_inv_A_T.dot(Training_set['y'])


            #Full Matrix
            full_matrix = []
            for j in range(len(dataset)):
                full_mat_row = array_gen(dataset['x'][j],n_power[i])
                full_matrix.append(full_mat_row)
            full_matrix = np.matrix(full_matrix)


            #Train Matrix
            Train_matrix = []
            for j in range(len(Training_set)):
                full_mat_row = array_gen(Training_set['x'][j],n_power[i])
                Train_matrix.append(full_mat_row)
            Train_matrix = np.matrix(Train_matrix)
        

            #Full Matrix
            Testing_matrix = []
            for j in range(len(Testing_set)):
                test_mat_row = array_gen(Testing_set['x'][j],n_power[i])
                Testing_matrix.append(test_mat_row)
            Testing_matrix = np.matrix(Testing_matrix)

            #Prediction of points 
            y_cal = full_matrix.dot(weights.T)
            y_cal_train = Train_matrix.dot(weights.T)
            y_cal_test = Testing_matrix.dot(weights.T)

                ##Error of whole model 
            error_sum = 0
            for i in range(len(Training_set)):
                error_sum += np.square(y_cal[i]- Training_set['y'][i])
            RMS_train = np.sqrt(error_sum/len(Training_set))
            

                ##Error of whole model 
            error_sum = 0
            for i in range(len(Testing_set)):
                error_sum += np.square(y_cal_test[i]- Testing_set['y'][i])
            RMS_test = np.sqrt(error_sum/len(Testing_set))

            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.scatter(data.iloc[:,0],data.iloc[:,1],y_cal, marker = '.', s=10, color='r',label='regression output')
            ax.scatter(data.iloc[:,0],data.iloc[:,1],data.iloc[:,2], marker = '.', s=10, color='b',label='target output')
            #plt.scatter(DataX,DataY, marker = 'o', s=10, facecolors='none', edgecolors='k', label = 'data')
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            #plt.title('Degree of polynomial: '+str(d))
            legend = plt.legend(loc='upper right', shadow=False, fontsize='x-small')
            legend.get_frame()#.set_facecolor('C0')
            fig.savefig('fig_'+str(regularized_parameter[k])+'_.png')
            filename = open('result.csv','a+')
            filename.write('\n Regularized '+''+str(regularized_parameter[k])+''+' Train '+str(RMS_train)+' Test'+''+str(RMS_test))

          
#######
# Q_7 #
#######
data = pd.read_csv('train100.txt', sep=" ", header=None)

#to remove last column 
Dataset = data
Dataset = Dataset.drop([2],axis =1)
print('Dataset: ', Dataset)

#kmeans
kmeans = KMeans(n_clusters=5, random_state=0).fit(Dataset)
Mean_values = kmeans.cluster_centers_
print('Mean_values: ', Mean_values)


#Gaussian Function 
sigma = [20]

basis_array = []
x_mu_j = []

#To try out diff values of sigma
for k in range(len(sigma)):
    #to run over all the entries 
    for i in range(len(Dataset)):
        X_entry = np.array(Dataset.iloc[i,:])

        #to find out second norm 
        for j in range(len(Mean_values)):
            x_mu_j.append(X_entry - Mean_values[j])
        # print('x_mu_j: ', x_mu_j)
        
        norm = np.linalg.norm(x_mu_j, ord=2, axis=None, keepdims=False)
        # print('norm: ', norm)

        basis_value = np.exp(np.square(norm)/(2*np.square(sigma[k])))
        # print('basis_value: ', basis_value)
        basis_array.append(basis_value)
    dataset = pd.DataFrame([])
    dataset['x'] = basis_array 
    dataset['y'] = data[2]


    regression_module(dataset,data) #passing data to plot original y 

    

