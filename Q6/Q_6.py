import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

def Function(points):
    return np.exp(np.tanh(2*np.pi*points))

actual_points  = np.linspace(0,1,1000)
actual_value = Function(actual_points)
points = np.linspace(0,1,1000)
#Noise
noise = np.random.normal(0,0.2,1000)
### Function assigned  exp(tanh(2πx))
y = Function(points)
y = y+noise

dataset = pd.DataFrame([])
dataset['x'] = points
dataset['y'] = y

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
n_power = [2,6,10]

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

    
regularized_parameter = [0.001,0.01,0.1]
for k in range (len(regularized_parameter)):
    #Matrix  Genration in the form of dataset 
    for i in range (len(n_power)):
        Error_store = []
        for m in range(1000):
            Training_set = Training_set.sample(n=10)     #Ramdonly selecting sample 
            Training_set = Training_set.reset_index(drop=True)   
            Testing_set = Testing_set.reset_index(drop=True)    
    
            data_len = int(len(Training_set))
            no_of_columns = n_power[i]+1
            #Traing Matrix
            A = []
            for j in range (len(Training_set)): #-1 
                training_mat_row = array_gen(Training_set['x'][j],n_power[i])
                A.append(training_mat_row)
            A = np.matrix(A)

            
            #Solving Matrix and calculating weights 
            A_Trans = (np.transpose(A))
            A_T_A = np.matmul(A_Trans,A)
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
        

            #Full Matrix
            Testing_matrix = []
            for j in range(len(Testing_set)):
                test_mat_row = array_gen(Testing_set['x'][j],n_power[i])
                Testing_matrix.append(test_mat_row)
            Testing_matrix = np.matrix(Testing_matrix)

            # Prediction of points 
            y_cal = full_matrix.dot(weights.T)
            y_cal_test = Testing_matrix.dot(weights.T)
            fig = plt.figure(0)
            plt.plot(Training_set['x'],Training_set['y'], 'bo')  # plot x and y using blue circle markers
            plt.plot(actual_points,actual_value,'-')
            plt.plot(Testing_set['x'],y_cal_test, 'r*')  # plot x and y using blue circle markers
            plt.show()            
            # Error for testing 
            error_sum = 0
            for j in range(len(y_cal_test)):
                error_sum += np.square(y_cal_test[j]- Testing_set['y'][j])
            Error = (error_sum/len(Testing_set))
            Error_store.append(Error)
        Error_store = np.array(Error_store)
        Error_store = Error_store.reshape(-1,1)
        print('Error_store: ', Error_store)
        print('Error_store: ', Error_store)
        # Generate data on commute times.
        # Fixing random state for reproducibility
        # Fixing random state for reproducibility
        np.random.seed(19680801)

        N_points = 100000
        n_bins = 13

        # Generate a normal distribution, center at x=0 and y=5
        x = Error_store

        fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)

        # We can set the number of bins with the `bins` kwarg
        axs.hist(x, bins=n_bins)
        plt.xlabel('Count')
        plt.ylabel('Probability')
        plt.savefig(str(n_power[i]-1)+'_'+str(regularized_parameter[k])+'.png')
        # plt.show()

   
print(A)

        


        



