import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

def Function(points):
    return np.exp(np.tanh(2*np.pi*points))

actual_points  = np.linspace(0,1,1000)
actual_value = Function(actual_points)
points = np.linspace(0,1,100)
#Noise
noise = np.random.normal(0,0.2,100)
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
filename = open("result.txt","a+")

#Polynomial regression 
n_power =[1,2,3,4,5,6,7,8,9,10,11,12]

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

regularized_parameter = [0]
for k in range (len(regularized_parameter)):
    #Matrix  Genration in the form of dataset 
    ERMS_train = []
    ERMS_test = []
    for i in range (len(n_power)):
        Training_set = Training_set.sample(n=30)     #Ramdonly selecting sample 
        Training_set = Training_set.reset_index(drop=True)    
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
        filename.write('\n'+str(weights))
        #Full Matrix
        full_matrix = []
        for j in range(len(dataset)):
            full_mat_row = array_gen(dataset['x'][j],n_power[i])
            full_matrix.append(full_mat_row)
        full_matrix = np.matrix(full_matrix)


        #Full Matrix
        Testing_matrix = []
        for j in range(len(Testing_set)):
            test_mat_row = array_gen(Testing_set.iloc[j,0],n_power[i])
            Testing_matrix.append(test_mat_row)
        Testing_matrix = np.matrix(Testing_matrix)


        #Prediction of points 
        plt.figure(k)
        y_cal = full_matrix.dot(weights.T)
        y_cal_test = Testing_matrix.dot(weights.T)
        plt.plot(Training_set['x'],Training_set['y'], 'bo')  # plot x and y using blue circle markers
        plt.plot(actual_points,actual_value,'-')
        plt.plot(points,y_cal,'r*')
        plt.xlabel('X values')
        plt.ylabel('Y values')
        plt.title('Predicted and Actual Graph for Degree :'+ str(n_power[i]) + 'Regularized Para:' + str(regularized_parameter[k]))
        plt.show()
        plt.savefig('Degree_'+ str(n_power[i] - 1) + '_Regularized Para_' + str(k))
        ##plotting 

         ##Error of whole model 
        error_sum = 0
        for i in range(len(dataset)):
            error_sum += np.square(y_cal[i]- dataset['y'][i])
        RMS_train = np.sqrt(error_sum/len(dataset))
        print('RMS: ', RMS_train)
        ERMS_train.append(RMS_train)

         ##Error of whole model 
        error_sum = 0
        for i in range(len(Testing_set)):
            error_sum += np.square(y_cal[i]- dataset['y'][i])
        RMS_test = np.sqrt(error_sum/len(dataset))
        print('RMS: ', RMS_test)
        ERMS_test.append(RMS_test)

ERMS_test = np.array(ERMS_test)
ERMS_test = ERMS_test.reshape(-1,1)

ERMS_train = np.array(ERMS_train)
ERMS_train = ERMS_train.reshape(-1,1)

# n_power=np.array(n_power)
# n_power.reshape(-1,1)
plt.plot(n_power,ERMS_test, 'bo-',label = 'Testing ERMS')  # plot x and y using blue circle markers
plt.plot(n_power,ERMS_train,'*-',label = 'Training ERMS')
plt.xlabel('Degree')
plt.ylabel('Error RMS')
plt.title('Root Mean vs  Degree') 
plt.savefig('rms')
plt.legend()
plt.show()

filename.close()

        


        




