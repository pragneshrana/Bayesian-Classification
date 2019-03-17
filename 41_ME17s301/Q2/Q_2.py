import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib as mlt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Loading Dataset
dataset = pd.read_csv('Dataset_2_Team_41.csv')
column = ['X_1', 'X_2', 'Class_value']
mean_file  = open("mean_file.txt","a+")
varaince_file = open("varaince_file.txt","w+")


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

# Number of class in the dataset
Data_class = dataset.iloc[:, -1].unique()  # classes in dataset
Data_class = np.sort(Data_class)
no_of_class = len(Data_class)  # number of class
no_of_feature = dataset.shape[1]-1  # -1 as last column is outcome

# Number of elemets in same class and minimum elements in the calss
class_sample_count = pd.DataFrame(dataset.Class_label.value_counts())
# min_data_in_class = class_sample_count.min() #Class of each class is not same

# Dataset sepearation of calss based on training set


def dataset_seperation(dataset, Data_class):
    '''
    Seperation of dataset based on class value 
    '''
    class_dataset = pd.DataFrame([])
    for i in range(len(dataset)):
        if (dataset.iloc[:, -1][i] == Data_class):
            class_dataset = class_dataset.append(dataset.iloc[i, :])
    return class_dataset

# Splitting the dataset


def split_dataframe(df, p=[0.7, 0.15, 0.15]):
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
Another_Training_set = Training_set
print('Training_set: ', Training_set)
# Training_set.to_csv("Training_set.csv")
# Validate_set.to_csv("Validate_set.csv")
# Testing_set.to_csv("Testing_set.csv")
number =[100,500,1000,2000,3000]
##############  QUE - 2  #################

for j in range(len(number)):
    filename = open('Accuracy_result_'+str(number[j])+'.csv','w+')
    Ano_Training_set = Another_Training_set.sample(n=number[j])     #Ramdonly selecting sample 
    for i in range(20):
        Training_set = Ano_Training_set.sample(n=20)     #Ramdonly selecting sample 
        def dataset_seperation(dataset, Data_class):
            '''
            Seperation of dataset based on class value 
            '''
            class_dataset = pd.DataFrame([])
            for i in range(len(dataset)):
                if (dataset.iloc[:, -1][i] == Data_class):
                    class_dataset = class_dataset.append(dataset.iloc[i, :])
            return class_dataset

        # Splitting the dataset


        def split_dataframe(df, p=[0.7, 0.15, 0.15]):
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
        # Training_set.to_csv("Training_set.csv")
        # Validate_set.to_csv("Validate_set.csv")
        # Testing_set.to_csv("Testing_set.csv")


        def class_wise_dataset(training_dataset):
            training_dataset = training_dataset.reset_index(
                drop=True)  # reindexing the old dataset
            # print('training_dataset', training_dataset)
            # Class dataset name and training dataset name
            class_dataset_name = []
            class_wise_dataframe = []

            Data_class = training_dataset.iloc[:, -1].unique()  # classes in dataset
            Data_class = np.sort(Data_class)
            # print('Data_class', Data_class)
            no_of_class = len(Data_class)  # number of class

            for i in range(no_of_class):
                class_dataset_name.append('class_dataset' + '_' + str(Data_class[i]))
                class_wise_dataframe.append(
                    'class_wise_dataframe' + '_' + str(Data_class[i]))
                # print('class_wise_dataframe', class_wise_dataframe)
                class_dataset = pd.DataFrame([])
                for j in range(len(training_dataset)):
                    if (training_dataset.iloc[:, -1][j] == Data_class[i]):
                        class_dataset = class_dataset.append(
                            training_dataset.iloc[j, :])
                # class wise dataframe
                # dataset_seperation(training_dataset,Data_class[i])
                class_wise_dataframe[i] = class_dataset
                # print('trining_dataset', class_wise_dataframe[i])
                # print("kk")
            return class_wise_dataframe


        training_class_dataset_name = class_wise_dataset(Training_set)
        # print('training_class_dataset_name', training_class_dataset_name)


        #################################
        # covariance Matrix calculation #
        #################################

        # Model_1
        def cov_calcualtion_1(data_cov):
            '''
            It will calculate cpvariance matrix.
            based on data set you have passed
            it calculates covariance matrix based passing two columns
            '''

            no_of_feature = data_cov.shape[1]
            covariance_matrix = np.identity(no_of_feature)
            return covariance_matrix

        # Model:2


        def cov_calcualtion_2(data_cov):
            '''
            It will calculate cpvariance matrix.
            based on data set you have passed
            it calculates covariance matrix based passing two columns
            '''

            no_of_feature = data_cov.shape[1]
            n = 2  # as pasing two values for correlation

            data = pd.DataFrame([])
            covariance_matrix = np.zeros((no_of_feature, no_of_feature))

            # identity = np.identity(no_of_feature)

            # For(sigma(x,y))
            for p in range(no_of_feature):
                for q in range(p, no_of_feature):  # p as matrix is semi-positive and symm

                    data[0] = data_cov.iloc[:, p]
                    data[1] = data_cov.iloc[:, q]

                    # mean vector
                    mean = []
                    for i in range(n):
                        mean.append(data.iloc[:, i].mean())

                    # minus from mean
                    data_minus_mean = []  # DMM
                    for i in range(len(data)):
                        X = data.iloc[i, :]
                        data_minus_mean.append(X-mean)

                    # multiplication of two column
                    multiply_DMM = []
                    for i in range(len(data)):
                        multi = 1
                        for j in range(n):
                            multi = data_minus_mean[i][j]*multi
                        multiply_DMM.append(multi)

                    # addition
                    summation_feature = []
                    covari = []
                    for i in range(n):
                        summation = 0
                        for j in range(len(data)):
                            summation = summation + multiply_DMM[j]
                        summation_feature.append(summation)
                        covari.append(summation_feature[i]/(len(data)-1))

                    # Adding ans to matrix
                    covariance_matrix[p, q] = covari[0]
                    covariance_matrix[q, p] = covari[1]

            # covariance_matrix = np.multiply(covariance_matrix,identity)
            return covariance_matrix

        # Model:3


        def cov_calcualtion_3(data_cov):
            '''
            It will calculate cpvariance matrix.
            based on data set you have passed
            it calculates covariance matrix based passing two columns
            '''

            no_of_feature = data_cov.shape[1]
            n = 2  # as pasing two values for correlation

            data = pd.DataFrame([])
            covariance_matrix = np.zeros((no_of_feature, no_of_feature))

            # identity = np.identity(no_of_feature)

            # For(sigma(x,y))
            for p in range(no_of_feature):
                for q in range(p, no_of_feature):  # p as matrix is semi-positive and symm

                    data[0] = data_cov.iloc[:, p]
                    data[1] = data_cov.iloc[:, q]

                    # mean vector
                    mean = []
                    for i in range(n):
                        mean.append(data.iloc[:, i].mean())

                    # minus from mean
                    data_minus_mean = []  # DMM
                    for i in range(len(data)):
                        X = data.iloc[i, :]
                        data_minus_mean.append(X-mean)

                    # multiplication of two column
                    multiply_DMM = []
                    for i in range(len(data)):
                        multi = 1
                        for j in range(n):
                            multi = data_minus_mean[i][j]*multi
                        multiply_DMM.append(multi)

                    # addition
                    summation_feature = []
                    covari = []
                    for i in range(n):
                        summation = 0
                        for j in range(len(data)):
                            summation = summation + multiply_DMM[j]
                        summation_feature.append(summation)
                        covari.append(summation_feature[i]/(len(data)-1))

                    # Adding ans to matrix
                    covariance_matrix[p, q] = covari[0]
                    covariance_matrix[q, p] = covari[1]

            # covariance_matrix = np.multiply(covariance_matrix,identity)
            return covariance_matrix

        # Model:4


        def cov_calcualtion_4(data_cov):
            '''
            It will calculate cpvariance matrix.
            based on data set you have passed
            it calculates covariance matrix based passing two columns
            '''

            no_of_feature = data_cov.shape[1]
            n = 2  # as pasing two values for correlation

            data = pd.DataFrame([])
            covariance_matrix = np.zeros((no_of_feature, no_of_feature))

            # For(sigma(x,y))
            for p in range(no_of_feature):
                for q in range(p, no_of_feature):  # p as matrix is semi-positive and symm

                    data[0] = data_cov.iloc[:, p]
                    data[1] = data_cov.iloc[:, q]

                    # mean vector
                    mean = []
                    for i in range(n):
                        mean.append(data.iloc[:, i].mean())

                    # minus from mean
                    data_minus_mean = []  # DMM
                    for i in range(len(data)):
                        X = data.iloc[i, :]
                        data_minus_mean.append(X-mean)

                    # multiplication of two column
                    multiply_DMM = []
                    for i in range(len(data)):
                        multi = 1
                        for j in range(n):
                            multi = data_minus_mean[i][j]*multi
                        multiply_DMM.append(multi)

                    # addition
                    summation_feature = []
                    covari = []
                    for i in range(n):
                        summation = 0
                        for j in range(len(data)):
                            summation = summation + multiply_DMM[j]
                        summation_feature.append(summation)
                        covari.append(summation_feature[i]/(len(data)-1))

                    # Adding ans to matrix
                    covariance_matrix[p, q] = covari[0]
                    covariance_matrix[q, p] = covari[1]
            return covariance_matrix


        # Model:5
        def cov_calcualtion_5(data_cov):
            '''
            It will calculate cpvariance matrix.
            based on data set you have passed
            it calculates covariance matrix by passing two columns
            '''
            varaince_file = open("varaince_file.txt","a+")

            no_of_feature = data_cov.shape[1]
            n = 2  # as pasing two values for correlation

            data = pd.DataFrame([])
            covariance_matrix = np.zeros((no_of_feature, no_of_feature))

            # For(sigma(x,y))
            for p in range(no_of_feature):
                for q in range(p, no_of_feature):  # p as matrix is semi-positive and symm

                    data[0] = data_cov.iloc[:, p]
                    data[1] = data_cov.iloc[:, q]

                    # mean vector
                    mean = []
                    for i in range(n):
                        mean.append(data.iloc[:, i].mean())

                    # minus from mean
                    data_minus_mean = []  # DMM
                    for i in range(len(data)):
                        X = data.iloc[i, :]
                        data_minus_mean.append(X-mean)

                    # multiplication of two column
                    multiply_DMM = []
                    for i in range(len(data)):
                        multi = 1
                        for j in range(n):
                            multi = data_minus_mean[i][j]*multi
                        multiply_DMM.append(multi)

                    # addition
                    summation_feature = []
                    covari = []
                    for i in range(n):
                        summation = 0
                        for j in range(len(data)):
                            summation = summation + multiply_DMM[j]
                        summation_feature.append(summation)
                        covari.append(summation_feature[i]/(len(data)-1))

                    # Adding ans to matrix
                    covariance_matrix[p, q] = covari[0]
                    covariance_matrix[q, p] = covari[1]
            varaince_file.write('\n' + str( covariance_matrix))
            return covariance_matrix


        ########################################
        # Data set mean calculation _classwise#
        ########################################
        def mean_of_class(data_set):
            '''
            caculated mean of each column
            '''

            mean_file  = open("mean_file.txt","a+")
            mean = []
            n = data_set.shape[1]  # no_of_feature

            for i in range(n):
                mean.append(data_set.iloc[:, i].mean())
            mean_file.write('\n' + str(list(mean)))
            return mean


        ######################################
        # Posterior probability of each class#
        ######################################

        def likelihood(class_dataset, data_likelyhood):
            '''
            This fucntion calculated likelyhood density of each datapoints 
            and returns the density vector for each class
            pass arguments as (class based dataset , whole dataset)
            '''

            # print('class_dataset', class_dataset)
            # print('data_likelyhood', data_likelyhood)
            # Number of class in the data_likelyhood
            header_list = list(data_likelyhood)
            if ('Class_label' in header_list):
                data_likelyhood = data_likelyhood.drop(
                    columns=['Class_label'])  # Dropped result colm
            header_list = list(class_dataset)
            if ('Class_label' in header_list):
                class_dataset = class_dataset.drop(
                    columns=['Class_label'])  # Dropped result colm
            no_of_feature = data_likelyhood.shape[1]
            n = no_of_feature
            # covariacne called
            # For Model:1
            # covariance_matrix = np.matrix(cov_calcualtion_1(class_dataset))
            # For Model:2
            # covariance_matrix = np.matrix(cov_calcualtion_2(data_likelyhood))
            # For Model:3
            # covariance_matrix = np.matrix(cov_calcualtion_3(class_dataset))
            # For Model:4
            # covariance_matrix = np.matrix(cov_calcualtion_4(data_likelyhood))
            # For Model:5
            covariance_matrix = np.matrix(cov_calcualtion_5(class_dataset))

            # print('covariance_matrix: ', covariance_matrix)
            inv_covariance_matrix = np.linalg.inv(covariance_matrix)

            # Multivariate gaussain distribution

            cov_matrix_det = np.linalg.det(covariance_matrix)
            # print('cov_matrix_det: ', cov_matrix_det)
            density_function_vector = []

            # mean vector
            mean = mean_of_class(class_dataset)

            for i in range(len(data_likelyhood)):
                X = np.array(data_likelyhood.iloc[i, :])
                a = (X-mean).reshape(n, 1)
                # Gaussiaan Calcualtion
                density_function = (1/(((2*np.pi)**(n/2)) * (np.sqrt(cov_matrix_det)))) * (np.exp((-1/2) * np.transpose(a) * inv_covariance_matrix * a))
                density_function_vector.append(float(density_function))
            # print('density_function_vector: ', density_function_vector)
            return density_function_vector


        def prediction(Training_set, training_class_dataset_name):
            #######################
            # Prior of each class #
            #######################
            Total_sample = len(Training_set)
            # print('Total_sample: ', Total_sample)
            class_prior = []

            # training_class_dataset_name = class_wise_dataset(Training_set)

            for i in range(no_of_class):
                class_prior.append(len(training_class_dataset_name[i])/Total_sample)
            # print('class_prior: ', class_prior)

            # Likelyhood calculation
            class_likelyhood = []
            density_dataset = pd.DataFrame([])
            for i in range(no_of_class):
                density_function_vector = likelihood(
                    training_class_dataset_name[i], Training_set)
                density_dataset["Density_fucntion_class_" +
                                str(i)] = density_function_vector
                # likelyhood into prior
                density_dataset["Density_fucntion_class_"+str(
                    i)] = density_dataset["Density_fucntion_class_"+str(i)].multiply(class_prior[i])

            # Calculation of Evidence
            density_dataset["Evidence"] = np.zeros(
                len(density_dataset))  # Evidence intialized with zeros
            for i in range(no_of_class):
                density_dataset["Evidence"] = density_dataset["Density_fucntion_class_" +
                                                            str(i)] + density_dataset["Evidence"]
                # print('density_dataset["Evidence"]: ', density_dataset["Evidence"])

            posterior_dataset = pd.DataFrame([])
            # Divide likelyhood into prior by evidence
            for i in range(no_of_class):
                posterior_dataset["Density_fucntion_class_"+str(
                    i)] = density_dataset["Density_fucntion_class_"+str(i)] / density_dataset["Evidence"]

            # print('density_dataset: ', density_dataset)

            ########################
            # loss Matrix Defined  #
            ########################
            l = np.matrix([[0, 1, 2],
                        [1, 0, 1],
                        [2, 1, 0]])
            # print('l: ', l)

            #############################
            # loss function calculation #
            #############################

            # True_count = 0
            # for i in range(len(dataset)):
            #     alpha_i = int(dataset["Class_label"][i])
            #     alpha_k = int(Predicted_class_label[i])
            #     LQ_righT = 0
            #     LQ_left = 0
            #     for j in range(no_of_class):
            #         left_condition  = l.iloc[alpha_i,j] * posterior_dataset["Density_fucntion_class_"+str(j)]
            #         right_condition = l.iloc[alpha_k,j] * posterior_dataset["Density_fucntion_class_"+str(j)]
            #         LQ_left += left_condition
            #         LQ_righT += right_condition
            #     if (LQ_left <= LQ_righT):
            #         True_count +=1

            # class Prediction
            posterior_probability = []
            Predicted_class_label = []
            for i in range(len(density_dataset)):
                posterior_array = density_dataset.iloc[i, :3]
                # loss_func and posterior array multiplication (1*3)(3*3)
                loss_func_array = posterior_array.dot(l)
                min_val = min(loss_func_array)
                posterior_probability.append(min_val)
                index_max = list(loss_func_array).index(min_val)
                print(index_max)
                Predicted_class_label.append(index_max)
            # print('posterior_probability', posterior_probability)

            return Predicted_class_label


        ####################
        # Confusion Matrix #
        ####################
        def confusion_matrix(actual_result, predicted_result):
            '''
            Plots the confusion matrix. For that you have to pass the two array actual result(given classes) and
            Predicted result 
            '''
            import pandas as pd
            import numpy as np
            classes_name = list(set(actual_result))
            no_of_classes = len(classes_name)
            conf_matrix = pd.DataFrame(np.zeros((no_of_classes, no_of_classes)))
            combined_array_dataset = pd.DataFrame()
            combined_array_dataset[0] = actual_result
            combined_array_dataset[1] = predicted_result
            for i in range(no_of_classes):
                subdataset = combined_array_dataset[combined_array_dataset[0]
                                                    == classes_name[i]]
                # print('subdataset: ', subdataset)
                val = subdataset[1].value_counts()
                val.sort_index(inplace=True)  # sorting values by index
                # print('val: ', val.shape)
                indexList = val.index
                # print('indexList: ', indexList)

                if (len(indexList) == no_of_classes):
                    # for j in range(no_of_classes):
                        # If there is no mis classification
                        if(indexList[j] == j):
                            # print('j: ', j)
                            # print('indexList[j]', indexList[j])
                            try:
                                conf_matrix.iloc[i, j] = val.iloc[j]
                            except IndexError:
                                conf_matrix.iloc[i, j] = 0
                            # print('conf_matrix: ', conf_matrix)
                        else:
                            conf_matrix.iloc[i, j] = 0
                            # print('conf_matrix: ', conf_matrix)
                else:
                    # In case if class has less misclassification than total class
                    m = 0
                    for j in range(no_of_classes):
                        # If there is no mis classification
                        try:
                            if(indexList[m] == j):
                                try:
                                    conf_matrix.iloc[i, j] = val.iloc[m]
                                except IndexError:
                                    conf_matrix.iloc[i, j] = 0
                                # print('conf_matrix: ', conf_matrix)
                                m += 1
                            else:
                                continue
                        except IndexError:
                            continue
            # print('conf_matrix: ', conf_matrix)
            return conf_matrix


        def accuracy(Predicted_class_label, Actual_class_label):
            # print("Predicted_class_label", Predicted_class_label)
            # print("Actual_class_label", Actual_class_label)
            # Acuuracy Check
            matched_count = 0
            not_matched_count = 0

            for i in range(len(Predicted_class_label)):
                if (Predicted_class_label[i] == Actual_class_label[i]):
                    matched_count += 1
                else:
                    not_matched_count += 1

            Accuracy = matched_count / (matched_count + not_matched_count)
            # print('Accuracy: ', Accuracy)
            return Accuracy


        Actual_class_label = Testing_set["Class_label"]
        Actual_class_label = Actual_class_label.reset_index(
            drop=True)  # reindexing the old dataset

        Predicted_class_label = prediction(Testing_set, training_class_dataset_name)
        Accuracy_data = accuracy(Predicted_class_label, Actual_class_label)
        filename.write(str(Accuracy_data) + '\n')
        print('testing')
        Training_set = Training_set[0:0]
    filename.close()

std_devi = []
mean_data = [] 
for j in range(len(number)):
    # print(number[j])
    # print('Accuracy_result_'+str(number[j])+'.csv')
    Accuracy_data = pd.read_csv('Accuracy_result_'+str(number[j])+'.csv')
    std_devi.append(Accuracy_data.stack().std() )
    mean_data.append(Accuracy_data.stack().mean() )

x_pos = np.arange(len(number))
CTEs = mean_data
error = std_devi

# Build the plot
fig, ax = plt.subplots()
ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Accuracy ')
ax.set_xticks(x_pos)
ax.set_title('Dataset Size')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.savefig('error_bars.png')
plt.show()


           