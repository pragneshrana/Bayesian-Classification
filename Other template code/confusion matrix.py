####################
# Confusion Matrix #
####################
def confusion_matrix(actual_result,predicted_result):
    '''
    Plots the confusion matrix. For that you have to pass the two array actual result(given classes) and
    Predicted result 
    '''
    import pandas as pd
    import numpy as np
    classes_name =  list (set (actual_result))
    no_of_classes = len(classes_name)
    conf_matrix = pd.DataFrame(np.zeros((no_of_classes, no_of_classes)))
    combined_array_dataset = pd.DataFrame()
    combined_array_dataset [0] = actual_result
    combined_array_dataset [1] = predicted_result
    for i in range(no_of_classes):
        subdataset = combined_array_dataset[combined_array_dataset [0] == classes_name[i]]
        print('subdataset: ', subdataset)
        val = subdataset[1].value_counts()
        val.sort_index(inplace=True)        #sorting values by index
        print('val: ', val.shape)
        indexList = val.index
        print('indexList: ', indexList)
            
        if (len(indexList) == no_of_classes):
            for j in range(no_of_classes):
                #If there is no mis classification
                if(indexList[j] == j):
                    print('j: ', j)
                    print('indexList[j]',indexList[j])
                    try:
                        conf_matrix.iloc[i,j] = val.iloc[j]
                    except IndexError:
                        conf_matrix.iloc[i,j] = 0
                    print('conf_matrix: ', conf_matrix)
                else:
                    conf_matrix.iloc[i,j] = 0
                    print('conf_matrix: ', conf_matrix)
        else:
            # In case if class has less misclassification than total class
            m=0
            for j in range(no_of_classes):
                #If there is no mis classification
                try:
                    if(indexList[m] == j):
                        try:
                            conf_matrix.iloc[i,j] = val.iloc[m]
                        except IndexError:
                            conf_matrix.iloc[i,j] = 0
                        print('conf_matrix: ', conf_matrix)
                        m+=1
                    else:
                        continue  
                except IndexError:
                    continue    
    print('conf_matrix: ', conf_matrix)


if __name__ == "__main__":  
    actual_result =   [0,1,2,0,1,0,2,0,1,2,0,1,2,0,1,0]
    predicted_result =[0,1,1,0,1,0,2,1,2,2,1,0,2,0,1,2]
    confusion_matrix(actual_result,predicted_result)
