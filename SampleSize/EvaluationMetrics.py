# This file defines the evaluation metrics
#%%
import pandas as pd
import numpy as np
import copy
from scipy import stats
from sklearn.metrics import r2_score
from minepy import MINE

#%% 
# The funtion needs 5 arguments as inputs
def EvaluationMetrics(Y_test, X_test_raw, Y_test_raw, yhat_test, yhat_test_raw):

    # #######################################################################################################
    # R2: Statistical goodness of fit of all outputs
    R2 = r2_score(Y_test, yhat_test)
    # print("R2 test:", R2)


    ######################################################################################################
    # Consistency of bivariate relationships: N-Quant and N-leaching
    # Coefficient between crop price and relative revenue in true data
    x = X_test_raw['nMin_Quant__mean'] 
    y = Y_test_raw['leachN_Quant__mean']

    # # calculate MIC between x and y 
    mine = MINE(alpha = 0.6, c = 15)
    mine.compute_score(x, y)
    mic_true = mine.mic()
    print('true MIC is', mic_true)

    # Now calculate for the predicted data
    # Coefficient between crop price and relative revenue in predicted data
    x = X_test_raw['nMin_Quant__mean'] 
    y = yhat_test_raw['leachN_Quant__mean']

    # # calculate MIC between x and y of predicted data
    mine = MINE(alpha = 0.6, c = 15)
    mine.compute_score(x, y)
    mic_pred = mine.mic()
    print('predicted MIC is', mic_pred)

    # Calcultae the abselute percentage error (APE)
    APE = abs(mic_true - mic_pred)/abs(mic_true)
    print("APE:", APE)



    #######################################################################################################
    # A1: accuracy in capturing corner solution of crop production
    crops = ['WinterWheat', 'SummerCere', 'WinterRape', 'Sugarbeet', 'WinterBarley', 'MaizCorn']

    sum_accuracy = 0

    All_CornerSolution = {}

    for crop in crops: 

        production = crop +'_levl__mean'

        # load true data
        Y = copy.deepcopy(Y_test_raw[production])
            
        # assign binary values to true production 
        Y.loc[Y > 0.01] = 1
    
        # load predicted data
        yhat = copy.deepcopy(yhat_test_raw[production])

        # set a threshold for predicted data
        yhat.loc[yhat> 0.01] = 1

        # accuracy of acorner solution of one crop
        accuracy =  K.get_value(tf.keras.metrics.binary_accuracy(Y, yhat))
        
        All_CornerSolution[crop] = accuracy 

        # print(accuracy)
        sum_accuracy += accuracy

    # print(All_CornerSolution)

    df = pd.DataFrame.from_dict(All_CornerSolution, orient="index")
    df.to_excel('All_CornerSolution.xlsx')


    mean_accuracy = sum_accuracy/len(crops)
    A1 = mean_accuracy
    print("mean accuracy of corner solutions:", A1)



    #######################################################################################################
    ## constraints of farm size
    farm_size = X_test_raw['totLand_arab__mean']  
    # print('farm size: ', farm_size)

    sum_crop_area = yhat_test_raw['WinterWheat_levl__mean'] + yhat_test_raw['SummerCere_levl__mean'] + yhat_test_raw['WinterRape_levl__mean'] + \
                yhat_test_raw['Sugarbeet_levl__mean'] + yhat_test_raw['WinterBarley_levl__mean'] + yhat_test_raw['MaizCorn_levl__mean']
                # yhat_test_raw['Idle_levl__mean']

                # idel is missing, must update the results

    # print('sum of crop areas: ', sum_crop_area)
    
    diff = sum_crop_area - farm_size

    threshold = 0

    correct_rows = diff[diff <= threshold].count()
    # Average accuracy of constraints holding
    A2 = correct_rows/len(diff)
    print(A2)

    # Find the index of the rows that do not hold the constraint
    index_list = diff.index[diff > threshold].tolist()
    # print(len(index_list))

    # # Mean error when contraint is violated
    # deviation = diff/farm_size
    # # print(deviation)
    # deviation_vio = deviation.loc[index_list]
    # # print(deviation_vio)
    # # Calculate the mean of the deviation
    # Mean_deviation = deviation_vio.mean()
    # print('Mean_deviation:', Mean_deviation)
  
    #####################################################################################################
    # Summarize all results into the EvaluationMetrics_dic
    EvaluationMetrics_dic = {'R2': R2, 'APE': APE, 'A1': A1, 'A2': A2} 
    
    print(EvaluationMetrics_dic)

    df = pd.DataFrame.from_dict(EvaluationMetrics_dic, orient="index")
    df.to_excel('EvaluationMetrics_dic.xlsx')

    return EvaluationMetrics_dic

