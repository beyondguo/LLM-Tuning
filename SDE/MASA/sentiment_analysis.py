import numpy as np
import pandas as pd

def weighted_kappa(confusion_matrix, weight=None):
    '''
    weight: weighted matrix of the weighted kappa score
    '''
    if weight == None:
        weight = np.eye(4)
    confusion_matrix = confusion_matrix.values
    #print(confusion_matrix)
    
    pe_rows = np.sum(confusion_matrix, axis=1)
    #print(pe_rows)
    pe_cols = np.sum(confusion_matrix, axis=0)
    sum_total = sum(pe_cols)
    #print(sum_total)
    
    num_ratings = len(confusion_matrix)
    pe = 0.0
    po = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            po += float(confusion_matrix[i][j])* float(weight[i][j]) / float(sum_total)
            pe += float(pe_rows[i]) * float(pe_cols[j]) * float(weight[i][j]) / (float(sum_total) * float(sum_total))
    
    try:
        kappa=(po - pe) / (1 - pe)
    except ZeroDivisionError:
        kappa=0
    return kappa
    

def get_confusion_matrix(write_f,dict_lines,aspects_list):
    '''
    write_f: the file to save the results; 
    dict_lines: the predictions(after parsers) in Dict format
    '''
    ###Initialization-----------
    #Initializing the Confusion matrix of each Aspect
    con_matrix = {
        'pre-正面': [0, 0, 0, 0],'pre-中性': [0, 0, 0, 0],'pre-负面': [0, 0, 0, 0],'pre-未提及': [0, 0, 0, 0]
    }
    con_matrix = pd.DataFrame(con_matrix, index=['la-正面', 'la-中性', 'la-负面', 'la-未提及'])
    #Storing Confusion matrix of each(and all) Aspect in a dict
    asp_con_dict = {}
    asp_con_dict['ALL'] = con_matrix.copy()
    for l in aspects_list:
        asp_con_dict[l] = con_matrix.copy()

    #Initializing the Accuracy of each Aspect
    acc_dict = {'Correct_Pre':0,'Incorrect_Pre':0,'Acc':0}
    #Storing Accuracy of each(and all) Aspect in a dict
    asp_acc_dict = {}
    asp_acc_dict['sentence'] = acc_dict.copy()#Accuracy of the whole sentences
    asp_acc_dict['ALL'] = acc_dict.copy()
    for l in aspects_list:
        asp_acc_dict[l] = acc_dict.copy()
    
    #Weighted matrix of the weighted kappa score
    kappa_weight_matrix = [[1,0.5,0,0.5], [0.67,1,0.67,0.67], [0,0.5,1,0.5], [0.5,0.67,0.5,1]]

    
    ####Counting-----------------------
    for i in range(len(dict_lines)):
        line = dict_lines[i]
        output = line['prediction']
        label = line['label']
        # Accuracy in whole sentences
        if label == output:
            asp_acc_dict['sentence']['Correct_Pre']+=1
        else:
            asp_acc_dict['sentence']['Incorrect_Pre']+=1
        
        #Accuracy and Confusion matrix in each(and all) aspect
        for asp,la in label.items():#asp: aspect, la: label, pre: prediction
            pre = output[asp]         
            #for Confusion matrix 
            asp_con_dict[asp]['pre-'+pre]['la-'+la]+=1##df[column][row]
            asp_con_dict['ALL']['pre-'+pre]['la-'+la]+=1
            #for Accuracy
            if pre == la:
                asp_acc_dict[asp]['Correct_Pre']+=1
                asp_acc_dict['ALL']['Correct_Pre']+=1
            else:
                asp_acc_dict[asp]['Incorrect_Pre']+=1
                asp_acc_dict['ALL']['Incorrect_Pre']+=1
            
    
    ####Calculating and Saving-----------------------------------
    with open(write_f,'a',encoding='utf-8') as f:
        #for Accuracy
        for asp, acc in asp_acc_dict.items():
            acc['Acc'] = round(acc['Correct_Pre']/(acc['Correct_Pre']+acc['Incorrect_Pre']), 4)
            f.write('Acc_' + asp + ':\n' + str(acc) + '\n')
        acc_all = asp_acc_dict['ALL']['Acc']

        #for Confusion matrix
        #calculating kappa, precision and recall
        kappa_all = 0
        for asp,matrix in asp_con_dict.items():
            f.write('\n')
            f.write('matrix_' + asp + ':\n')
            matrix.to_csv(f, sep='\t', index=True)
            #calculating precision and recall
            la_sum = matrix.sum(axis=1)#row
            pre_sum = matrix.sum(axis=0)#column     
            for i in ['正面','中性','负面','未提及']:
                correct = matrix['pre-'+i]['la-'+i]
                try:
                    precision = round((correct/pre_sum['pre-'+i]),4)
                except ZeroDivisionError:
                    precision = np.nan
                try:
                    recall = round((correct/la_sum.loc['la-'+i]),4)
                except ZeroDivisionError:
                    precision = np.nan
                f.write(i+'_pre_num:'+str(pre_sum['pre-'+i])+'; precision:'+str(precision)+'\n')  
                f.write(i+'_la_num:'+str(la_sum.loc['la-'+i])+'; recall:'+str(recall)+'\n') 
            
            #calculating kappa 
            kappa = round(weighted_kappa(matrix, kappa_weight_matrix),4)
            f.write('kappa:' + str(kappa) + '\n')  
            if asp == 'ALL':
                kappa_all = kappa
            
    return(acc_all,kappa_all)
