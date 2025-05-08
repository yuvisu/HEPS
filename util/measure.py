import os
import csv
import shap
import pickle
import matplotlib.pyplot as plt
import numpy as np

from util.io import check_saving_path, save_model, load_model
from sklearn.metrics import confusion_matrix, roc_auc_score ,roc_curve,auc,accuracy_score
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score

def calculate_score(score_dict,root_dir, output_dir,model_name,model_id):
    head = score_dict.keys()
    head = [val for val in head for _ in (0, 1)]
    
    for idx, val in enumerate(head): 
        if idx%2!=0: head[idx] = val+"_std"
        else: head[idx] = val+"_mean"

    flatten_result = []
    for key in score_dict:
        flatten_result.append(np.mean(score_dict[key]))
        flatten_result.append(np.std(score_dict[key]))
    
    save_path = check_saving_path(root_dir, output_dir,model_name.split("_")[0],model_name+model_id) 
    
    with open(save_path, 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
        # writing the fields 
        csvwriter.writerow(head) 
        # writing the data rows 
        csvwriter.writerow(flatten_result)
        
def calculate_auroc(model, X_train, X_test, y_train, y_test, root_dir, output_dir,model_name,model_id):
    # predicted value: training error
    pred_logtrain = model.predict_proba(X_train)
    fpr_logtrain, tpr_logtrain, thresh_logtrain = roc_curve(y_train, pred_logtrain[:, 1], pos_label=1)
    auc_logtrain = roc_auc_score(y_train, pred_logtrain[:, 1])
    
    plt.figure()
    plt.plot(fpr_logtrain, tpr_logtrain, linestyle='-', label='Training set AUC (' + str(round(auc_logtrain, 3)) + ')')
    plt.legend()
    plt.xlabel('1 - specificity (FPR)')
    plt.ylabel('Sensitivity (TPR)')

    # predicted: tesing error
    pred_logtest = model.predict_proba(X_test)
    fpr_logtest, tpr_logtest, thresh_logtest = roc_curve(y_test, pred_logtest[:, 1], pos_label=1)
    auc_logtest = roc_auc_score(y_test, pred_logtest[:, 1])
    plt.plot(fpr_logtest, tpr_logtest, linestyle='-', label='Testing set AUC (' + str(round(auc_logtest, 3)) + ')')
    plt.legend()
    plt.xlabel('1 - specificity (FPR)')
    plt.ylabel('Sensitivity (TPR)')
    plt.title('Logistic model training and testing ROC for "hospitalization_flag" - ROC Curve (AUC)')
    
    save_path = check_saving_path(root_dir, output_dir,model_name.split("_")[0],model_name+model_id) 
    plt.savefig(save_path)
    plt.close()

def calculate_shap(model, data, used_variables, root_dir, output_dir,model_name,model_id):
    explainer_log = shap.Explainer(model, data, feature_names = used_variables)
    shap_values_log = explainer_log(data)
    shap.plots.beeswarm(shap_values_log)
    
    shap.summary_plot(shap_values_log, data, max_display=25, feature_names=used_variables, plot_type="bar",show=False)   
    save_path = check_saving_path(root_dir, output_dir,model_name.split("_")[0],model_name+model_id) 
    plt.savefig(save_path)
        
def performance_score(ground_truth, pred, pred_prob):
    matric = confusion_matrix(ground_truth, pred)
    f1 = f1_score(ground_truth, pred, zero_division=0, average='binary')
    auroc = roc_auc_score(ground_truth, pred_prob)
    recall = recall_score(ground_truth, pred, zero_division=0,average='binary')
    precision = precision_score(ground_truth, pred, zero_division=0,average='binary')
    
    tn, fp, fn, tp = confusion_matrix(ground_truth, pred).ravel()
    
    specificity = tn / (tn+fp)
    
    return (round(f1,4),round(auroc,4), round(recall,4), round(precision,4), round(specificity,4), np.array2string(matric))
        
def fariness_score(protected_ground_truth, privileged_ground_truth, protected_pred, privileged_pred, outcome_flag = None):
    print('fairness:', np.unique(protected_ground_truth), np.unique(protected_pred),np.unique(privileged_ground_truth), np.unique(privileged_pred))
    ## confusion matrix
    if len(set(protected_ground_truth)) == 1 or len(set(protected_pred)) == 1:
        if outcome_flag == 'ad_mci_control':
            protected_cm = confusion_matrix(protected_ground_truth, protected_pred, labels=[0, 1, 2])
        else:
            protected_cm = confusion_matrix(protected_ground_truth, protected_pred, labels=[0, 1])  
    else:
        protected_cm = confusion_matrix(protected_ground_truth, protected_pred)

    if len(set(privileged_ground_truth)) == 1 or len(set(privileged_pred)) == 1:
        if outcome_flag == 'ad_mci_control':
            privileged_cm = confusion_matrix(privileged_ground_truth, privileged_pred, labels=[0, 1, 2])
        else:
            privileged_cm = confusion_matrix(privileged_ground_truth, privileged_pred, labels=[0, 1])  
    else:
        privileged_cm = confusion_matrix(privileged_ground_truth, privileged_pred)
    
    # basic model accuracy metrics
    ## protected
    protected_TP = protected_cm[1, 1]
    protected_FP = protected_cm[0, 1]
    protected_TN = protected_cm[0, 0]
    protected_FN = protected_cm[1, 0]
    
    ## privileged
    privileged_TP = privileged_cm[1, 1]
    privileged_FP = privileged_cm[0, 1]
    privileged_TN = privileged_cm[0, 0]
    privileged_FN = privileged_cm[1, 0]
    
    # No.1 Predictive parity: PPV: 0 if inf
    protected_PPV = protected_TP/(protected_TP + protected_FP)
    privileged_PPV = privileged_TP/(privileged_TP + privileged_FP)
    if np.isinf(protected_PPV):
        protected_PPV = 0
    if np.isinf(privileged_PPV):
        privileged_PPV = 0  
          
    # No.2 False positive error rate balance: FPR: 1 if inf
    protected_FPR = protected_FP/(protected_FP + protected_TN)
    privileged_FPR = privileged_FP/(privileged_FP + privileged_TN)
    if np.isinf(protected_FPR):
        protected_FPR = 1
    if np.isinf(privileged_FPR):
        privileged_FPR = 1      
    # No.3 Equalized odds: equal TPR and FPR: 0 if inf
    protected_TPR = protected_TP/(protected_TP + protected_FN)
    privileged_TPR = privileged_TP/(privileged_TP + privileged_FN)
    if np.isinf(protected_TPR):
        protected_TPR = 0
    if np.isinf(privileged_TPR):
        privileged_TPR = 0 

    # No.4 Conditional use accuracy equality: equal PPV, NPV: 0 if inf
    protected_NPV = protected_TN/(protected_TN + protected_FN)
    privileged_NPV = privileged_TN/(privileged_TN + privileged_FN)
    if np.isinf(protected_NPV):
        protected_NPV = 0
    if np.isinf(privileged_NPV):
        privileged_NPV = 0 

    # No.5 Treatment equality: FN/FP:0 if inf
    protected_te = protected_FN/protected_FP
    privileged_te = privileged_FN/privileged_FP
    if np.isinf(protected_te):
        protected_te = 0
    if np.isinf(privileged_te):
        privileged_te = 0 

    # No.6 False negative error rate balance: FNR = FN/(FN + TP):1 if inf
    protected_FNR = protected_FN/(protected_FN + protected_TP)
    privileged_FNR = privileged_FN/(privileged_FN + privileged_TP)
    if np.isinf(protected_FNR):
        protected_FNR = 1
    if np.isinf(privileged_FNR):
        privileged_FNR = 1 

    # No.7 Overall accuracy equality: 0 if inf
    protected_ACC = accuracy_score(protected_ground_truth, protected_pred)
    privileged_ACC = accuracy_score(privileged_ground_truth, privileged_pred)
    if np.isinf(protected_ACC):
        protected_ACC = 0
    if np.isinf(privileged_ACC):
        privileged_ACC = 0 

    return (round(protected_PPV, 4), round(privileged_PPV, 4), 
            round(protected_FPR, 4), round(privileged_FPR, 4), 
            round(protected_TPR, 4), round(privileged_TPR, 4), 
            round(protected_NPV, 4), round(privileged_NPV, 4), 
            round(protected_te, 4), round(privileged_te, 4), 
            round(protected_FNR, 4), round(privileged_FNR, 4),
            round(protected_ACC, 4), round(privileged_ACC, 4), np.array2string(protected_cm), np.array2string(privileged_cm))
'''           
def shap_accumulation(model,X_test)
    explainer = shap.Explainer(model.predict, X_test)
    shap_values = explainer(X_test)
    return shap_values
'''
def get_mpec(op_enc_count_claim, op_enc_count_ehr, ip_enc_count_claim, ip_enc_count_ehr):
    # count mpec
    if op_enc_count_claim != 0 :
        mpec_o = op_enc_count_ehr / op_enc_count_claim
    else:
        mpec_o = -1
    if ip_enc_count_claim != 0:
        mpec_i = ip_enc_count_ehr / ip_enc_count_claim
    else:
        mpec_i = -1
 

    if mpec_o >= 0 and mpec_i >= 0:
        mpec = (mpec_o + mpec_i)/2
    elif op_enc_count_claim > 0 and ip_enc_count_claim <= 0:
        mpec = mpec_o
    elif op_enc_count_claim <= 0 and ip_enc_count_claim > 0:
        mpec = mpec_i
    else:
        mpec = -1
    return mpec_i, mpec_o, mpec