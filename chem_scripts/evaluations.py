
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, auc, mean_squared_error
from itertools import combinations



def softmax(array):
    exp = np.exp(array)
    totals = np.sum(exp, axis=1)
    for i in range(len(exp)):
        exp[i, :] /= totals[i]
    return exp



def cs_auc(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    return auc



def cs_multiclass_auc(y_true, y_pred):
    n = y_pred.shape[1]
    auc_dict = dict()
    for pair in combinations(range(n), 2):
        subset = [i for i in range(len(y_true)) if 1 in [y_true[i, pair[0]], y_true[i, pair[1]]]]
        y_true_temp = y_true[subset]
        y_pred_temp = y_pred[subset]
        y_pred_temp = y_pred_temp[:, [pair[0], pair[1]]]
        y_pred_temp = softmax(y_pred_temp)
        auc_dict[pair] = roc_auc_score(y_true_temp[:, pair[1]], y_pred_temp[:, 1])
    total = 0.0
    for key in auc_dict.keys():
        total += auc_dict[key]
    total /= len(list(combinations(range(n), 2)))
    return total


def cs_compute_results(model, classes=None, train_data=None, valid_data=None, test_data=None, df_out=None, channel='engA'):
    
    # Evaluate results on training set
    X_tmp = train_data[0]
    y_tmp = train_data[1]    
    loss_train = model.evaluate(X_tmp, y_tmp, verbose=0)
    
    if classes == 1:
        rmse_train = np.sqrt(loss_train)
    elif classes == 2:
        y_preds_train = model.predict(X_tmp)
        auc_train = cs_auc(y_tmp, y_preds_train)
    elif classes > 2:
        y_preds_train = model.predict(X_tmp)
        auc_train = cs_multiclass_auc(y_tmp, y_preds_train)
    else:
        raise(Exception('Error in determine problem type'))
        
    # Evaluate results on validation set    
    X_tmp = valid_data[0]
    y_tmp = valid_data[1]
    loss_valid = model.evaluate(X_tmp, y_tmp, verbose=0)
    if classes == 1:
        rmse_valid = np.sqrt(loss_valid)
    elif classes == 2:
        y_preds_valid = model.predict(X_tmp)
        auc_valid = cs_auc(y_tmp, y_preds_valid)
    elif classes > 2:
        y_preds_valid = model.predict(X_tmp)
        auc_valid = cs_multiclass_auc(y_tmp, y_preds_valid)
    else:
        raise(Exception('Error in determine problem type'))
    
    # Evaluate results on test set
    X_tmp = test_data[0]
    y_tmp = test_data[1]    
    loss_test = model.evaluate(X_tmp, y_tmp, verbose=0)
    if classes == 1:
        rmse_test = np.sqrt(loss_test)
	#Nathaniel added 
        Y_pred_test = model.predict(X_tmp)
        print('class==1')
        cs_parity_plot(y_tmp, Y_pred_test, channel);
    elif classes == 2:
        y_preds_test = model.predict(X_tmp)
        auc_test = cs_auc(y_tmp, y_preds_test)
        cs_plot_auc(y_tmp, y_preds_test, classes)
    elif classes > 2:
        y_preds_test = model.predict(X_tmp)
        auc_test = cs_multiclass_auc(y_tmp, y_preds_test)
        cs_plot_auc(y_tmp, y_preds_test, classes)
    else:
        raise(Exception('Error in determine problem type'))
    
    if classes == 1:
        print("\nFINAL TRA_LOSS: %.3f"%(loss_train[0]))
        print("FINAL VAL_LOSS: %.3f"%(loss_valid[0]))
        print("FINAL TST_LOSS: %.3f"%(loss_test[0]))
        print("FINAL TRA_RMSE: %.3f"%(rmse_train[0]))
        print("FINAL VAL_RMSE: %.3f"%(rmse_valid[0]))
        print("FINAL TST_RMSE: %.3f"%(rmse_test[0]))
        df_out.loc[len(df_out)] = [loss_train[0], loss_valid[0], loss_test[0], rmse_train[0], rmse_valid[0], rmse_test[0]]
    else:
        print("\nFINAL TRA_LOSS: %.3f"%(loss_train))
        print("FINAL VAL_LOSS: %.3f"%(loss_valid))
        print("FINAL TST_LOSS: %.3f"%(loss_test))
        print("FINAL TRA_AUC: %.3f"%(auc_train))
        print("FINAL VAL_AUC: %.3f"%(auc_valid))
        print("FINAL TST_AUC: %.3f"%(auc_test))
        df_out.loc[len(df_out)] = [loss_train, loss_valid, loss_test, auc_train, auc_valid, auc_test]

def cs_keras_to_seaborn(history):
    tmp_frame = pd.DataFrame(history.history)
    keys = list(history.history.keys())
    features = [x for x in keys if "val_" not in x and "val_" + x in keys]
    cols = ['epoch', 'phase'] + features
    output_df = pd.DataFrame(columns=cols)
    epoch = 1
    for i in range(len(tmp_frame)):
        new_row = [epoch, 'train'] + [tmp_frame.loc[i, f] for f in features]
        output_df.loc[len(output_df)] = new_row
        new_row = [epoch, 'validation'] + [tmp_frame.loc[i, "val_" + f] for f in features]
        output_df.loc[len(output_df)] = new_row
        epoch += 1
    return output_df



def cs_make_plots(hist_df,hist,channel, filename=None):
    print('plot made '+channel)
    plt.style.use('fivethirtyeight')
    print(hist.history.keys())
    # summarize history for accuracy
    plt.plot(hist.history['mean_squared_error'])
    plt.plot(hist.history['val_mean_squared_error'])
    plt.title('model mse '+channel)
    plt.ylabel('mean squared error')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.tight_layout()
    plt.savefig('mean square error plot '+channel)
    plt.gcf().clear()

    #summarize history for loss
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss '+channel)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.tight_layout()
    plt.savefig('model loss plot '+channel)

def cs_parity_plot(x,y,channel):
    plt.gcf().clear()
    plt.style.use('fivethirtyeight')
    plt.scatter(x,y, label='degree Celsius',color='#30a2da')
    plt.plot([100,500],[100,500],lw=2, label='best fit', color='#fc4f30')
    plt.ylim(100,500)
    plt.xlim(100, 500)
    plt.title('Carroll Data '+channel)
    plt.legend(loc='lower right')
    plt.ylabel('Predicted (degrees Kelvin)')
    plt.xlabel('Experimental (degrees Kelvin)')
    plt.tight_layout()
    plt.savefig('parity plot '+channel)

def cs_plot_auc(y_test, y_score, classes):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange',
        lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.savefig("roc curve");

def cs_plot_cm(cm, classes, 
               normalize=False,
               title='Confusion Matrix',
               cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion matrix')

# inputs dataFrame, outputs mean std
def data_stats(dataFrame, property):
   mean = dataFrame[property].mean()
   std = dataFrame[property].std()
   return mean, std 
