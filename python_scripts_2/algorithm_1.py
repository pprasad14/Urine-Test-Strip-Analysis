import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

def Look_up_table(look_up_dict,pad_name,corrected_value,corrected_dict):
    #print(pad_name,corrected_value)
    pad_dict=look_up_dict[pad_name]
    sorted_by_value = sorted(pad_dict.items(), key=lambda kv: kv[1], reverse=True)
    for x in sorted_by_value:
        if (corrected_value>x[1]) :
            pad_value=x[0]
            break;
    else :
        pad_value=sorted_by_value[len(sorted_by_value)-1][0]
    corrected_dict[pad_name]=pad_value


def get__values(corrected_dict, pad_name, channel, cc1, cc2,cc3,diff,color_checker,pad,look_up_dict) :
    X = np.array([color_checker.loc[cc1, channel], color_checker.loc[cc2, channel],
                             color_checker.loc[cc3, channel]])
    Y = np.array([diff.loc[cc1, channel], diff.loc[cc2, channel], diff.loc[cc3, channel]])
    X = X.reshape(-1, 1)
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    a = regr.coef_
    b = regr.intercept_
    test = np.array([pad.loc[pad_name, channel]])

    p_diff = a * test[0] +b

    corrected_value=test+p_diff

    Look_up_table(look_up_dict, pad_name,corrected_value,corrected_dict)
    # print(pad_name, " ", channel, 'value is', test[0], 'difference is ', p_diff[0], " corrected_value is ",
          # corrected_value[0],'look_up_value is ',look_up_dict[pad_name])
