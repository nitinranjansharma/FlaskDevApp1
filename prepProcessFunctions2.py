import pandas as pd
import pickle
def identify_col_types_manual(ls):
    with open('ls.pickle', 'wb') as handle:
        pickle.dump(ls, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return("Success")
    