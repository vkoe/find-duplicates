import pandas as pd
import numpy as np
import sklearn as sk
import xgboost as xgb
import fasttext as ft
import jarowinkler as jw
import io

def load_vectors(fname):
    """ Load in English word vectors for fastText (UNUSED FOR NOW)

    Args:
        fname (str): File path of pre-trained word vectors

    Returns:
        data (dict): Loaded text model (?)
    """
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data


def read_data(path:str) -> np.ndarray:
    """ Read in affiliations dataset and compute similarity scores
    
    Args:
        path(str): location of dataset

    Returns:
        np.ndarray: numpy labels array, features array
    
    """
    affs = pd.read_csv(path)
    affs = affs['name'].values
    out = []
    for i, aff1 in enumerate(affs):
        for j, aff2 in enumerate(affs[i+1:]):
            out.append([(aff1, aff2), (i,j)])

    df = pd.DataFrame(out, columns=['pairname', 'pairindex'])

    jwscore = lambda x: jw.jarowinkler_similarity(x[0], x[1])
    df['jwdist'] = df['pairname'].apply(jwscore)

    #df['pairname_vec'] = ft.get_sentence_vector()
    return df


def small_train_set(df: pd.DataFrame) -> pd.DataFrame:
    """ Generate small training set from affiliations

    Args:
        df (pd.DataFrame): Affiliations dataset w/ jw similarity scores

    Returns:
        jwtrain (pd.DataFrame): Small training set
    """
    rng = np.random.default_rng(987123) #using seed for now so hand-labelling stays consistent

    # (NEEDS TO BE CLEANED UP)
    # (roughly) uniformly draw data based on jw score
    jwtrain = pd.DataFrame(rng.choice(df.loc[df['jwdist']>.9], size=5), columns=['pairname', 'pairindex', 'jwdist'])
    for i in range(10):
        if not df.loc[(df['jwdist']<=(i/10))& (df['jwdist']>((i-1)/10))].empty:
            jwtrain = jwtrain.append(pd.DataFrame(rng.choice(df.loc[(df['jwdist']<=(i/10))& (df['jwdist']>((i-1)/10))], size=5), 
                                              columns=['pairname', 'pairindex', 'jwdist']))
    jwtrain = jwtrain.append(pd.DataFrame(rng.choice(df.loc[(df['jwdist']<=.1)], size=5), 
                                              columns=['pairname', 'pairindex', 'jwdist']))
    
    jwtrain.reset_index(drop=True, inplace=True)
    return jwtrain
    

def train(train_data: np.ndarray, train_labels: np.ndarray) -> xgb.XGBClassifier:
    """ Train binary classifier on training features and labels

    Args:
        train_data (np.ndarray): numpy array with training features
        train_labels (np.ndarray): numpy array with training labels

    Returns:
        xgb.XGBClassifier: fitted model

    """
    model = xgb.XGBClassifier()
    model.fit(train_data, train_labels)
    return model


def predict(model: xgb.XGBClassifier, test_data: np.ndarray) -> np.ndarray:
    """ Predict labels from test data

    Args:
        model (xgb.XGBClassifier): previously trained model
        test_data (np.ndarray): numpy array of test features

    Returns:
        model_labels (np.ndarray): numpy array of predicted labels

    """
    model_labels = model.predict(test_data)
    return model_labels


def assess(model_labels: np.ndarray, test_labels: np.ndarray) -> tuple():
    """ Assess binary model (UNUSED FOR NOW)

    Args:
        model_labels (np.ndarray): numpy array of predicted labels from model
        test_labels (np.ndarray): numpy array of test labels

    Returns:
        tuple(): tuple containing floats: accuracy, precision, recall, and F1 score of model

    """
    true_all = np.where(model_labels==test_labels)[0]
    true_pos = np.where(test_labels[true_all]==0)[0]
    false_all = np.where(model_labels!=test_labels)[0]
    false_pos = np.where(test_labels[false_all]==1)[0]
    false_neg = np.where(test_labels[false_all]==0)[0]
    
    accuracy = len(true_all)/len(test_labels)
    precision = len(true_pos)/(len(true_pos) + len(false_pos))
    recall = len(true_pos)/(len(true_pos) + len(false_neg))
    f1score = (2*precision*recall)/(precision+recall)
  
    print('Accuracy:  ', f"{accuracy:.1%}")
    print('Precision: ', f"{precision:.1%}")
    print('Recall:    ', f"{recall:.1%}")
    print('F1 Score:  ', f"{f1score:.1%}")
    return (accuracy, precision, recall, f1score)


if __name__ == '__main__':
    # read in affiliations data, generate small training set
    df = read_data('affiliations.csv')
    dfjwtrain = small_train_set(df)
    
    # hand-label matches from small train set
    dfjwtrain['match'] = 0
    dfjwtrain['match'][0]= 1
    dfjwtrain['match'][1]= 1
    dfjwtrain['match'][2]= 1
    dfjwtrain['match'][3]= 1
    dfjwtrain['match'][4]= 1

    # basic model trained on small sample of data, using jw score as only feat
    base_model = train(np.array(dfjwtrain['jwdist'].values.reshape(len(dfjwtrain),1)), np.asarray(dfjwtrain['match']))

    # predict match using base model (NEED TO REMOVE TRAIN ROWS FROM DF)
    df['predicted_match'] = None
    df['predicted_match'] = predict(base_model, df['jwdist'])