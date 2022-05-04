import pandas as pd
import argparse
import json

def getFuncMakeArray(iDict):

    def makeArray(word):
        l = []
        for c in word:
            l.append(iDict[c])
        return l

    return makeArray

addDelimeters = lambda x:'\n'+x+'\t'

def getDictOfChar(series):
    outputSet = set()

    for word in series:
        for char in word:

            if char not in outputSet:
                outputSet.add(char)

    return {c:i for i,c in enumerate(sorted(list(outputSet)))}

def main(lg):

    if not lg:
        lg='hi'

    train_tsv = 'dakshina_dataset_v1.0/'+lg+'/lexicons/'+lg+'.translit.sampled.train.tsv'
    valid_tsv = 'dakshina_dataset_v1.0/'+lg+'/lexicons/'+lg+'.translit.sampled.dev.tsv'
    test_tsv = 'dakshina_dataset_v1.0/'+lg+'/lexicons/'+lg+'.translit.sampled.test.tsv'

    train_tsv = pd.read_csv(train_tsv,sep='\t',names=[lg,'en','c']).dropna()
    valid_tsv = pd.read_csv(valid_tsv,sep='\t',names=[lg,'en','c']).dropna()
    test_tsv = pd.read_csv(test_tsv,sep='\t',names=[lg,'en','c']).dropna()


    en = 'en'

    train_tsv[lg] = train_tsv[lg].apply(addDelimeters)
    train_tsv[en] = train_tsv[en].apply(addDelimeters)
    valid_tsv[lg] = valid_tsv[lg].apply(addDelimeters)
    valid_tsv[en] = valid_tsv[en].apply(addDelimeters)
    test_tsv[lg] = test_tsv[lg].apply(addDelimeters)
    test_tsv[en] = test_tsv[en].apply(addDelimeters)

    dictOfLgChar = getDictOfChar(train_tsv[lg])
    dictOfEnChar = getDictOfChar(train_tsv[en])

    train_tsv[lg] = train_tsv[lg].apply(getFuncMakeArray(dictOfLgChar))
    train_tsv[en] = train_tsv[en].apply(getFuncMakeArray(dictOfEnChar))

    valid_tsv[lg] = valid_tsv[lg].apply(getFuncMakeArray(dictOfLgChar))
    valid_tsv[en] = valid_tsv[en].apply(getFuncMakeArray(dictOfEnChar))

    test_tsv[lg] = test_tsv[lg].apply(getFuncMakeArray(dictOfLgChar))
    test_tsv[en] = test_tsv[en].apply(getFuncMakeArray(dictOfEnChar))

    train_tsv.to_csv('train.csv',index=False)
    valid_tsv.to_csv('valid.csv',index=False)
    test_tsv.to_csv('test.csv',index=False)

    with open('dict.json','w') as f:
        json.dump({en:dictOfEnChar,lg:dictOfLgChar},f)


if __name__=='__main__':

    #Parse cmd args
    parser = argparse.ArgumentParser(description='Data Preparation')
    parser.add_argument('--lg', dest='lg', type=str, help='Language to be used')
    main(parser.parse_args().lg)