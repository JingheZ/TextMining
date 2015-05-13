"""
Created on Sat Feb  7 15:37:01 2015
Description: CS6501 Text Mining 
MP1_Part1 Vector Space Model: get the basic idea of text processing,
e.g., tokenization, stemming, and normalization, construct vector space 
representation for text documents, TF/IDF weighting, and compute similarity 
among different text documents;
@author: Jinghe Zhang
"""
import os
import json
import nltk
#import numpy as np
import pickle
#from sklearn.metrics.pairwise import cosine_similarity 
#import multiprocessing as mp
from collections import Counter

tokenizer = nltk.tokenize.RegexpTokenizer(r'\d|\w+')
stemmer = nltk.stem.PorterStemmer()

''' Read json files '''
def readJson(jpath, s):
    reviews = {}
    ttfList = []
    dfList = []
    ttfs = Counter()
    dfs = Counter()  
    j = 0
    for f in os.listdir(jpath):
        fpath = os.path.join(jpath, f)
        if os.path.isfile(fpath):
            jfile = open(fpath).read()
            jsondata = json.loads(jfile)
            try:
                for k in range(len(jsondata['Reviews'])):
                    try:
                        reviews[s+str(j)] = jsondata['Reviews'][k]['Content']
                        tokens = tokenizer.tokenize(reviews[s+str(j)])
                        stemmedTokens = []
                        stemmedTokens_df = []
                        for t in tokens:
                            if t.isdigit():
                                t = "NUM"
                            else:
                                t = t            
                            stemmedToken = stemmer.stem(t.lower())
                            stemmedTokens.append(stemmedToken)
                        stemmedTokens_df = list(set(stemmedTokens))
                        ttfList += stemmedTokens
                        dfList += stemmedTokens_df
                        j += 1
                    except ValueError:
                        print 'Cannot find Review Content!'
            except ValueError:
                print 'Cannot find Review!'
    ttfs = Counter(ttfList)
    dfs = Counter(dfList)
    return ttfs, dfs
    

trainfolder = r'./Yelp_small0/train/'  
ttfTrain, dfTrain = readJson(trainfolder, 'train')
testfolder = r'./Yelp_small0/test/'  
ttfTest, dfTest = readJson(testfolder, 'test')

ttfAll = ttfTrain + ttfTest
dfAll = dfTrain + dfTest

ttfs_sorted = ttfAll.most_common()
ttfsVal = []
for i in range(len(ttfs_sorted)):
    ttfsVal.append(ttfs_sorted[i][1])

dfs_sorted = dfAll.most_common()
dfsVal = []
for i in range(len(dfs_sorted)):
    dfsVal.append(dfs_sorted[i][1])

def dotplot(freq, ylabl):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    
    x = np.log(np.arange(1, len(freq)+1))
    y = np.log(np.asarray(freq))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    r_squared = r_value**2
    p1 = plt.plot(x, y, 'b.', label = str(ylabl))
    lx = np.log(np.arange(1,len(freq)+1, 0.01))
    ly = intercept + slope * lx
    p2 = plt.plot(lx, ly, 'r-', label = 'LinearRegression')
    plt.xlabel('Log(Token Rank)')
    plt.ylabel('Log('+str(ylabl)+')')
    plt.legend(loc = 'upper right')
    plt.savefig(str(ylabl)+'_plot'+'.png')
    plt.show()
    
    return slope, intercept, r_squared

slopeTTF, interceptTTF, r2TTF = dotplot(ttfsVal, 'TTF')   
slopeDF, interceptDF, r2DF = dotplot(dfsVal, 'DF')   


'''Part 1.2'''

import os
import json
import nltk
import numpy as np
#import pickle
from sklearn.metrics.pairwise import cosine_similarity 
from collections import Counter
import csv
from operator import itemgetter   

tokenizer = nltk.tokenize.RegexpTokenizer(r'\d|\w+')
stemmer = nltk.stem.PorterStemmer()

''' Read json files '''
def readJson2(jpath, s):

    i = 0
    dfs = Counter()
    reviews = {}
    j = 0
    allgrams = []
    for f in os.listdir(jpath):
        fpath = os.path.join(jpath, f)
        if os.path.isfile(fpath):
            jfile = open(fpath).read()
            jsondata = json.loads(jfile)
            try:
                for k in range(len(jsondata['Reviews'])):
                    try:
                        reviews[s+str(j)] = jsondata['Reviews'][k]['Content']
                        i += 1
                        tokens = tokenizer.tokenize(reviews[s+str(j)])
                        stemmedTokens = []

                        bigram = []
                        for t in range(len(tokens)):
                            try:
                                tk = int(tokens[t])
                                tk = "NUM"
                            except ValueError:
                                tk = tokens[t]            
                            stemmedToken = stemmer.stem(tk.lower())
                            stemmedTokens.append(stemmedToken)
                        for m in range(len(stemmedTokens)-1):
                            bigram.append(stemmedTokens[m] + '-' + stemmedTokens[m+1])                        
                        unibigram_df = list(set(stemmedTokens + bigram))
#                        c1 = Counter(unibigram_df)
#                        dfs = dfs + c1
                        allgrams += unibigram_df
                        j += 1
                    except ValueError:
                        print 'Cannot find Review Content!'
            except ValueError:
                print 'Cannot find Review!'
    dfs = Counter(allgrams)
    return dfs, i

trainfolder = r'./Yelp_small0/train/'  
dfsVocb, numReview = readJson2(trainfolder, 'train')
   
    
'''Sort on DFs'''
with open('dfsVocb.pickle') as fi:
    dfsVocb = pickle.load(fi)
    
    
dfsVocbSorted = dfsVocb.most_common()
dfsVocbSortedVal = []
dfsVocbSortedTerm = []
for i in range(len(dfsVocbSorted)):
    dfsVocbSortedVal.append(dfsVocbSorted[i][1])    
    dfsVocbSortedTerm.append(dfsVocbSorted[i][0])    

topNgrams = dfsVocbSortedTerm[:100]  
        
stopwords = []
stopwordfile = 'english.stop.txt'
with open(stopwordfile, 'r') as f:
    for line in f:
        stopwords.append(line.split('\n')[0])
f.close()

'''Compare initial stopword list with the specific stopwords'''
swCommon = set(stopwords).intersection(set(topNgrams))
swDiff = set(stopwords).difference(set(topNgrams))
swDiff2 = set(topNgrams).difference(set(stopwords))
swNew = set(stopwords).union(set(topNgrams))
print 'SwCommon:'
print swCommon

file1 = open('swCommon.csv', 'wb')
writer = csv.writer(file1)
for row in swCommon:
    writer.writerow(row)
file1.close()    
    
print 'swDiff:'
print swDiff


file2 = open('swDiff.csv', 'wb')
writer = csv.writer(file2)
for row in swDiff:
    writer.writerow(row)
file2.close()    


print 'swDiff2:'
print swDiff2


print 'swDiff2:'
print swDiff2

file3 = open('swDiff2.csv', 'wb')
writer = csv.writer(file3)
for row in swDiff2:
    writer.writerow(row)
file3.close()    
#numReview = 629921


'''Merge stopwords, remove rare words, contruct controlled vocabulary'''
stopwordFinal = list(set(topNgrams + stopwords))
rareIndex = dfsVocbSortedVal.index(49)
ctrlVocb = dfsVocbSortedTerm[100:rareIndex]
ctrlVocbDF = dfsVocbSortedVal[100:rareIndex]
sizeCtrlVocb = len(ctrlVocb)
ctrlVocbIDF = 1 + np.log(numReview/np.asarray(ctrlVocbDF))

ctrlVocbDict = {}
for i in range(len(ctrlVocb)):
    ctrlVocbDict[ctrlVocb[i]] = ctrlVocbIDF[i]

print 'Get final controlled Vocb!'

top50Tokens = ctrlVocb[:50]
top50IDF = 1 + np.log(sizeCtrlVocb/np.asarray(ctrlVocbDF[:50]))
top50 = zip(top50Tokens, top50IDF)

print 'Top 50 Ngrams as follows:'
print top50
    
bottom50Tokens = ctrlVocb[-50:]
bottom50IDF = 1 + np.log(sizeCtrlVocb/np.asarray(ctrlVocbDF[-50:]))
bottom50 = zip(bottom50Tokens, bottom50IDF)

print 'Bottom 50 Ngrams as follows:'
print bottom50

'''Sort on DFs'''
dfsVocbSorted = dfsVocb.most_common()
dfsVocbSortedVal = []
dfsVocbSortedTerm = []
for i in range(len(dfsVocbSorted)):
    dfsVocbSortedVal.append(dfsVocbSorted[i][1])    
    dfsVocbSortedTerm.append(dfsVocbSorted[i][0])    

topNgrams = dfsVocbSortedTerm[:100]  
        
stopwords = []
stopwordfile = 'english.stop.txt'
with open(stopwordfile, 'r') as f:
    for line in f:
        stopwords.append(line.split('\n')[0])
f.close()

'''Compare initial stopword list with the specific stopwords'''
swCommon = set(stopwords).intersection(set(topNgrams))
swDiff = set(stopwords).difference(set(topNgrams))
swDiff2 = set(topNgrams).difference(set(stopwords))
swNew = set(stopwords).union(set(topNgrams))
print 'SwCommon:'
print swCommon

file1 = open('swCommon.csv', 'wb')
writer = csv.writer(file1)
for row in swCommon:
    writer.writerow(row)
file1.close()    
    
print 'swDiff:'
print swDiff

file2 = open('swDiff.csv', 'wb')
writer = csv.writer(file2)
for row in swDiff:
    writer.writerow(row)
file2.close()    

print 'swDiff2:'
print swDiff2

print 'swDiff2:'
print swDiff2

file3 = open('swDiff2.csv', 'wb')
writer = csv.writer(file3)
for row in swDiff2:
    writer.writerow(row)
file3.close()    


'''Merge stopwords, remove rare words, contruct controlled vocabulary'''

    
stopwordFinal = list(set(topNgrams + stopwords))
rareIndex = dfsVocbSortedVal.index(49)
ctrlVocb = dfsVocbSortedTerm[100:rareIndex]
ctrlVocbDF = dfsVocbSortedVal[100:rareIndex]
sizeCtrlVocb = len(ctrlVocb)
ctrlVocbIDF = 1 + np.log(sizeCtrlVocb/np.asarray(ctrlVocbDF))

ctrlVocbDict = {}
for i in range(len(ctrlVocb)):
    ctrlVocbDict[ctrlVocb[i]] = ctrlVocbIDF[i]

print 'Get final controlled Vocb!'

top50Tokens = ctrlVocb[:50]
top50IDF = 1 + np.log(sizeCtrlVocb/np.asarray(ctrlVocbDF[:50]))
top50 = zip(top50Tokens, top50IDF)

print 'Top 50 Ngrams as follows:'
print top50
    
bottom50Tokens = ctrlVocb[-50:]
bottom50IDF = 1 + np.log(sizeCtrlVocb/np.asarray(ctrlVocbDF[-50:]))
bottom50 = zip(bottom50Tokens, bottom50IDF)

print 'Bottom 50 Ngrams as follows:'
print bottom50

print 'Part 1.2 Finished!' 

#with open('reviewStopwords.pickle', 'wb') as f:
#    pickle.dump([topNgrams, top50, bottom50, sizeCtrlVocb, swCommon, swDiff, swDiff2],f)
#
#with open('ctrlVocbDict.pickle', 'wb') as f:
#    pickle.dump(ctrlVocbDict,f)
#with open('ctrlVocbDict.pickle', 'rb') as f:
#    ctrlVocbDict = pickle.load(f)    
    

'''Part 1.3'''

'''First read query and find all the terms and then use this and the intersection with controlled vocb to read test reviews'''
ctrlVocbDict = {}
for i in range(len(ctrlVocb)):
    ctrlVocbDict[ctrlVocb[i]] = ctrlVocbIDF[i]

with open('ctrlVocbDict.pickle', 'wb') as fi:
    pickle.dump(ctrlVocbDict,fi)   
    
    
    
def readQuery(jsonfile, s):
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\d|\w+')
    stemmer = nltk.stem.PorterStemmer()
    tfs = {}
    reviews = {}
    authors = {}
    dates = {}
    j = 0
    if os.path.isfile(jsonfile):
        jfile = unicode(open(jsonfile).read(), 'ISO-8859-1')
        jsondata = json.loads(jfile)
        try:
            for k in range(len(jsondata['Reviews'])):
                try:
                    reviews[s+str(j)] = jsondata['Reviews'][k]['Content']
                    authors[s+str(j)] = jsondata['Reviews'][k]['Author']
                    dates[s+str(j)] = jsondata['Reviews'][k]['Date']
                    tokens = tokenizer.tokenize(reviews[s+str(j)])
                    stemmedTokens = []
                    bigram = []
                    for t in range(len(tokens)):
                        try:
                            tk = int(tokens[t])
                            tk = "NUM"
                        except ValueError:
                            tk = tokens[t]            
                        stemmedToken = stemmer.stem(tk.lower())
                        stemmedTokens.append(stemmedToken)
                    for m in range(len(stemmedTokens)-1):
                        tm = stemmedTokens[m] + '-' + stemmedTokens[m+1]
                        bigram.append(tm)                        
                    unibigram = stemmedTokens + bigram
                    c1 = Counter(unibigram)
                    tfs[s+str(j)] = c1
                    j += 1
                except ValueError:
                    print 'Cannot find Review Content!'
        except ValueError:
            print 'Cannot find Review!'
        
    return tfs, reviews, authors, dates    
     
queryfolder = r'./Yelp_small0/query.json'
queryTFs, queryReviews, queryAuthors, queryDates = readQuery(queryfolder, 'query')    
    
queryTerms = []
for k in queryTFs.keys():
    queryTerms += list(queryTFs[k])
queryTerms = set(queryTerms)

finalTerms = list(set(queryTerms).intersection(set(ctrlVocbDict.keys())))
finalTermsIDF = []
for i in finalTerms:
    finalTermsIDF.append(ctrlVocbDict[i])
     
'''Compute TF-IDF weighted vector'''
def TFIDF(tfs, finalTerms, finalTermsIDF):

    TFIDFs = {}
    for key, value in tfs.iteritems():
        tfidfEachReview = []
        for k in range(len(finalTerms)):
            if value[finalTerms[k]] > 0:
                tf = 1 + np.log(value[finalTerms[k]])
            else:
                tf = 0
            tfidf = tf * finalTermsIDF[k]
            tfidfEachReview.append(tfidf)
        TFIDFs[key] = tfidfEachReview
    return TFIDFs   
      
queryTFIDF = TFIDF(queryTFs, finalTerms, finalTermsIDF)

tokenizer = nltk.tokenize.RegexpTokenizer(r'\d|\w+')
stemmer = nltk.stem.PorterStemmer()   

def readJson3(jpath, s, finalTerms, finalTermsIDF, queryTFIDFs):
    reviews = {}
    authors = {}
    dates = {}
    j = 0
    cosinesAll = {}
    for key0 in queryTFIDFs.keys():
        cosinesAll[key0] = []
    for f in os.listdir(jpath):
        fpath = os.path.join(jpath, f)
        if os.path.isfile(fpath):
            jfile = open(fpath).read()
            jsondata = json.loads(jfile)
            try:
                for k in range(len(jsondata['Reviews'])):
                    try:
                        reviews[s+str(j)] = jsondata['Reviews'][k]['Content']
                        authors[s+str(j)] = jsondata['Reviews'][k]['Author']
                        dates[s+str(j)] = jsondata['Reviews'][k]['Date']
                        tokens = tokenizer.tokenize(reviews[s+str(j)])
                        stemmedTokens = []
                        stemmedTokenF = []
                        bigram = []
                        for t in range(len(tokens)):
                            try:
                                tk = int(tokens[t])
                                tk = "NUM"
                            except ValueError:
                                tk = tokens[t]            
                            stemmedToken = stemmer.stem(tk.lower())
                            stemmedTokens.append(stemmedToken)
                            if stemmedToken in finalTerms:
                                stemmedTokenF.append(stemmedToken)
                        for m in range(len(stemmedTokens)-1):
                            tm = stemmedTokens[m] + '-' + stemmedTokens[m+1]
                            if tm in finalTerms:
                                bigram.append(tm)                        
                        unibigram = stemmedTokenF + bigram
                        c1 = Counter(unibigram)
                        tfidfEachReview = []
                        for x in range(len(finalTerms)):
                            if c1[finalTerms[x]] > 0:
                                tf = 1 + np.log(c1[finalTerms[x]])
                            else:
                                tf = 0
                            tfidf = tf * finalTermsIDF[x]
                            tfidfEachReview.append(tfidf) 
                        for key0, value0 in queryTFIDFs.iteritems():
                            cosine = cosine_similarity(value0, tfidfEachReview)
                            infoDoc = (cosine, reviews[s+str(j)], authors[s+str(j)], dates[s+str(j)])
                            cosinesAll[key0].append(infoDoc)
                        j += 1
                    except ValueError:
                        print 'Cannot find Review Content!'
            except ValueError:
                print 'Cannot find Review!'
    cosines = {}
    for k, value in cosinesAll.iteritems():
        cosines[k] = sorted(cosinesAll[k], key=itemgetter(0), reverse = True)[:3]
    return cosines
    
testfolder = r'./Yelp_small0/test/'      
cosineSims = readJson3(testfolder, 'test', finalTerms, finalTermsIDF, queryTFIDF)

simReviews = {}
for k, value in cosineSims.iteritems():
    simReviews[k] = sorted(cosineSims[k], key=itemgetter(0), reverse = True)[:3]
    
print 'Part 1.3 Finished!'



