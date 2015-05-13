# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 14:41:22 2015
Description: Machine Problem 1, Part2: Statistical Language Models
@author: Jinghe Zhang
"""
import os
import json
import nltk
import operator
import random
import bisect
import numpy as np

''' Read json files '''
def readJson(jpath, s):
    reviews = {}
    uniCount = {}
    biCount = {}
    j = 0
    linkDict = dict()
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
                        lenT = len(tokens)
                        if lenT > 0:                            
                            t0 = stemmer.stem(tokens[0].lower())
                            if uniCount.__contains__(t0):
                                uniCount[t0] += 1
                            else:
                                uniCount[t0] = 0
                                uniCount[t0] += 1
                            if lenT > 1:                               
                                for t in tokens[1:]:
                                    if t.isdigit():
                                        t = "NUM"
                                    else:
                                        t = t            
                                    stemmedToken = stemmer.stem(t.lower())
                                    if uniCount.__contains__(stemmedToken):
                                        uniCount[stemmedToken] += 1
                                    else:
                                        uniCount[stemmedToken] = 0
                                        uniCount[stemmedToken] += 1
                                    biToken = t0 + '%#%' + stemmedToken
                                    if biCount.__contains__(biToken):
                                        biCount[biToken] += 1
                                    else:
                                        biCount[biToken] = 0
                                        biCount[biToken] += 1
                                    if linkDict.__contains__(t0):
                                        linkDict[t0].add(stemmedToken)
                                    else: 
                                        linkDict[t0] = set()
                                        linkDict[t0].add(stemmedToken)                                
                                    t0 = stemmedToken                            
                        j += 1
                    except ValueError:
                        print 'Cannot find Review Content!'
            except ValueError:
                print 'Cannot find Review!'
    return uniCount, biCount, linkDict
    
    
def constructLM(uniCount, biCount, linkDict, lam, delta):    
    lenUnigram = sum(uniCount.values())
    biprobL = biCount.copy()
    biprobA = biCount.copy()
    for key, value in biCount.iteritems():
        w0, w1 = key.split('%#%')
        prob1 = float(value) / uniCount[w0]
        prob2 = float(uniCount[w1]) / lenUnigram
        biprobL[key] = lam * prob1 + (1-lam) * prob2
        
    for key, value in biCount.iteritems():
        w0, w1 = key.split('%#%')
        S = len(linkDict[w0])    
        term1 = float(max(value - delta, 0)) / uniCount[w0]    
        term2 = delta * S / uniCount[w0] * uniCount[w1] / lenUnigram
        biprobA[key] = term1 + term2
    return biprobL, biprobA

def rankWords(biprobL, biprobA, linkDict, uniCount, lam, delta):    
    lenUnigram = sum(uniCount.values())
    goodL = {}
    goodA = {}
    S = len(linkDict['good'])  
    for w in uniCount.keys():
        if w in linkDict['good']:
            goodL['good%#%' + w] = biprobL['good%#%' + w]
            goodA['good%#%' + w] = biprobA['good%#%' + w]
        else: 
            goodL['good%#%' + w] = ((1 - lam) * uniCount[w] / lenUnigram)
            goodA['good%#%' + w] = (delta * S / uniCount['good'] * uniCount[w] / lenUnigram) 
    sorted_L = sorted(goodL.items(), key=operator.itemgetter(1), reverse = True)[:10]    
    sorted_A = sorted(goodA.items(), key=operator.itemgetter(1), reverse= True)[:10] 
    print sum(goodL.values())
    print sum(goodA.values())
    return sorted_L, sorted_A
    
def sentence1(uniCount):
    #initialize sampling
    p = 1
    wd = ''
    lenUnigram = sum(uniCount.values())
    words = uniCount.keys()
    #sampling 
    for i in range(15):
        probs = []
        v = 0
        for w in words:
            v += float(uniCount[w]) / lenUnigram
            probs.append(v)
        num = random.uniform(0, v)
        windex = bisect.bisect_left(probs, num)
        wi = words[windex]  
        p1 = float(uniCount[wi]) / lenUnigram
        p = p * p1
        wd = wd + '-' + wi
    return wd, p
    
def sentence2(uniCount, biprob, linkDict, biM):
    #initialize sampling
    p = 1
    probs = []
    v = 0
    lenUnigram = sum(uniCount.values())
    words = uniCount.keys()
    #sample the first word in a sentence
    for w in words:
        v += float(uniCount[w]) / lenUnigram
        probs.append(v)
    num = random.uniform(0, v)
    windex = bisect.bisect_left(probs, num)
    wi = words[windex]  
    p1 = float(uniCount[wi]) / lenUnigram
    p = p * p1
    newDict = {}
    wd = wi
    #smaple the 2-15 words in a sentence
    for i in range(1,15):
        probs = []        
        v = 0
        S = len(linkDict[wi]) 
        for w in words:
            if w in linkDict[wi]:
                newDict[wi + '%#%' + w] = biprob[wi + '%#%' + w] 
            elif biM == 'L':
                newDict[wi + '%#%' + w] = (1 - lam) * uniCount[w] / lenUnigram 
            else:
                newDict[wi + '%#%' + w] = delta * S / uniCount[wi] * uniCount[w] / lenUnigram      
            v += newDict[wi + '%#%' +w]                 
            probs.append(v)
        num = random.uniform(0, v)
        windex = bisect.bisect_left(probs, num)
        wnew = words[windex]
        if i == 0:
            p1 = float(uniCount[wnew]) / lenUnigram
        else:
            p1 = newDict[wi + '%#%' + wnew]
        p = p * p1
        wd = wd + '-' + wnew
        wi = wnew
    return wd, p
        
def testJson(jpath, s):
    reviews = {}
    j = 0
    uniDoc = {}
    biDoc = {}
    unigrams = {}
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
                        uniDoc[s+str(j)] = []    
                        
                        lenT = len(tokens)                        
                        if lenT > 0:                            
                            t0 = stemmer.stem(tokens[0].lower())
                            uniDoc[s+str(j)].append(t0)
                            if not unigrams.__contains__(t0):
                                unigrams[t0] = 0
                            if lenT > 1: 
                                biDoc[s+str(j)] = []
                                for t in tokens[1:]:
                                    if t.isdigit():
                                        t = "NUM"
                                    else:
                                        t = t            
                                    stemmedToken = stemmer.stem(t.lower())
                                    uniDoc[s+str(j)].append(stemmedToken)
                                    if not unigrams.__contains__(stemmedToken):
                                        unigrams[stemmedToken] = 0
                                    biToken = t0 + '%#%' + stemmedToken
                                    biDoc[s+str(j)].append(biToken)                                        
                                    t0 = stemmedToken 
                            j += 1
                    except ValueError:
                        print 'Cannot find Review Content!'
            except ValueError:
                print 'Cannot find Review!'
    return uniDoc, biDoc, unigrams
    
def perplexity(uniDoc, biDoc, unigrams, delta2, uniCount, biCount, linkDict, delta, lam):
    lenUnigram = sum(uniCount.values())
    pp1 = []
    ppL = []
    ppA = []    
    vs = len(uniCount) + 1
    #smoothing and perplexity computation for unigram LM
    for key, value in uniDoc.iteritems():
        p = 0 
        for i in value:
            if uniCount.__contains__(i):
                unigrams[i] = float(uniCount[i] + delta2) / (lenUnigram + delta2 * vs)    
            else:
                unigrams[i] = float(delta2) / (lenUnigram + delta2 * vs)   
            p = p + np.log2(1./unigrams[i])
        val = np.exp((1./len(value)) * p)        
        pp1.append(val)
    # perplexity for bigram LMs using the smoothed unigrams 
    for key, value in biDoc.iteritems(): 
        start = uniDoc[key][0]
        pL = np.log2(1./unigrams[start])
        pA = np.log2(1./unigrams[start])  
        for i in value:
            w0, w1 = i.split('%#%')
            if biCount.__contains__(i):                                
                S = len(linkDict[w0])
                pL0 = lam * biCount[i] / uniCount[w0] + (1 - lam) * unigrams[w1]
                pA0 = float(max(biCount[i] - delta, 0)) / uniCount[w0] + delta * S / uniCount[w0] * unigrams[w1]
            elif uniCount.__contains__(w0):
                pL0 = (1 - lam) * unigrams[w1]
                pA0 = delta * S / uniCount[w0] * unigrams[w1]
            else:
                pL0 = (1 - lam) * unigrams[w1]
                pA0 = 1 * unigrams[w1] 
            pL = pL + np.log2(pL0)
            pA = pA + np.log2(pA0)
        valL = np.exp(pL * (1./len(value)))
        valA = np.exp(pA * (1./len(value)))
        ppL.append(valL)
        ppA.append(valA)      
    return pp1, ppL, ppA                
                            
'''Part 2.1'''
jpath = r'./Yelp_small0/train/'   
tokenizer = nltk.tokenize.RegexpTokenizer(r'\d|\w+')
stemmer = nltk.stem.PorterStemmer()
uniCount, biCount, linkDict = readJson(jpath, 'train')  
lam = 0.9
delta = 0.1 
biprobL, biprobA = constructLM(uniCount, biCount, linkDict, lam, delta)     
sorted_L, sorted_A = rankWords(biprobL, biprobA, linkDict, uniCount, lam, delta)
print sorted_L
print sorted_A


'''Part 2.2'''
sentence = []
sentenceProb = []
for i in range(10):
    s, p = sentence1(uniCount)
    sentence.append(s)
    sentenceProb.append(p)
print sentence
print sentenceProb


sentenceL = []
sentenceProbL = []
for i in range(10):
    s, p = sentence2(uniCount, biprobL, linkDict, 'L')
    sentenceL.append(s)
    sentenceProbL.append(p)
print sentenceL
print sentenceProbL

sentenceA = []
sentenceProbA = []
for i in range(10):
    s, p = sentence2(uniCount, biprobA, linkDict, 'A')
    sentenceA.append(s)
    sentenceProbA.append(p)
print sentenceA
print sentenceProbA

'''Part 2.3'''

delta2 = 0.1
jpathtest = r'./Yelp_small0/test/'  
uniDoc, biDoc, unigrams = testJson(jpathtest, 'test')
pp1, ppL, ppA = perplexity(uniDoc, biDoc, unigrams, delta2, uniCount, biCount, linkDict, delta, lam)
pp1m = []
for i in pp1:
    if i != inf:
        pp1m.append(i)
ppLm = []
for i in ppL:
    if i != inf:
        ppLm.append(i)        
ppAm = []
for i in ppA:
    if i != inf:
        ppAm.append(i)  
meanpp1 = np.mean(pp1m)
stdpp1 = np.std(pp1m)
meanppL = np.mean(ppLm)
stdppL = np.std(ppLm)
meanppA = np.mean(ppAm)
stdppA = np.std(ppAm)



