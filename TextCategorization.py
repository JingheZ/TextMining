# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 14:41:22 2015
Description: Machine Problem 2: Text Categorization
@author: Jinghe Zhang
"""
import os
import json
import nltk
import operator
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.metrics.pairwise import cosine_similarity
import random
import time
from sklearn.cross_validation import KFold
from scipy import stats
# import csv
import pickle


def addReview(tokens, uniCount, doc):
    '''construct document dictionary of terms with TF and token dictionary of df'''
    for t in tokens:
        if t.isdigit():
            t = "NUM"
        t0 = stemmer.stem(t.lower())
        if doc.__contains__(t0):
            doc[t0] += 1
        else:
            doc[t0] = 1
            if uniCount.__contains__(t0):
                uniCount[t0] += 1
            else:
                uniCount[t0] = 1

    return uniCount, doc


def readJson(jpath):
    '''Read Json file and save token dictionary, document dictionary, and review dictionary'''
    reviews = {}
    posTokens = {}
    negTokens = {}
    posDocs = {}
    negDocs = {}
    j = 0
    for f in os.listdir(jpath):
        fpath = os.path.join(jpath, f)
        if os.path.isfile(fpath):
            jfile = open(fpath).read()
            jsondata = json.loads(jfile)
            try:
                for k in range(len(jsondata['Reviews'])):
                    try:
                        review0 = jsondata['Reviews'][k]['Content']
                        rate = jsondata['Reviews'][k]['Overall']
                        rate = float(rate)
                        tokens = tokenizer.tokenize(review0)
                        lenT = len(tokens)
                        if lenT > 0:
                            if rate >= 4.0:
                                reviews['pos'+str(j)] = review0
                                posTokens, posDocs['pos'+str(j)] = addReview(tokens, posTokens, {})
                            else:
                                reviews['neg'+str(j)] = review0
                                negTokens, negDocs['neg'+str(j)] = addReview(tokens, negTokens, {})
                        j += 1
                    except ValueError:
                        print 'Cannot find Review Content or Rating!'
            except ValueError:
                print 'Cannot find Review!'
    return posTokens, negTokens, posDocs, negDocs, reviews


def dfSelection(posTokens, negTokens, stopwords):
    '''Remove terms with small df and remove stopwords'''
    tokenAll = {}
    allkeys = set(posTokens.keys()).union(set(negTokens.keys()))
    for i in allkeys:
        tokenAll[i] = 0.
        if posTokens.__contains__(i):
            tokenAll[i] += posTokens[i]
        if negTokens.__contains__(i):
            tokenAll[i] += negTokens[i]
    features = []
    for key, value in tokenAll.iteritems():
        if value >= 50 and key not in stopwords:
            features.append(key)
    return features


def infoCls(pr):
    '''Compute the p*log2(p) in information gain'''
    if pr > 0:
        info = pr * np.log2(pr)
    else:
        info = 0.0
    return info


def featureSelection(posTokens, negTokens, posNum, negNum, stopwords):
    '''Information gain and chi-square for feature selection'''
    features0 = dfSelection(posTokens, negTokens, stopwords)
    py1 = float(posNum)/(posNum + negNum)
    py0 = float(negNum)/(posNum + negNum)
    term1 = -(infoCls(py1) + infoCls(py0))
    featureIG = []
    featureChiS = []
    for t in features0:
        if posTokens.__contains__(t):
            df1 = posTokens[t]
        else:
            df1 = 0.
        if negTokens.__contains__(t):
            df0 = negTokens[t]
        else:
            df0 = 0.
        df = df1 + df0
        p1 = float(df)/(posNum+negNum)
        p0 = 1 - p1
        py1t1 = float(df1)/posNum * py1 / p1
        py0t1 = float(df0)/negNum * py0 / p1
        py1t0 = float(posNum-df1)/posNum * py1 / p0
        py0t0 = float(negNum-df0)/negNum * py0 / p0
        term2 = p1 * (infoCls(py1t1) + infoCls(py0t1))
        term3 = p0 * (infoCls(py1t0) + infoCls(py0t0))
        #information gain
        IG = term1 + term2 + term3
        #compute chi-square value
        chiS = (posNum + negNum) * ((df1 * (negNum - df0) - (posNum - df1) * df0)**2) / (df * (posNum-df1 + negNum-df0) * posNum * negNum)
        featureTupleIG = (t, IG)
        featureIG.append(featureTupleIG)
        if chiS >= 3.841:
            featureTupleChiS = (t, chiS)
            featureChiS.append(featureTupleChiS)
    featureIG.sort(key = operator.itemgetter(1), reverse = True)
    featureChiS.sort(key = operator.itemgetter(1), reverse = True)

    IG_selected = []
    chiS_selected = []

    lenIG = min(5000, len(featureIG))
#    lenIG = 5000
    for i in range(lenIG):
        IG_selected.append(featureIG[i][0])

    lenChiS = min(5000, len(featureChiS))
    for i in range(lenChiS):
        chiS_selected.append(featureChiS[i][0])

    feature_selected = set(IG_selected).union(set(chiS_selected))

    return feature_selected, featureIG, featureChiS


def initFeatureInfo(features_selected):
    '''Construct feature dictionary of DF and TTF'''
    features = {}
    for i in features_selected:
        features[i] = {}
        features[i]['DF'] = 0.
        features[i]['TTF'] = 0.
    return features


def docSelection(Docs, features_selected):
    '''filter out documents with no more than 5 selected features; compute the TTF and DF of features based on the selected documents'''
    features = initFeatureInfo(features_selected)
    Docs2 = {}
    for key, value in Docs.iteritems():
        commonfeatures = features_selected.intersection(set(value.keys()))
        if len(commonfeatures) > 5:
            Docs2[key] = {}
            for v in commonfeatures:
                Docs2[key][v] = value[v]
                features[v]['DF'] += 1
                features[v]['TTF'] += Docs2[key][v]
    return Docs2, features


def computeTTFDF(Docs, features_selected):
    '''Compute the TTF and DF of selected features based on selected documents'''
    pos_features = initFeatureInfo(features_selected)
    neg_features = initFeatureInfo(features_selected)

    for key, value in Docs.iteritems():
        if 'pos' in key:
            for v in value.keys():
                pos_features[v]['DF'] += 1
                pos_features[v]['TTF'] += Docs[key][v]
        else:
            for v in value.keys():
                neg_features[v]['DF'] += 1
                neg_features[v]['TTF'] += Docs[key][v]

    return pos_features, neg_features


def tttf(featureInfo):
    '''For smoothing, to compute the sum of total term frequency'''
    tttf = 0
    for i in featureInfo.keys():
        tttf += featureInfo[i]['TTF']
    return tttf


def featureProb(features, term, tttf, sizeV):
    '''additive smoothing to compute probability of a particular feature'''
    pr = (features[term]['TTF'] + 0.1) / (tttf + 0.1 * (sizeV + 1))
    return pr


def maxPosterior(features, pos_feature, neg_feature):
    '''Compute the max a posterior of features and their log ratios'''
    logRatios = []
    featureDict = {}
    sizeVpos = len(pos_feature)
    sizeVneg = len(neg_feature)
    tttf_pos = tttf(pos_feature)
    tttf_neg = tttf(neg_feature)
    for i in features:
        logr = np.log2(featureProb(pos_feature, i, tttf_pos, sizeVpos)) - np.log2(featureProb(neg_feature, i, tttf_neg, sizeVneg))
        logRatios.append((i, logr))
        featureDict[i] = logr
    logRatios.sort(key = operator.itemgetter(1), reverse = True)

    return logRatios, featureDict


def naiveBayesCls(featureDict, doc, constant):
    '''Naive Bayes classifier for a document; output predicted label'''
    fx = 0
    for key, value in doc.iteritems():
        fx += featureDict[key] * value # should I consider the words by freq or binary???
    fx += constant

    if fx >= 0:
        label = 1
    else:
        label = 0
    return fx, label


def computeFx(Docs, lenposDocs, lennegDocs, featureDict):
    '''Nayes Bayes Classifer for many documents'''
    constant = np.log2(lenposDocs/lennegDocs)
    results = []
    for key, value in Docs.iteritems():
        fx, label = naiveBayesCls(featureDict, value, constant)
        tup = (key, fx, label)
        results.append(tup)
    results.sort(key = operator.itemgetter(1), reverse = True)
    return results


def computePR(predictPos, posNum):
    '''compute precision and recall of a classification outcome'''
    TP = 0.0
    for i in predictPos:
        if 'pos' in i[0]:
            TP += 1
    if len(predictPos) > 0:
        precision = TP/len(predictPos)
    else:
        precision = 0.
    if posNum > 0:
        recall = TP/posNum
    else:
        recall = 0.
    return precision, recall

def prcurve(results, posNum, n):
    '''Plot precision-recall curve'''
    minFx = results[-1][1]
    maxFx = results[0][1]
    thresholds = np.linspace(minFx, maxFx, n)
    precisions = []
    recalls = []
    for i in thresholds[1:-1]:
        for r in range(len(results)):
            if results[r][1] < i:
                break
        predictPos = results[:r]
        prec, rec = computePR(predictPos, posNum)

        precisions.append(prec)
        recalls.append(rec)

    plt.figure()
    plt.scatter(recalls, precisions, marker = '.')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.title('Precision-Recall Curve of Naive Bayes Classifier')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig('precision-recall.png')
    plt.show()

    return precisions, recalls


def constructVector(Docs, lenDocs, posfeatureInfo, negfeatureInfo):
    '''Compute tfidf with sublinear scaling of TF for query and create a vector for each query'''
    origVector = {}
    for key, value in Docs.iteritems():
        origVector[key] = {}
        for term in value.keys():
            tf = 1. + np.log2(value[term])
            df = 0.
            if posfeatureInfo.__contains__(term):
                df += posfeatureInfo[term]['DF']
            if negfeatureInfo.__contains__(term):
                df += negfeatureInfo[term]['DF']
            if df > 0:
                idf = 1. + np.log2(lenDocs) - np.log2(df)
                tfidf = tf * idf
                origVector[key][term] = tfidf
    return origVector


def readQuery(jsonfile):
    '''read queries from json files'''
    reviews = {}
    queryTokens = {}
    queryDocs = {}
    queryRates = {}
    j = 0
    if os.path.isfile(jsonfile):
        jfile = unicode(open(jsonfile).read(), 'ISO-8859-1')
        jsondata = json.loads(jfile)
        try:
            for k in range(len(jsondata['Reviews'])):
                try:
                    reviews['query'+str(j)] = jsondata['Reviews'][k]['Content']
                    rate = jsondata['Reviews'][k]['Overall']
                    rate = float(rate)
                    queryRates['query'+str(j)] = rate
                    tokens = tokenizer.tokenize(reviews['query'+str(j)])
                    lenT = len(tokens)
                    if lenT > 0:
                        queryTokens, queryDocs['query'+str(j)] = addReview(tokens, queryTokens, {})
                    j += 1
                except ValueError:
                    print 'Cannot find Review Content or Rating!'
        except ValueError:
            print 'Cannot find Review!'
    return queryTokens, queryDocs, reviews, queryRates


def majorityVoting(data, name):
    '''majority voting for KNN to determine the predicted label'''
    num0 = 0.
    num1 = 0.
    for i in data:
        if 'pos' in i:
            num1 += 1
        else:
            num0 += 1
    if num1 > num0:
        label = 1
    else:
        label = 0
    result = (name, (num1, num0), label)
    return result


def randomProjection(l, features_selected):
    '''create random unit vectors'''
    randvector = {}
    features_selected = list(features_selected)
    for i in range(l):
        randvector0 = []
        randvector[i] = {}
        srqtsum = 0.
        for j in range(len(features_selected)):
            num = np.random.normal(0,1)
            srqtsum += num ** 2
            randvector0.append(num)
        srqtsum = np.sqrt(srqtsum)
        for j in range(len(randvector0)):
            randvector[i][features_selected[j]] = randvector0[j]/srqtsum
    return randvector


def convertbucket(value, randvectors):
    bucket = ''
    for keyrd, valuerd in randvectors.iteritems():
        v0 = 0
        for term in value.keys():
            v0 += value[term] * valuerd[term]
        if v0 > 0:
            v = '1'
        else:
            v = '0'
        bucket += v
    return bucket


def constructBuckets(DocVectors, randvector):
    '''project original vectors to random vectors'''
    allbucket = {}
    for key, value in DocVectors.iteritems():
        bucket = convertbucket(value, randvector)
        if allbucket.__contains__(bucket):
            allbucket[bucket].append(key)
        else:
            allbucket[bucket] = [key]
    return allbucket

def cosineSimilarity(query, vector):
    terms = set(query.keys()).intersection(set(vector.keys()))
    querynorm = 0.
    for i in query.values():
        querynorm += i ** 2
    querynorm = np.sqrt(querynorm)
    vectornorm = 0.
    for j in vector.values():
        vectornorm += j ** 2
    vectornorm = np.sqrt(vectornorm)
    sim = 0.
    for t in terms:
        sim += query[t] * vector[t]
    sim = sim / vectornorm / querynorm
    return sim


def computeCosine(query, origVector, k):
    '''Compute the cosine similarity between query and doc and sort the docs for each query by the cosine value;
    call the majority voting function to determine the predicted label for each query'''
    cosineQ = []
    neighbors = []
    for key0, value0 in origVector.iteritems():
        cosine = cosineSimilarity(query, value0)
        cosineQ.append((key0, cosine))
    cosineQ.sort(key=operator.itemgetter(1), reverse=True)
    for i in range(k):
        neighbors.append(cosineQ[i][0])
    return neighbors


def KNNclassifier_rp(allbucket, query, randvectors, DocVectors, k):
    '''classify test reviews using KNN'''
    results = []
    allneighbors = {}
    for key, value in query.iteritems():
        testbucket = convertbucket(value, randvectors)
        if allbucket.__contains__(testbucket):
            neighbors = findNeighbors(value, allbucket[testbucket], DocVectors, k)
        else:
            neighbors = findNeighbors(value, DocVectors.keys(), DocVectors, k)
        prediction = majorityVoting(neighbors, key)
        results.append(prediction)
        allneighbors[key] = neighbors
    return results, allneighbors


def findNeighbors(query, docNames, DocVectors, k):
    '''find the doc names of the k nearest neighbors'''
    if len(docNames) <= k:
        neighbors = docNames
    else:
        bucketVector = {}
        for i in docNames:
            bucketVector[i] = DocVectors[i]
        neighbors = computeCosine(query, bucketVector, k)
    return neighbors


def ParameterTuningKNN(vectors, features_selected, nfold, ks, ls):
    random.seed(0)
    keys = vectors.keys()
    kf = KFold(len(vectors), n_folds=nfold)
    train, test = creatCVfolds(vectors, keys, kf)
    all_result = []
    #create different random vectors
    for l0 in ls:
        randvector0 = randomProjection(l0, features_selected)
        results0 = []
        for n in range(nfold): #cross validation
            resultsfolds = []
            buckets = constructBuckets(train[n], randvector0)
            results00 = []
            for k0 in ks:
                results00.append([])
            for key, value in test[n].iteritems():
                testbucket = convertbucket(value, randvector0)

                if buckets.__contains__(testbucket):
                    cosineq = []
                    for key0 in buckets[testbucket]:
                        cosine = cosineSimilarity(value, train[n][key0])
                        cosineq.append((key0, cosine))
                    cosineq.sort(key=operator.itemgetter(1), reverse=True)
                    for k in range(len(ks)):
                        neighbor = []
                        if len(cosineq) < ks[k]:
                            neighbor = buckets[testbucket]
                        else:
                            for k00 in range(ks[k]):
                                neighbor.append(cosineq[k00][0])
                        pred = majorityVoting(neighbor, key)
                        results00[k].append(pred)
            for k1 in range(len(ks)):
                metrics = confusionMat(results00[k1])
                resultsfolds.append(metrics)
            results0.append(resultsfolds)
        all_result.append(results0)
    # The final results contains several layers of info: layer 1 - All different random vectors; layer 2: 10 folds; layer 3: all different k
    return all_result


def KNNclassifier(query, DocVectors, k):
    '''classify test reviews using KNN'''
    results = []
    allneighbors = {}
    for key, value in query.iteritems():
        neighbors = findNeighbors(value, DocVectors.keys(), DocVectors, k)
        prediction = majorityVoting(neighbors, key)
        results.append(prediction)
        allneighbors[key] = neighbors
    return results, allneighbors


def creatCVfolds(Docs, keys, kf):
    '''create the sampling index for train and test in cross-validation'''
    train = []
    test = []
    for trainindex, testindex in kf:
        train0 = {}
        test0 = {}
        for i in trainindex:
            train0[keys[i]] = Docs[keys[i]]
        for j in testindex:
            test0[keys[j]] = Docs[keys[j]]
        train.append(train0)
        test.append(test0)

    return train, test


def computeF1(prec, rec):
    '''Compute the F1 using precision and recall'''
    if prec > 0 and rec > 0:
        F1 = 1./(.5/prec + .5/rec)
    else:
        F1 = 0.
    return F1


def confusionMat(results):
    '''compute precision, recall, and F1 for a classification output'''
    predictedPos = []
    posNum = 0.
    for i in results:
        if i[2] == 1:
            predictedPos.append(i)
        if 'pos' in i[0]:
            posNum += 1.
    prec, rec = computePR(predictedPos, posNum)
    F1 = computeF1(prec, rec)
    result = (prec, rec, F1)
    return result


def crossValidationEvaluation(Docs, vectors, featureDict, lenposDocs, lennegDocs, nfold, randvectors, k):
    '''Implementation of cross validation '''
    random.seed(0)
    keys = Docs.keys()
    kf = KFold(len(Docs), n_folds = nfold)
    trainDocs, testDocs = creatCVfolds(Docs, keys, kf)
    trainVectors, testVectors = creatCVfolds(vectors, keys, kf)
    resultsa = []
    resultsb = []

    for i in range(nfold):
        '''Naive Bayes'''
        results00a = computeFx(testDocs[i], lenposDocs, lennegDocs, featureDict)
        result0a = confusionMat(results00a)

        '''KNN'''
        buckets = constructBuckets(trainVectors[i], randvectors)
        predictions_knn, neighbors_knn = KNNclassifier_rp(buckets, testVectors[i], randvectors, trainVectors[i], k)
        result0b = confusionMat(predictions_knn)
        resultsa.append(result0a)
        resultsb.append(result0b)
    return resultsa, resultsb, trainVectors, testVectors


def pairedT(resultsA, resultsB, nfold):
    final_results = []
    for i in range(3):
        sampleA = []
        sampleB = []
        for j in range(nfold):
            sampleA.append(resultsA[j][i])
            sampleB.append(resultsB[j][i])
        avgA = np.mean(sampleA)
        avgB = np.mean(sampleB)
        ttest = stats.ttest_rel(sampleA,sampleB)
        tup = (avgA, avgB, ttest)
        final_results.append(tup)
    return final_results


# def writetocsv(data, filename):
#     f = open(filename, 'wb')
#     mywriter = csv.writer(f)
#     mywriter.writerow(data)
#     f.close()
#

# def writerandvectorstocsv(data, filename):
#     f = open(filename, 'wb')
#     mywriter = csv.writer(f)
#     for i in data:
#         mywriter.writerow(i)
#     f.close()
#
#
# def writevectorscsv(data, filename):
#     f = open(filename, 'wb')
#     mywriter = csv.writer(f)
#     for key, value in data.iteritems():
#         mywriter.writerow(value)
#     f.close()
#
#
# def writedictcsv(dictData, filename):
#     f = open(filename, 'wb')
#     mywriter = csv.writer(f)
#     for key, value in dictData.iteritems():
#         mywriter.writerow((key,value))
#     f.close()
#
#
# def readfromcsv(filename):
#     data = []
#     f = open(filename, 'rb')
#     myreader = csv.reader(f)
#     for i in myreader:
#         data.append(i)
#     f.close()
#     return data[0]
#
#
# def readcsvdict(filename):
#     data = {}
#     f = open(filename, 'rb')
#     myreader = csv.reader(f)
#     for i in myreader:
#         data[i[0]] = {}
#         value = i[1].split(',')
#         for v in value:
#             k, c = v.split(':')
#             c = c.strip(' ')
#             data[i[0]][k] = c
#     f.close()
#     return data


def Task1and2(jpath, stopwords):
    posTokens, negTokens, posDocs, negDocs, reviews = readJson(jpath)
    lenposDocs = len(posDocs)
    lennegDocs = len(negDocs)
    features_selected, featureIG, featureChiS = featureSelection(posTokens, negTokens, lenposDocs, lennegDocs, stopwords)

    posDocs_filtered, pos_featureInfo = docSelection(posDocs, features_selected)
    negDocs_filtered, neg_featureInfo = docSelection(negDocs, features_selected)
    Docs = posDocs_filtered.copy()
    Docs.update(negDocs_filtered)
    logRatios, featureDict = maxPosterior(features_selected, pos_featureInfo, neg_featureInfo)
    results = computeFx(Docs, lenposDocs, lennegDocs, featureDict)
    precisions, recalls = prcurve(results, len(posDocs_filtered), 5000)
    return Docs, reviews, features_selected, pos_featureInfo, neg_featureInfo, featureDict, logRatios[:20], logRatios[-20:], featureIG[:20], featureChiS[:20], lenposDocs, lennegDocs


def Task3(Docs, features_selected, pos_featureInfo, neg_featureInfo, l):
    queryfolder = r'./Yelp_small/query.json'
    queryTokens, queryDocs, queryReviews, queryRates = readQuery(queryfolder)
    DocVectors = constructVector(Docs, len(Docs), pos_featureInfo, neg_featureInfo)
    queryVector = constructVector(queryDocs, len(Docs), pos_featureInfo, neg_featureInfo)
    k = 5
    t0a = time.time()
    predictions, neighbors = KNNclassifier(queryVector, DocVectors, k)
    t0b = time.time()
    timeKNN = t0b-t0a
    print 't0b-t0a:'
    print timeKNN

    randvectors = randomProjection(l, features_selected)
    buckets = constructBuckets(DocVectors, randvectors)
    t1a = time.time()
    predictions_rp, neighbors_rp = KNNclassifier_rp(buckets, queryVector, randvectors, DocVectors, k)
    t1b = time.time()
    timeKNN_rp = t1b-t1a
    print 't1b-t1a:'
    print timeKNN_rp

    return DocVectors, predictions, neighbors, timeKNN, randvectors, timeKNN_rp, predictions_rp, neighbors_rp, queryReviews, queryRates


def Task4(Docs, vectors, randvectors, featureDict, lenposDocs, lennegDocs):

    nfold = 10
    k = 5
    resultsNaive, resultsKNN, trainVectors, testVectors = crossValidationEvaluation(Docs, vectors, featureDict, lenposDocs, lennegDocs, nfold, randvectors, k)
    comparison = pairedT(resultsNaive, resultsKNN, nfold)

    return resultsNaive, resultsKNN, comparison, trainVectors, testVectors


def ParameterTuningResults(results, nfold, ks, l):
    ydata = []
    for i in range(len(ks)):
        ydata.append([])
    for r in range(len(results)):
        for k in range(len(results[r])):
            ydata[k].append(results[r][k])

    precisions = []
    recalls = []
    f1s = []
    for k0 in range(len(ks)):
        precision = 0.
        recall = 0.
        f1 = 0.
        for n in range(nfold):
            precision += ydata[k0][n][0]
            recall += ydata[k0][n][1]
            f1 += ydata[k0][n][2]
        precision = precision / nfold
        recall = recall / nfold
        f1 = f1 /nfold
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return precisions, recalls, f1s


def plotEvaluation(precisions, recalls, f1s, ks, xlab):
    plt.figure()
    plt.plot(ks, precisions)
    plt.plot(ks, recalls)
    plt.plot(ks, f1s)
    plt.xlabel(xlab)
    plt.ylabel('Performance')
    # plt.ylim([0.9998, 1.0001])
    plt.legend(['precision', 'recall', 'F1'], loc = 'lower right')
    plt.savefig(str(l)+'evaluation.png')
    plt.show()


def TuningEvaluation(all_results, ls, ks, nfold):
    precisions = []
    recalls = []
    f1s = []
    for i in range(len(all_results)):
        precision, recall, f1 = ParameterTuningResults(all_results[i], nfold, ks, ls[i])
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return precisions, recalls, f1s


def getMetricMax(metric):
    maxValue = []
    for i in metric:
        maxValue.append(max(i))
    return maxValue


'''MP2: Task1 Feature Selection'''
jpath = r'./Yelp_small/train/'
tokenizer = nltk.tokenize.RegexpTokenizer(r'\d|\w+')
stemmer = nltk.stem.PorterStemmer()

stopwords = []
stopwordfile = 'english.stop.txt'
with open(stopwordfile, 'r') as f:
    for line in f:
        stopwords.append(line.split('\n')[0])
f.close()

print 'Start Task 1 & 2...'
Docs, reviews, features_selected, pos_featureInfo, neg_featureInfo, featureDict, logRatiosTop, logRatiosBottom, featureIGtop, featureChiStop, lenposDocs, lennegDocs = Task1and2(jpath, stopwords)

print 'Selected Document Length:'
print len(Docs)
print 'Selected Words Length:'
print len(features_selected)
print 'Log Ratio Top 20 Words:'
print logRatiosTop
print 'Log Ratio Bottom 20 Words:'
print logRatiosBottom
print 'Info Gain Top 20 Words:'
print featureIGtop
print 'Chi Square Top 20 Words:'
print featureChiStop
print 'Selected Positive Document Length:'
print lenposDocs
print 'Selected Negative Document Length:'
print lennegDocs
# with open('Docs.json', 'wb') as f:
#    json.dump(Docs,f)
# f.close()

# with open('trainReviews.json', 'wb') as f:
#    json.dump(reviews,f)
# f.close()

# with open('featureDict.json', 'wb') as f:
#    json.dump(featureDict,f)
# f.close()
#
# with open('pos_featureInfo.json', 'wb') as f:
#    json.dump(pos_featureInfo,f)
# f.close()
#
# with open('neg_featureInfo.json', 'wb') as f:
#    json.dump(neg_featureInfo,f)
# f.close()
#
# # Selected Positive Document Length:
# # 443349
# # Selected Negative Document Length:
# # 186548
#
print 'Task 1 & 2 Finished!'


print 'Start Task 3...'
# Docs = json.loads(open('Docs.json').read())
# pos_featureInfo = json.loads(open('pos_featureInfo.json').read())
# neg_featureInfo = json.loads(open('neg_featureInfo.json').read())
# features_selected = set(pos_featureInfo).union(set(neg_featureInfo))

l = 10
DocVectors, predictions_knn, neighbors_knn, timeKNN, randvectors_knn, timeKNN_rp, predictions_rp, neighbors_rp, queryReviews, queryRates = Task3(Docs, features_selected, pos_featureInfo, neg_featureInfo, l)
print 'Prediction Results on Queries Using Brute Force KNN:'
print predictions_knn
print 'Brute Force KNN: neighbors:'
print neighbors_knn
print 'Time for Brute Force KNN:'
print timeKNN
print 'Prediction Results on Queries Using KNN with Random Projection:'
print predictions_rp
print 'KNN with Random Projection: neighbors:'
print neighbors_rp
print 'Time for KNN with Random Projection:'
print timeKNN_rp

# with open('DocVectors.json', 'wb') as f:
#    json.dump(DocVectors, f)
# f.close()
#
# with open('randvectors.json', 'wb') as f:
#    json.dump(randvectors, f)
# f.close()
print 'Task 3 Finished!'

print 'Start Task 4...'
# Docs = json.loads(open('Docs.json').read())
# DocVectors = json.loads(open('DocVectors.json').read())
# randvectors = json.loads(open('randvectors.json').read())
# featureDict = json.loads(open('featureDict.json').read())
# lenposDocs = 443349
# lennegDocs = 186548

resultsNaive, resultsKNN, comparison, trainVectors, testVectors = Task4(Docs, DocVectors, randvectors_knn, featureDict, lenposDocs, lennegDocs)
print 'Performance of naive bayes classifier:'
print resultsNaive
print 'Performance of KNN classifier:'
print resultsKNN
print 'Comparison by Paired T test:'
print comparison

# with open('trainVectors.json', 'wb') as f:
#    json.dump(trainVectors, f)
# f.close()
#
# with open('testVectors.json', 'wb') as f:
#    json.dump(testVectors, f)
# f.close()
print 'Task 4 Finished!'


print 'Start Bouns Task...'
# DocVectors = json.loads(open('DocVectors.json').read())
# pos_featureInfo = json.loads(open('pos_featureInfo.json').read())
# neg_featureInfo = json.loads(open('neg_featureInfo.json').read())
# features_selected = set(pos_featureInfo).union(set(neg_featureInfo))

nfold = 10
ks = range(0, 60, 10)
ks[0] = 1
ks.insert(1, 5)
ls = [10, 30, 50, 100]
all_results = ParameterTuningKNN(DocVectors, features_selected, nfold, ks, ls)
with open('all_results2.pickle', 'rb') as f:
   all_results2 = pickle.load(f)
f.close()
precisionsTunning, recallsTunning, f1sTunning = TuningEvaluation(all_results2, ls, ks, nfold)
plotEvaluation(precisionsTunning, recallsTunning, f1sTunning, ks, 'k')
maxPrecisions = getMetricMax(precisionsTunning)
maxRecalls = getMetricMax(recallsTunning)
maxF1s = getMetricMax(f1sTunning)
plotEvaluation(maxPrecisions, maxRecalls, maxF1s, ls, 'l')
ks = range(0, 20, 5)
ks[0] = 1
ls = [15, 20, 25]

# with open('all_results.pickle', 'wb') as f:
#    pickle.dump(all_results, f)
# f.close()
# with open('all_results.pickle', 'rb') as f:
#    all_results = pickle.load(f)
# f.close()
print 'Bonus Task Finished!'



