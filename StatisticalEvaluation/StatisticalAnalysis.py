import scipy.stats as stats
import numpy as np
import scikit_posthocs as sp

from nemenyi_test import NemenyiTestPostHoc
from scipy.stats import friedmanchisquare, rankdata, norm

class StatisticalAnalysis:

    def __init__(self):
        pass


    def testHypothesis(self, errorTable):

        #please forgive me, stats.f_oneway dosn't accept list of lists or tuple
        a = errorTable[0]
        b = errorTable[1]
        c = errorTable[2]

        statistic, pvalue = stats.friedmanchisquare(a, b, c)
        self.printNullHypothesisResults(statistic, pvalue)
        posthoc = self.evaluate(pvalue, errorTable)
        print("="*40)

        return statistic, pvalue, posthoc


    def printNullHypothesisResults(self, statistic, pvalue):
        print("="*40)
        print("Comparison of extractors over multiple datasets result")
        print("statistic: ", statistic)
        print("pvalue: ", pvalue)


    def evaluate(self, pvalue, errorTable):
        if self.evaluateH0Hypothesis(pvalue):
            print("Post hoc testing skipped due to not rejected ANOVA H0 hypothesis (u1 = u2 = u3)")

        return self.doPostHocTesting(errorTable)


    def evaluateH0Hypothesis(self, pvalue):
        if pvalue < 0.04:
            print("Rejecting H0: no evidence to support hypothesis")
            return False
        elif pvalue > 0.06:
            print("Fail to reject H0: strong indication, that hypothesis is in line with data")
            return True
        else:
            print("No conclusive result")
        return False


    def doPostHocTesting(self, errorTable):
        print('\n\n')
        #print("errorTable: ", errorTable)
        #print("errorTable[0]: ", errorTable[0])
        #print("errorTable[0][0]: ", errorTable[0][0])

        x = np.asarray([tuple(errorTable[0]),
            tuple(errorTable[1]),
            tuple(errorTable[2])])
        #x = np.transpose(x[:, 0, :])
        #res = sp.posthoc_nemenyi(x)
        #print("x: ", x)
        #print("x[0]: ", x[0])

        print("friedmanchisquare: ", friedmanchisquare(x[0], x[1], x[2]))
        nemenyi = NemenyiTestPostHoc(x)
        meanRanks, pValues = nemenyi.do()
        print("meanRanks: ", meanRanks)
        print("pValues: \n", pValues)

        #print("res:")
        #print(res)
        return str(pValues)
