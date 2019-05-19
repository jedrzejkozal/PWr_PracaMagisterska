from StatisticalAnalysis import *

#results from datasets
#every row is results from difrrent folds
#rows are in order: ReNet, ModifiedReNet, Conv
x_ray = np.array([[0.9129692830730217, 0.9087030718757838, 0.9266211600026987, 0.9264957266995031, 0.9205128206147088],
                [0.9325938570621477, 0.9291808877788713, 0.9308873724205096, 0.9247863249901014, 0.938461538665315],
                [0.0, 0.0, 0.0, 0.0, 0.0]])
print("x_ray = ", x_ray)

flowers = np.array([[0.5323325635103926, 0.5219399538106235, 0.5526011560693641, 0.5549132947976878, 0.5331010444654405],
                [0.5554272517321016, 0.5369515011547344, 0.5399, 0.5930635838150289, 0.5435540062763689],
                [0.0, 0.0, 0.0, 0.0, 0.0]])

#avrage accuracy across folds
x_ray_avrg = np.mean(x_ray, axis=1).T
print("x_ray_avrg = ", x_ray_avrg)

#in rows are avrg results on dataset for given algorithm
acc_avrg = np.array([[1, 10, 1, 1],
                    [20, 1, 20, 20],
                    [50, 50, 50, 50]])

analysis = StatisticalAnalysis()
statistic, pvalue, posthoc = analysis.testHypothesis(acc_avrg)
