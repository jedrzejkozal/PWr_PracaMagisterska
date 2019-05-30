from StatisticalAnalysis import *
from saveTexTable import *
import numpy as np

#results from datasets
#every row is results from difrrent folds
#rows are in order: ReNet, ModifiedReNet, Conv
x_ray = np.array([[0.9129692830730217, 0.9087030718757838, 0.9266211600026987, 0.9264957266995031, 0.9205128206147088],
                            [0.9325938570621477, 0.9291808877788713, 0.9308873724205096, 0.9247863249901014, 0.938461538665315],
                            [0.9445392489433289, 0.955631399520835, 0.9411262798634812, 0.9495726491650964, 0.9589743585668058]])

flowers = np.array([[0.5323325635103926, 0.5219399538106235, 0.5526011560693641, 0.5549132947976878, 0.5331010444654405],
                            [0.5554272517321016, 0.5369515011547344, 0.5399, 0.5930635838150289, 0.5435540062763689],
                            [0.7066974595842956, 0.6466512702078522, 0.6878612716763006, 0.6034682080924856, 0.6445993034127975]])

fashion_mnist = np.array([[0.8707142857142857, 0.8708571428571429, 0.8611, 0.8694285713604518, 0.8736428572109768],
                            [0.7, 0.7, 0.7, 0.7, 0.7],
                            [0.9, 0.9, 0.9, 0.9, 0.9]])

natural_images = np.array([[0.846710050657704, 0.8277858175693131, 0.841304347826087, 0.8511256354393609, 0.8474945533769063],
                            [0.7715112075629824, 0.7532561505065123, 0.741304347826087, 0.7378358750907771, 0.7763253449527959],
                            [0.9501084598698482, 0.9442836468023088, 0.9318840579710145, 0.9419026870007262, 0.9070442992011619]])

#avrage accuracy across folds
x_ray_avrg = np.mean(x_ray, axis=1)
flowers_avrg = np.mean(flowers, axis=1)
fashion_avrg = np.mean(fashion_mnist, axis=1)
natural_avrg = np.mean(natural_images, axis=1)
print("x_ray_avrg = ", x_ray_avrg)
print("flowers_avrg = ", flowers_avrg)
print("fashion_avrg = ", fashion_avrg)
print("natural_avrg = ", natural_avrg)

#in rows are avrg results on dataset for given algorithm
acc_avrg = np.array([[1, 10, 1, 1],
                    [20, 1, 20, 20],
                    [50, 50, 50, 50]])

acc_avrg = np.vstack([x_ray_avrg, flowers_avrg, fashion_avrg, natural_avrg])
print(acc_avrg.T)

table_cross_validation = [[' ', 'ReNet', 'modif ReNet', 'conv'],
                ['\makecell{Chest X-Ray\\\\ Images (Pneumonia)}'] + list(acc_avrg[0]),
                ['\makecell{Flowers Recognition}'] + list(acc_avrg[1]),
                ['\makecell{Fashion MNIST}'] + list(acc_avrg[2]),
                ['\makecell{Natural Images}'] + list(acc_avrg[3])]

save_tex_table(table_cross_validation, 'cross_validation')

analysis = StatisticalAnalysis()
statistic, pvalue, posthoc = analysis.testHypothesis(acc_avrg.T)

print("statistic = ", statistic)
print("pvalue = ", pvalue)
print(posthoc)


table_stats = [['wartość_statystki_Friedmana', statistic],
                ['p-wartość', pvalue]]
save_tex_table(table_stats, 'table_stats')

table_post_hoc = [[' ', 'ReNet', 'modif ReNet', 'conv'],
                    ['ReNet'] + list(posthoc[0]),
                    ['modif ReNet'] + list(posthoc[1]),
                    ['conv'] + [posthoc[0][2], posthoc[1][2], 1]]

save_tex_table(table_post_hoc, 'table_post_hoc')
