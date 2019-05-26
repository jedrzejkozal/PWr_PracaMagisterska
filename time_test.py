from ReNet.Utils.LoadDataset.LoadXray import *
from ReNet.Utils.LoadDataset.LoadFlowers import *
from ReNet.Utils.LoadDataset.LoadNaturalImages import *
from ReNet.Utils.LoadDataset.LoadFashionMnist import *

from Models.xray import *
from Models.flowers import *
from Models.natural_img import *
from Models.fashion_mnist import *

def do_tests(loaders, reNet_models, modif_reNet_models):
    for loader, reNet_model, modif_reNet_model in zip(loaders, reNet_models, modif_reNet_models):
        x, y = loader.load()
        x, y = x[:100], y[:100]

        avrg_epoch_time = measure_traning_time(reNet_model, x, y)
        print("avrg_epoch_time = ", avrg_epoch_time)

def measure_traning_time(model, x_train, y_train):
    timer = TimerCallback()
    batch_size = 100
    history = model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=50,
        callbacks=[timer])
    results = sut.get_results()
    return np.mean(results, axis=0)


loaders = [LoadXray(), LoadFlowers()] #, LoadNaturalImages(), LoadFashionMnist()]
reNet_models = [get_xray_reNet(), get_flowers_reNet()] #, get_natural_img_reNet(), get_fashion_mnist_reNet()]
modif_reNet_models = [get_xray_modif_reNet(), get_flowers_modif_reNet()] #, get_natural_img_reNet, get_fashion_mnist_modif_reNet()]

do_tests(loaders, reNet_models, modif_reNet_models)
