import os
import time
import json
import matplotlib.pyplot as plt

class SaveResults(object):

    def __init__(self, history, path):
        self.localtime = self.get_formated_time()

        self.dirname = path + "/results/"
        self.create_dir()

        self.save_acc_plot(history)
        self.save_loss_plot(history)
        self.save_hist(history)


    def get_formated_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        localtime = localtime.replace(' ', '_')
        print("localtime: ", localtime)
        return localtime


    def create_dir(self):
        try:
            os.mkdir(self.dirname)
        except FileExistsError:
            pass


    def save_acc_plot(self, history):
        plt.clf()
        self.prepare_acc_plot(history)
        filename = self.dirname + self.localtime + "_accuracy.png"
        plt.savefig(filename)


    def save_loss_plot(self, history):
        plt.clf()
        self.prepare_loss_plot(history)
        filename = self.dirname + self.localtime + "_loss.png"
        plt.savefig(filename)


    def prepare_acc_plot(self, history):
        plt.plot(history.history['categorical_accuracy'])
        plt.plot(history.history['val_categorical_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')


    def prepare_loss_plot(self, history):
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')


    def save_hist(self, history):
        filename = self.dirname + self.localtime + ".json"
        with open(filename, 'w') as f:
            json.dump(history.history, f)
