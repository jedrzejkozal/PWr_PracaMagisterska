

class Logger(object):
    def save_logs(self, results, hyperparams_names):
        formated = self.__get_formated_results(results, hyperparams_names)
        self.__dump_logs(formated)


    def __get_formated_results(self, results, hyperparams_names):
        formated = ""
        for test_results, hyperparams_values in results:
            formated = formated + self.__format_single_line(test_results,
                                                hyperparams_values,
                                                hyperparams_names)
        return formated


    def __format_single_line(self,
                    test_results,
                    hyperparams_values,
                    hyperparams_names):
        result = ""
        for value, name in zip(hyperparams_values, hyperparams_names):
            result = result + name + " = " + str(value) + ", "
        result = result + "\n" + str(test_results) + "\n"
        return result


    def __dump_logs(self, logs):
        f = open("hyperparams.log", "w")
        f.write(logs)
        f.close()
