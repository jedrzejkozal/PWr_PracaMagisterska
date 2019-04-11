

def cartesian_product(args):
    if type(args[0]) is not list:
        raise TypeError("recived arg is not tuple of lists")
    if len(args) == 1:
        return list(args)
    return compute_cartesian_product(args)


def compute_cartesian_product(args):
    result = []
    for arg in args[::-1]:
        if len(result) == 0:
            initize_results(result, arg)
        else:
            generate_combinations_from_args(result, arg)
    return result


def initize_results(result, arg):
    for i in range(len(arg)):
        result.append([arg[i]])


def generate_combinations_from_args(result, arg):
    extend_list_n_times(result, len(arg)-1)
    segment_len = len(result) // len(arg)
    for i in range(len(result)):
        result[i].insert(0, arg[i // segment_len])


def extend_list_n_times(arg_list, n):
    orginal_list_len = len(arg_list)
    for i in range(n):
        add_orginal_list_copy(arg_list, orginal_list_len)


def add_orginal_list_copy(arg_list, orginal_list_len):
    for j in range(orginal_list_len):
        arg_list.append(list_deep_copy(arg_list[j]))


def list_deep_copy(arg):
    result = []
    for k in range(len(arg)):
        result.append(arg[k])
    return result
