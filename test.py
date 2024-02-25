my_dict = {'a': 1, 'b': 2, 'c': 3, 'd': 4}

stop_value = 'c'

# 定义一个生成器函数，用于检查是否达到停止值
def stop_iteration(stop_value):
    for key in my_dict:
        yield key
        if key == stop_value:
            return

# 创建一个迭代器
my_iterator = stop_iteration(stop_value)

# 使用迭代器遍历字典
for key in my_iterator:
    print(my_dict[key])
