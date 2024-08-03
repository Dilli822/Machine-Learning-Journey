

# # # data types in python
# # # integer, float, string, boolean, None
# # floatt = 34.344
# # integr = 23
# # name = "dilli"
# # isValid = True 
# # isDied = False 
# # tranformer = None 

# # # function in python
# # def funtion():
# #     return "Hello"

# # result = funtion()
# # print(result)

# # def take(i):
# #     return i

# # # input = input("enter :  ")
# # take(input)



# # def printhe(*args):
# #     return args

# # # the arguments are collected as tuples since tuples are ordered, non mutable sequences of variables 
# # r = printhe("hello",True, 23, 45.55, None, 234343, 345454, False,)

# # printhe("hello",True, 23, 45.55, None)
# # print(r)


# # def printr(*args, **kwargs):
# #     print(args)
# #     print(kwargs)

# # y = printr(True, "hello world", age = 25, name="dilli hang rai")
# # print(y)

# # # with statement is used for resource management in python 
# # # as keyword is used to create a alias , alias is nothing but a naming converntion or the keyword aleternative name?
# # with open("h.txt", 'r') as f:
# #     data = f.read()
# #     print(data)


# def example(*args):
#     for i in args:
#         print(i)
        
# example(1,2,3,"sdsds", 232.2323,True, False)
# # print(example(1,2,3,"sdsds", 232.2323,True, False))

# # kwargs gives us a list 
# def example1(**kwargs):
#     for i in kwargs:
#         print(i)

# example1(fullname="dilli", age = 25, weight=65.55)

# with open("h.txt", "r") as file:
#     data = file.read()
#     for i in data:
#         print(i)
#     print(data)

# # special data types list, tuple, set, dict

# # dict key value, mutable
# dict_example = { "name": "dilli", "age": 25, "isSingle": True}
# print(type(dict_example))
# for i in dict_example:
#     print(i , dict_example[i])

# tup = ("gee",)
# print(type(tup))
# for i in tup:
#     print(i)


# tup = ("sds", 45, True)
# print(tup[0])
# print(tup[0])

# lis = ["bello"]
# print(lis[0])
# lis[0] = "hello"
# print(lis)

# # sets - unordered and mutable
# seteg = { 343,  3434, 654.45}
# print(type(seteg))

# seteg.add("hello")
# print(type(seteg))
# print(seteg)

# seteg.add("agi")
# print(seteg)

# integers, float, strings, set, tuples, list, dictionaries, boolean, None
a = None
print(a)

# list 
list = ["dilli", 34, 343.3434, True, None, False, ("hello", "world"), {"this is", "set"}, {"name": "dilli", "datatype": "dict"}, ["this is a list"]]
# print(list)

# for i in list:
#     print(i)

# print(len(list))

# print(type(list[6]))

# for i in list:
#     print(i)
    
# accessing elements
# print(list[0])
# print(list[0:5:2])

# # skip 3 index
# print(list[::3])
# # igonre last 3
# print(list[:3])
# # ignore first 3
# print(list[3:])

# print(list[:4])

def argss(*args):
    return args 

a = argss("hello", 232, 45.454, None, False, True)
# print(a)

def kwargss(**kwargs):
    return kwargs

k = kwargss(age = 25, name = "dilli")
# print(k)


list = ["dilli", 12, True, None, 34.34]
# print(list[2:4])
# print(list[:3])
# print(list[3:])

# # from the start to the end
# print(list[4:])
# # from the start to the end
# print(list[:3])


# print(list[:3])

# print(list[3:])

# print(list[2:4:1])

tuple = ("dilli", "hang", 12, 343)

# for i in tuple:
#     print(tuple)
    
# print(tuple[:2])
# print(tuple[2:])

# print(tuple[::3])
# print(tuple[2::])


# for i in range(100):
#     print(i)
    
# count = 0
# while count < 5:
#     print(count)
#     count += 1

# price = 232.23
# while price < 303.343:
#     print(price)
#     price += 12.344
    
# person = { "name": "dilli", "role": "CS student", "age": 25}
# # for i,v in person.items():
# #     print(f" {i}:{v}")
 
# # for a, b in person.items():
# #     print(f"{a}: {b}")

# for i in person:
#     print(i)
    
# for i in range(23):
#     print(i)
    
# count = 5
# while count < 15:
#     print(count)
#     count = count + 3
    
# for i in range(5):
#     for j in range(10):
#         print(i,j)