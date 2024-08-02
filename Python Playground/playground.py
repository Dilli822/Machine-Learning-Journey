

# # data types in python
# # integer, float, string, boolean, None
# floatt = 34.344
# integr = 23
# name = "dilli"
# isValid = True 
# isDied = False 
# tranformer = None 

# # function in python
# def funtion():
#     return "Hello"

# result = funtion()
# print(result)

# def take(i):
#     return i

# # input = input("enter :  ")
# take(input)



# def printhe(*args):
#     return args

# # the arguments are collected as tuples since tuples are ordered, non mutable sequences of variables 
# r = printhe("hello",True, 23, 45.55, None, 234343, 345454, False,)

# printhe("hello",True, 23, 45.55, None)
# print(r)


# def printr(*args, **kwargs):
#     print(args)
#     print(kwargs)

# y = printr(True, "hello world", age = 25, name="dilli hang rai")
# print(y)

# # with statement is used for resource management in python 
# # as keyword is used to create a alias , alias is nothing but a naming converntion or the keyword aleternative name?
# with open("h.txt", 'r') as f:
#     data = f.read()
#     print(data)


def example(*args):
    for i in args:
        print(i)
        
example(1,2,3,"sdsds", 232.2323,True, False)
# print(example(1,2,3,"sdsds", 232.2323,True, False))

# kwargs gives us a list 
def example1(**kwargs):
    for i in kwargs:
        print(i)

example1(fullname="dilli", age = 25, weight=65.55)

with open("h.txt", "r") as file:
    data = file.read()
    for i in data:
        print(i)
    print(data)

# special data types list, tuple, set, dict

# dict key value, mutable
dict_example = { "name": "dilli", "age": 25, "isSingle": True}
print(type(dict_example))
for i in dict_example:
    print(i , dict_example[i])

tup = ("gee",)
print(type(tup))
for i in tup:
    print(i)


tup = ("sds", 45, True)
print(tup[0])
print(tup[0])

lis = ["bello"]
print(lis[0])
lis[0] = "hello"
print(lis)

# sets - unordered and mutable
seteg = { 343,  3434, 654.45}
print(type(seteg))

seteg.add("hello")
print(type(seteg))
print(seteg)

seteg.add("agi")
print(seteg)