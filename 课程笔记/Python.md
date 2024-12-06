## Python

### ==试卷错题==

- ```python
  list('[1,2,3]') 
  # 结果为['[', '1', ',', '2', ',', '3', ']']
  
  x=(y=z+1) # Error！这是非法语句
  ```

- python中支持复数的表示：`3+4j`是合法的数字类型

- ```python
  re.match() # 只匹配字符串的开始位置
  re.search() # 匹配整个字符串，知道找到一个匹配
  ```

- ```python
  str.join(iterable) # 将iterable变量的每一个元素后增加一个str字符串
  str.len() # 长度
  str.count(char,strat,end) # 统计某个字符出现的次数
  str.capitalize() # 第一个字母大写，其余小写
  str.title() # 所有英文单词首字母大写，其余小写
  str.lower() # 所有大写字母转换为小写字母
  str.upper() 
  str.swapcase() # 大小写字母同时进行互换
  str.maketrans() # 生成翻译转换表
  str.translate() # 翻译字符串
  str.format() # 用{}作为占位符，用format()中的参数按顺序替换
  
  # 字符串位置
  str.index(sub,start,end) # 与find()同，但若无则会报错
  str.rindex()
  str.find(sub,start,end) # 查找字符串中指定的子字符串sub第一次出现的位置，可指定查找范围，若无则返回-1
  str.rfind(sub,start,end)
  str.center(width,'fc') # 返回一个长度为width,中间为str,两边用fc填充的字符串,若长度小则返回原字符串
  str.ljust(width,'fc') # 返回一个原字符串左对齐，并用fc填充的字符串 
  str.rjust(width,'fc')
  str.max()
  str.min()
  str.replace(old,new,count) # 将old替换成new,次数不超过count次
  
  # 字符串判断
  str.isalnum() # 判断是否由字母和数字组成
  str.isalpha() # 判断是否只由字母组成
  str.isdigit() # 判断是否只由数字组成，不包含中文数字，但包含①②... 
  str.isdecimal() # 判断是否只包含十进制字符
  str.islower() # 判断是否全由小写字符组成
  str.isupper()
  str.isnumeric() # 判断字符串是否只由数字组成，包含中文数字，也包含①②... 
  str.isspace() # 判断是否只由空格组成
  str.istitle() # 判断是否所有单词的首字母大写，其他全为小写
  str.startswith(suffix,start,end) # 判断是否以指定字符或子串开头，可指定范围
  str.endswith(suffix,start,end)
  
  # 
  str.encode(encoding) # 以指定的编码格式编码字符串，默认为utf8
  str.decode(encoding) # 以指定的编码格式解码字符串，默认为字符串编码
  str.strip(char) # 去除开头和结尾指定的字符，默认为空格或换行符
  str.lstrip()
  str.rstrip()
  str.splitlines() # 按照\n,\r,\r\n等进行分割
  str.split()  # 拆分字符串，返回拆分后的列表
  str.rsplit() # 从右边开始分割
  str.zfill(width) # 相当于rjust(width,'0')
  str.expandtabs(tabsize) # 将\t替换为一定数量的空格，默认为8
  ```

- ```python
  str.encode() # 一个中文字符是三个字节，一个英文字符一个字节
  ```

- ```python
  tup = () # 创建一个空元祖
  tup = (1,) # 单元素元祖的表示
  ```

- ```python
  sorted([1,2,3],reverse=True)==reversed([1,2,3])
  # False
  ```

- ```python
  # 与 %H:%M:%S 等价的写法是 %T
  # 与 %Y-%m-%d 等价的写法是 %F
  # 与 %H:%M 等价的写法是 %R
  ```

- ```python
  """
  python 字符串前加 r 表示原始字符串（b表示转为bytes
  类型，f表示格式化字符串）
  """
  str1 = "Hello\nworld"
  str2 = r"Hello \n world"
  print(str1)
  """
  Hello
  World
  """
  print(str2) # Hello \n World
  ```


  - 面向对象的特点：

      封装：隐藏对象的属性和实现细节，只对外提供必要的方法。相当于将"细节封装起来",只对外暴露“相关调用方法”。通过私有属性、私有方法的方式实现封装。

      继承：①如果一个新类继承自一个设计好的类，就直接具备了已有类的特征，就大大降低了工作难度，已有的类，我们称为"父类或基类"，新的类，我们称为“子类或派生类”。②继承可以让子类具有父类的特性，提高了代码的重用性。③Python支持多重继承，一个子类可以继承多个父类

      多态：多态(polymorphism)是指同一个方法调用，由于对象不同可能会产生不同的行为。
      ==(注意：①多态是方法的多态，属性没有多态 ②多态的存在有两个必要条件：继承、方法重写)==

- ```python
  def decorator_a(func):
      print ('Get in decorator_a')
      def inner_a(*args, **kwargs):
          print ('Get in inner_a')
          return func(*args, **kwargs)
      return inner_a
  
  def decorator_b(func):
      print ('Get in decorator_b')
      def inner_b(*args, **kwargs):
          print ('Get in inner_b')
          return func(*args, **kwargs)
      return inner_b
  
  @decorator_b
  @decorator_a
  def f(x):
      print ('Get in f')
      return x * 2
  
  """
  Get in decorator_a
  Get in decorator_b
  Get in inner_b
  Get in inner_a
  Get in f
  """
  ```

- ```python
  def is_palindrome(word):
      # 过滤掉非英文字母和数字的字符，并转换为小写
      word = ''.join(char.lower() for char in word if char.isalnum())
  
      # 判断是否是回文序列
      return word == word[::-1]
  ```

- ```python
  file_author_dict = {"file1.txt": "Author1", "file2.txt": "Author2", "file3.txt": "Author1", "file4.txt": "Author3"}
  
  # 创建一个空的字典用于存储转换后的结果
  author_file_dict = {}
  
  # 遍历原始字典
  for filename, author in file_author_dict.items():
      # 如果作者已经在新的字典中，将文件名添加到其值中（列表形式）
      if author in author_file_dict:
          # 如果值是字符串，将其变为列表
          if isinstance(author_file_dict[author], str):
              author_file_dict[author] = [author_file_dict[author]]
          author_file_dict[author].append(filename)
      else:
          # 如果作者不在新字典中，创建一个新的键值对
          author_file_dict[author] = filename
  
  # 打印结果
  for author, files in author_file_dict.items():
      print(f"{author}: {files}")
  
  ```

- ```python
  locals() # 查看当前作用域内所有全局变量和值的字典
  globals()
  ```

- ```python
  def demo():
      for i in range(4):
          yield i
  
  g = demo()
  g_a = (i for i in g)
  g_b = (i for i in g_a)
  
  print(list(g))
  print(list(g_a))
  print(list(g_b))
  """
  [0, 1, 2, 3]
  []
  []
  """
  ```

### 绪论

#### 1.python解释器

(1)CPython：C语言开发，提示符是`>>>`

(2)IPython：基于CPython，提示符是`ln[序号]:`

(3)PyPy

(4)Jython：运行在Java平台上

(5)IronPython：运行在.Net上面

### 基础语法知识

#### 1.输入输出

(1)输出：print()接受多个字符串用逗号隔开，依次打印每个字符串，遇到逗号输出空格
(2)输入：input()输入的永远是一个字符串

#### 2.标识符

(1)标识符有如下特定的规则：

①区分大小写
②第一个字符必须是字母、下划线，其后的字符是字母、数字、下划线
③不能使用关键字
④以双下划线开头和结尾的名称通常由特殊含义

#### 3.对象

(1)python中一切皆对象

(2)每个对象由标识、类型、值组成：

①标识用于唯一标识对象，通常对应于对象在计算机内存中的地址。使用内置函数id(obj)可以返回对象的标识
②类型用于表示对象存储的“数据”的类型。类型可以限制对象的取值范围以及可执行的操作，可使用type(obj)获得对象的所属类型
③值表示对象所存储的数据的信息。使用print(obj)可打印出值

(3)对象的本质：一个内存块，拥有特定的值，支持特定类型的相关操作

> 在静态类型语言中，变量是一块具有确定类型的内存的名字，一旦定义了该变量，则在销毁该变量前，其变量名始终指向这块内存。与静态类型语言不同，Python 是一个动态类型语言，变量名仅是一个对象的名字，并不是占据一块内存的那个对象本身，一个变量名可以随时指向不同的对象，直到程序运行时，才能根据其指向的对象知道该对象的数据类型

#### 4.赋值

(1)

```python
x, *y = 99, 88, 77
print(y) # [88, 77]
```

(2)

```python
a = 99
b = 88
b, a = a, a+b
print(a) # 187
print(b) # 99
```

(3)

```python
x=[99,88]
y=x
y.append(77)
print(x) # [99,88,77]
print(y) # [99,88,77]
```

#### 5.引用

==搞懂Python中的引用!==

- 变量：
  ①变量名本身是没有类型的，类型只存在对象中，变量只是引用了对象
  ②变量和对象之间的关系为引用
- 对象：
  ①对象是有类型的
  ②对象是分配的一块内存空间，来表示它的值
  ③每一个对象都具有两个标准的头部信息：类型标志符（标识对象的类型）和引用计数器（用来决定对象是不是进行回收）
- 引用：
  ①在Python中，从变量到对象的连接，称为引用
  ②引用是一种关系，以内存中的指针形式实现
  ③赋值操作时，自动建立变量和对象之间的关系，即引用
- 垃圾回收机制：（https://blog.csdn.net/TFATS/article/details/129859219）

==浅拷贝和深拷贝！！==（https://www.runoob.com/w3cnote/python-understanding-dict-copy-shallow-or-deep.html）

#### 6.编码

==(必考)==

#### 7.数据类型概要

![image-20231226170742760](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231226170742760.png)

- Basic/Primitive、Container/Collections、None、User Defined - class、bytes

- Number、String、Boolean、None、list、tuple、dict、set、bytes

- 可变对象：字典、列表、集合、自定义对象等
  不可变对象：数字、布尔值、字符串、元祖、None等
  (相比于可变对象，不可变对象的访问速度更快)
  
  > ==python难点（一定要弄懂！！！）==
  >
  > https://blog.csdn.net/weixin_42143579/article/details/109011604

- 类型转换：
  ①隐式类型转换（向上转换，无需显式）
  ②显式类型转换

#### 8.数据类型具体

- 数字Number：Integer、Float、Complex（复数）
  ①二进制数：使用bin()，前缀为"0b"
  ②八进制数：使用oct()，前缀为"0o"
  ③十六进制数：使用hex()，前缀为"0x"

- 字符串String：有序序列，不可修改（掌握所有的字符串方法!）

  ```python
  # 
  str.join(iterable) # 将iterable变量的每一个元素后增加一个str字符串
  str.len() # 长度
  str.count(char,strat,end) # 统计某个字符出现的次数
  str.capitalize() # 第一个字母大写，其余小写
  str.title() # 所有英文单词首字母大写，其余小写
  str.lower() # 所有大写字母转换为小写字母
  str.upper() 
  str.swapcase() # 大小写字母同时进行互换
  str.maketrans() # 生成翻译转换表
  str.translate() # 翻译字符串
  str.format() # 用{}作为占位符，用format()中的参数按顺序替换
  
  # 字符串位置
  str.index() # 与find()同，但若无则会报错
  str.rindex()
  str.find(sub,start,end) # 查找字符串中指定的子字符串sub第一次出现的位置，可指定查找范围，若无则返回-1
  str.rfind(sub,start,end)
  str.center(width,'fc') # 返回一个长度为width,中间为str,两边用fc填充的字符串,若长度小则返回原字符串
  str.ljust(width,'fc') # 返回一个原字符串左对齐，并用fc填充的字符串 
  str.rjust(width,'fc')
  str.max()
  str.min()
  str.replace(old,new,count) # 将old替换成new,次数不超过count次
  
  # 字符串判断
  str.isalnum() # 判断是否由字母和数字组成
  str.isalpha() # 判断是否只由字母组成
  str.isdigit() # 判断是否只由数字组成，不包含中文数字，但包含①②... 
  str.isdecimal() # 判断是否只包含十进制字符
  str.islower() # 判断是否全由小写字符组成
  str.isupper()
  str.isnumeric() # 判断字符串是否只由数字组成，包含中文数字，也包含①②... 
  str.isspace() # 判断是否只由空格组成
  str.istitle() # 判断是否所有单词的首字母大写，其他全为小写
  str.startswith(suffix,start,end) # 判断是否以指定字符或子串开头，可指定范围
  str.endswith(suffix,start,end)
  
  # 
  str.encode(encoding) # 以指定的编码格式编码字符串，默认为utf8
  str.decode(encoding) # 以指定的编码格式解码字符串，默认为字符串编码
  str.strip(char) # 去除开头和结尾指定的字符，默认为空格或换行符
  str.lstrip()
  str.rstrip()
  str.splitlines() # 按照\n,\r,\r\n等进行分割
  str.split()  # 拆分字符串，返回拆分后的列表
  str.rsplit() # 从右边开始分割
  str.zfill(width) # 相当于rjust(width,'0')
  str.expandtabs(tabsize) # 将\t替换为一定数量的空格，默认为8
  ```

- 列表List：有序序列，可修改

  ```python
  list.append(obj)
  list.extend(list)
  list.pop()
  list.clear() # 清空列表
  list.remove(obj)
  list.insert(index,obj)
  list.reverse() # 返回值为None
  list.sort()
  list.count(obj)
  list.index(obj)
  ```

  △：思考

  ```python
  nums = [1, 2, 2, 2, 3, 4, 2]
  for num in nums:
      print(num)
      if num == 2:
          nums.remove(2)
  print(nums)
  """
  1
  2
  2
  4
  2
  [1, 3, 4, 2]
  """
  ```

- 元祖tuple：有序序列，不可修改

  ```python
  tup = () # 创建一个空元祖
  tup = (1,) # 单元素元祖的表示
  ```

  △：序列与其他类型的对比

- 字典Dictionary：键值对，原先是无序排列，python3.6开始有序排列

  ```python
  dict.fromkeys()
  dict.update()
  dict.copy()
  dict.setdefault()
  dict.clear()
  dict.pop()
  dict.get()
  dict.items()
  dict.keys()
  dict.values()
  ```

  ```python
  for key,value in dict.items(): # key-value遍历字典
  ```

  ```python
  file_author_dict = {"file1.txt": "Author1", "file2.txt": "Author2",
                      "file3.txt": "Author1", "file4.txt": "Author3"}
  
  author_file_dict = {}
  
  # 直接遍历原始字典，只会遍历 key
  for filename in file_author_dict:
      print(filename)
      
  """
  file1.txt
  file2.txt
  file3.txt
  file4.txt
  """
  ```

- 集合Set：
  ①集合是无序的唯一元素的集合，没有相同元素
  ②我们使用大括号来定义集合。默认情况下，这些大括号用于字典，但我们在这些大括号中使用逗号分隔元素列表，则将其视为集合
  ③我们也可以使用set()函数创建集合
  ④集合中的项目没有索引位置
  ==⑤集合是根据其元素的哈希值存储元素的，所以无法计算哈希值的对象不能作为集合的元素。例如，list 对象是无法计算哈希值的，所以不能作为集合的元素。（可哈希的就是不可变元素）==
  
  ==⑥集合是可变的还是不可变的？==
  
- bytes类型：

  > bytes 以字节序列的形式（二进制形式）来存储数据，至于这些数据到底表示什么内容（字符串、数字、图片、音频等），完全由程序的解析方式决定。如果采用合适的字符编码方式（字符集），字节串可以恢复成字符串；反之亦然，字符串也可以转换成字节串

  ```python
  # 通过构造函数创建空bytes
  b1 = bytes()
  
  # 通过空字符串创建空bytes
  b2 = b''
  
  # 通过b前缀将字符串转换成bytes
  b3 = b'http://c.biancheng.net/python/'
  
  print(b1)  # b''
  print(b2)  # b''
  print(b3)  # b'http://c.biancheng.net/python/'
  
  # 为bytes()方法置顶字符集
  b4 = bytes('你好', encoding='utf8')
  
  print(b4)  # b'\xe4\xbd\xa0\xe5\xa5\xbd'
  print('你好'.encode('utf8'))  # b'\xe4\xbd\xa0\xe5\xa5\xbd'
  ```
  
  > 一个中文字符是三个字节，一个英文字符一个字节


#### 9.格式化输出

(1)使用%格式化

```python
x = 'looked'
print("Misha %s and %s around." % ('walked', x))
# Misha walked and looked around

print("The value of pi is %6.4f" % 3.14159)
# The value of pi is 3.1416
```

(2)使用format()格式化

```python
"here {} then also {}".format('sth1','sth2')
# here sth1 then also sth2

"{2} {1} {0}".format('directions','the','read')
# "'read' 'the' 'directions'"

"a: {a},b: {b},c: {c}".format(a=1,b='Two',c=12.3)
# "a: 1,b: 'Two',c: 12.3"

"The value of pi is: {0:7.5f}".format(3.141592)
# "The value of pi is 3.14159"
```

(3)f-string

```python
name = 'Ele'
print(f"my name is {name}")

num=3.14159
print(f"{num:{1}.{5}}")
# 3.1416
```

(4)substitute

```python
from string import Template
n1 = 'hello'
n2 = 'geeks'
n = Template('$n3! this is $n4.')
print(n.substitute(n3=n1,n4=n2))
```

> ==浮点数输出规则==
>
> ①f规则
> `a.bf`代表输出的总长度若小于a，则在前面补空格补到总长为a，保留b位小数，四舍五入
> ②直接规则
> `a.b`代表输出的总长度若小于a，则在前面补空格补到总长为a，然后从左往右保留b位数字，四舍五入

#### 10.运算类型

(1)算数运算符（`//`是整除符号）

```python
9//2 = 4
9.0//2 = 4.0
-11//3 = -4
-11.0//3 = -4.0
```

(2)位运算符：对运算数的二进制位进行相应的运算。其中，&(位与)、丨(位或)、^(异或)是对2个运算数的对应二进制位进行运算，而～(取反)、<<(左移位)、>>(右移位)则是对1 个运算数的二进制位进行运算

>①&(位与)运算：只有p 和q 都是1 时，p&q 的结果才是1；如果有一个为0，则结果是0。
>②丨(位或)运算：只要p 和q 有一个是1 时，p|q 的结果就是1；如果p 和q 都为0，则结果是0。
>③^(异或)运算：当p 和q 不同(一个是0，另一个是1)时，p^q 的结果是1，否则结果是0。
>④～(取反)：将每个二进制取反(0 变成1，1 变成0)。例如，对x，~x 的结果相当于-x-1，如22 的补是-23。
>⑤<< (左移)：各二进制位全部左移若干位，高位丢弃，低位补0。
>⑥\>>(右移)：各二进制位全部右移若干位，无符号数，高位补0。

(3)逻辑运算符and、or、not 分别表示逻辑与、逻辑或、逻辑非运算。在逻辑运算中，True、非0 或非空对象就是真(True)，而False、0 或空对象就是假(False)。

```python
if(a in b)
if(a not in b)
if(a is b)
if(a is not b)
```

> 支持`!=`但不支持`!`

#### 11.条件判断

(1)python中没有swich语法，这里介绍一个switch的替代方法：

```python
day = 6
switcher = {
    0: "Sunday",
    1: "Monday",
    2: "Sunday",
}
day_name = swicher.get(day, 'unknown') 
# 'unknown'相当于switch中default中的取值
print(day_name)
```

(2)python3.10增加了match分支结构，相当于switch

```python
def http_error(status):
    match status:
        case 400:
            return "Bad request"
        case 404:
            return "Not found"
        case _:
            return "Something's wrong"
```

(3)while和for也能加else，一但循环正常执行完毕就会执行else语句内的内容（如果break跳出循环了就不会执行）

#### 12.函数

(1)函数执行完毕也没有return语句时，会自动return None

(2)返回值：函数可以同时返回多个值，但其实就是一个tuple；没有return语句的函数也有返回值，返回值是None

(3)常见的参数叫位置参数，即指定好位置的参数

(4)默认参数：必须放在最右边

默认参数在函数被定义时就已经创建，而非在程序运行时！！！（易错题如下）

```python
def add_end(lst=[]):
    lst.append('end')
    return lst

print(add_end()) # ['end']
print(add_end()) # ['end','end']
```

(5)可变参数：见下例

```python
def calc(*numbers):
    sum = 0
    for n in numbers:
        sum = sum + n * n
    return sum

# 注意这里的使用方式
nums = [1,2,3]
calc((1,2,3))
calc(nums[0],nums[1],nums[2])
calc(*nums)
```

(6)关键字参数：见下例，理解为可变键值对参数

```python
def person(name, age, **kw):
    # kw打印出来是字典格式
    print('name:', name, 'age:', age,'other:', kw)
    # 还可进行自定义设计
    if 'city' in kw:
        ...
    if 'job' in kw:
        ...
 
# 使用方式
person('Adam', 45)
person('Adam', 45, city='Beijing')
person('Adam', 45, gender='M', job='Engineer')
```

(7)命名关键字参数：见下例

```python
def person_info(name, age, *, gender, city):
    print(name, age, gender, city)

# 使用方式
person_info('Steve', 22, gender='Male', city='NY')
```

(8)注意点：

- ==默认参数一定要用不可变对象，如果是可变对象，程序运行时会有逻辑错误==
- 要注意定义可变参数和关键字参数的语法：*args是可变参数，接收的是一个tuple；**kw是关键字参数，接收的是一个dict
- 调用函数时，传入可变参数和关键字参数的语法：
  ①可变参数既可以直接传入：func(1, 2, 3)，也可以先组装成list或tuple，再通过\*args传入：func(*(1, 2, 3))
  ②关键字参数既可以直接传入：func(a=1, b=2)，也可以先组装dict，再通过\*\*kw传入：func(**{'a': 1, 'b': 2})

(9)函数说明文档：直接在函数体的最上面，添加三个双引号对进行注释

```python
def func():
"""
there is help context
"""
```

#### 13.作用域

<img src="https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20230927170634970.png" alt="image-20230927170634970"/>

==变量/函数的查找顺序：L->E->G->B（不可逆，例如在global中无法访问local中赋值的变量）==

==△：循环语句、条件语句均不会引起作用域的变化！基本只有函数和类会！==

(1)``globals()``与``locals()``：以字典的形式存储所有全局或局部变量

![](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20230927171350408.png)

```python
# globals() ：以dict的方式存储所有全局变量
def foo():
    print("I am a func")

def bar():
    foo = "I am a string"
    foo_dup = globals().get("foo")
    foo_dup()
    
bar()

# locals()：以dict的方式存储所有局部变量
other = "test"

def foobar():
    name = "MING"
    gender = "male"
    for key, value in locals().items():
        print(key, "=", value)

foobar()
```

(2)闭包closure

- 构成闭包的三个条件==（必考）==：
  ①存在函数嵌套（函数中函数）
  ②内部函数使用了外部函数的变量
  ③外部函数的返回值是内部函数的函数名（变量名也对）

- 闭包案例

  ```python
  # example1
  def deco():
      name = "Ming"
      def wrapper():
          print(name)
      return wrapper
  
  deco()() # 会调用wrapper()
  
  # example2
  def nth_power(exponent):
      # 注意，这里exponent也算外部函数的变量
      def exponent_of(base):
          return base ** exponent
      return exponent_of
  
  square = nth_power(2)
  cube = nth_power(3)
  print(square(2)) # 计算2的平方
  print(cube(3)) # 计算3的立方
  ```

- 闭包的\_\_closure\_\_属性：闭包比普通的函数多了一个\_\_closure\_\_属性，该属性记录着自由变量的地址。当闭包被调用时，系统就会根据该地址找到对应的自由变量，完成整体的函数调用

(3)改变作用域

- ① `global`：将局部变量变为全局变量
  ② `nonlocal`：可以在闭包函数中，引用并使用闭包外部函数的变量

- 任何一层子函数，若直接使用全局变量且不对其改变的话，则共享全局变量的值；一旦子函数中改变该同名变量，则其降为该子函数所属的局部变量（在改变之前就已经是了）

  ```python
  num = 1
  def func():
      num += 1 # 报错！
  ```

-  global可以用于任何地方，声明变量为全局变量（声明时，不能同时赋值）；声明后再修改，则修改了全局变量的值

  ```python
  x = 0
  def outer():
      x = 1
      def inner():
          global x
          x = 2
          print("inner:", x) # 这里的x是第一行的x
  ```

- ==而nonlocal的作用范围仅限于所在子函数的上一层函数中拥有的局部变量，必须在上层函数中已经定义过，且非全局变量，否则报错==

  ```python
  def func():
      num = 1
      def subfunc():
          nonlocal num # 没有这一行则报错！
          num += 1
          return num
      return subfunc
  
  print(func()())
  ```


#### 14.推导式

```python
# 列表推导式
[表达式 for 迭代变量 in 可迭代对象 [if 条件表达式]]

# 字典/集合推导式
{表达式 for 迭代变量 in 可迭代对象 [if 条件表达式]}
```

#### 15.匿名函数

```python
#lambda表达式返回一个函数对象例子
func = lambda x, y : x+y

#func 相当于下面这个函数
def func(x,y):
    return x+y

[(lambda x:x*x)(x) for x in range(1,11)]
```

#### 16.高阶函数

(1)特点

- 函数是对象类型（Object type）的实例
- 可以将函数存储在变量中
- 可以将函数作为参数传递给另一个函数
- 您可以从函数返回函数
- 您可以将它们存储在数据结构中，如哈希表、列表等…

(2)内置高阶函数

```python
sorted() 
"""
list_tuples = [(1, 'byd'), (3, 'xiaopeng'), (2, 'tesla'), (4, 'weilai')]
listed_tuples = sorted(list_tuples, key=lambda x: x[1])
"""

map(func,list) # 返回类型为 <class 'map'>
"""
items = [1,2,3,4,5]
def f(x):
	return x**2
squared = list(map(f,items))
"""

reduce(func,iterable)
"""
def add(x, y):           
    return x + y
sum1 = reduce(add,[1,2,3,4,5]) # 计算列表所有元素之和
"""

filter(func,list) # 返回类型为 <class 'filter'>
"""
def is_odd(n):
	return n % 2 == 1
f_list = filter(is_odd,[1,2,3,4,5,6,7])
"""
```

==△：易错！！！必须得看==

https://nankai.feishu.cn/docx/RjL3d6GD5obvdGxgLmkciMZLnLv

(3)偏函数partial()

```python
from functools import partial

int2 = partial(int, base=2)
print(int2('111000111')) # 455
```

### 装饰器

#### 1.定义

装饰器是一种常见的设计模式，常备用于有切面需求的场景，较为经典的应用有插入日志、增加计时逻辑、增加触发器等

#### 2.种类

(1)不带参数的装饰器

①第一种用法

```python
import time
def func():
    print("Hello world")
    
def time_counter(fn):
    def wrapper():
        start = time.time()
        fn()
        end = time.time()
        return end - start
    return wrapper

# 第一种用法
func = time_counter(func)
```

==△：易错题！！!==

==一定要弄懂[closure_t_c.py](C:\Users\12774\Downloads\closure_t_c.py)里面所有的情况！！！==

```python
"""第一题！！！"""
def count_steps(original_steps=0):
    def wrapper(new_steps):
        nonlocal original_steps  # 添加nonlocal是为了改变变量original_steps的值
        total_steps = original_steps + new_steps
        original_steps = total_steps
        return original_steps
    return wrapper

go = count_steps(8)
# 闭包可以保存当前的运行环境（有记忆）!!!
print(go(2)) # 10
print(go(3)) # 13
print(go(5)) # 18
print(go(5)) # 23

"""第二题！！！"""
def func_2():
    i = 'I am a closure function'
    def wrapper():
        print(i, 'in func_2')
    i = 888
    return wrapper

func_2()() # 888 in func2

"""第三题！！！"""
def func_3():
    list_funcs = []
    for i in range(1, 5):
        def wrapper():
            print(i)

        list_funcs.append(wrapper)
    return list_funcs


func_3()[0]() # 4

def func_4():
    list_funcs = []
    for i in range(1, 5):
        def wrapper(i):
            def inner():
                print(i)

            return inner

        list_funcs.append(wrapper(i))
    return list_funcs


func_4()[0]() # 1
func_4()[1]() # 2
func_4()[2]() # 3
func_4()[3]() # 4
```

②第二种用法

```python
import time
def time_counter(fn):
    def wrapper():
        start = time.time()
        fn()
        end = time.time()
        return end - start
    return wrapper

# 第二种用法
@time_counter
def func():
    print("Hello world")
    
# 函数有参数的情况
import time
def time_counter(fn):
    def wrapper(x):
        start = time.time()
        fn(x)
        end = time.time()
        return end - start
    return wrapper

# 第二种用法
@time_counter
def func(x):
    print(x)
```

(2)带参数的装饰器

```python
def logging(flag):
    def decorator(fn):
        def inner(num1, num2):
            if flag == "+":
                print("正在努力进行加法计算")
            elif flag == "-":
                print("正在努力进行减法计算")
                result = fn(num1, num2)
            return result
        return inner
    return decorator

@logging("+")
def add(num1, num2):
    return num1 + num2

@logging("-")
def sub(num1, num2):
    return num1 - num2

# 解释为add = logging("+")(add)
```

(3)多个装饰器

```python
@decorator_one
@decorator_two
def func():
    pass

# 解释为func = decorator_one(decorator_two(func))
```

例题：

```python
def make_bold(fn):
    def wrapped():
        return "<b>" + fn() + "</b>"
    return wrapped

def make_italic(fn):
    def wrapped():
        return "<i>" + fn() + "</i>"
    return wrapped

@make_bold
@make_italic
def func():
    return "hello world"

# Question: func的返回结果是什么
# Answer: "<b><i>hello world</i></b>"
```

多个带参数函数装饰器：

```python
def make_html_tag(tag, *args, **kwargs):
    def real_decorator(fn):
        css_class = " class='{0}'".format(kwargs("css_class")) if "css_class" in kwargs else ""
        
        def wrapped(*args, **kwargs):
            return "<" + tag + css_class +fn(*args, **kwargs) + "</" + tag + ">"
        return wrapped
    return real_decorator

@make_html_tag(tag="b", css_class="bold_css")
@make_html_tag(tag="i", css_class="italic_css")
def func(x):
    return x

func("Hello world!") # <b class='bold_css'><i class='italic_css'>Hello world!</i></b>
```

(4)类装饰器

```python
class Decorator:
    def __init__(self, fn):
    	print("inside Decorator __init__()")
    	self.fn = fn
    def __call__(self):
        self.fn()
        print("inside Decorator __call__()")
        
@Decorator
def func():
    print("inside func()")
    
func()
"""
打印结果如下：
inside Decorator __init__()
inside func()
inside Decorator __call__()
"""
```

> 多个带参数类装饰器
>
> ![image-20231226181710923](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231226181710923.png)

(5)装饰类

```python
def refac_str(cls):
    def __str__(self):
        return str(self.__dict__)
    
    cls.__str__ = __str__
    return cls

@refac_str
class MyClass:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
a = MyClass(1,2) 
print(str(a)) # {'x': 1, 'y': 2}
```

(6)装饰器的副作用

①属性丢失

![image-20231025160907412](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231025160907412.png)

例如这里可以看到foo函数的\__name__变成了wrapper

②如何抵消装饰器的副作用（注意这里不能完全抵消）
(两种方法：wraps和update_wrapper)

![image-20231025161559958](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231025161559958.png)

(7)装饰器的应用案例：①权限控制 ②计时功能 ③日志 ④利用缓存 ⑤限制接口

==例：利用缓存计算斐波那契数列==
![image-20231025163040830](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231025163040830.png)

### 三器语法

(1)“三器”语法

装饰器、迭代器、生成器

(2)可迭代对象 iterable：每次能返回其一个成员的对象，即实现了\_\_iter\_\_()或\_\_getitem\_\_()协议的对象

- Python提供了两个通用迭代器对象：
  ①序列对象：list，str，tuple
  ②非序列对象：dict，file objects
- 可迭代对象可用于for循环，以及其它需要序列的地方（如zip()、map() ...）
- 使用内置函数 iter()，或者\_\_iter\_\_方法，可将可迭代对象转换成迭代器iterator

(3)迭代器：用来表示一连串数据流的对象，称为迭代器

- 判断题：迭代器与可迭代对象共同的特点是它们都实现了\_\_iter\_\_()的方法
  答案：错误，可迭代对象不一定实现了\_\_iter\_\_()的方法，有可能只实现了\_\_getitem\_\_()的方法

- 迭代器是实现迭代器协议的对象，它包含方法\_\_iter\_\_()和\_\_next\_\_()

  > ①迭代器的\_\_iter\_\_()方法用来返回该迭代器自身，故迭代器必定是可迭代对象
  >
  > ②迭代器的\_\_next\_\_()方法（或将其传给内置函数next()）将逐个返回数据流中的项，当没有数据可用时将引发StopIteration异常

- ==range() 不是迭代器！！！==

(4)生成器 Generator：是一种特殊的迭代器

- 生成器表达式

  ```python
  # 列表推导式
  [i * 2 for i in range(10) if i % 2 == 0]
  # 集合推导式
  {i * 2 for i in "abcd"}
  # 字典推导式
  {k: v for k, v in zip(("one","two","three"), (1,2,3))}
  # 生成器表达式 (得到的不是元组)
  (i * 2 for i in range(10))
  ```

- 生成器函数：使用yield语句的函数或方法

  ①当生成器函数被调用时，它会返回一个名为生成器的迭代器
  ②每个yield会临时暂停处理，记住当前位置执行状态，当该生成器迭代器恢复时，它会从离开位置继续执行，这与普通函数调用(重新开始)差别很大
  ③当yield表达式是赋值语句右侧的唯一表达式时，括号可以省略，即yield语句在语义上等同于yield表达式，但建议总是加上，如右侧代码 `val = (yield i)`
  ==④在未执行next之前，生成器函数内部的一切都不会真正执行！==

- 生成器函数与函数，yield与return的区别

  ①任何包含了yield关键字的函数都是生成器函数
  ②生成器函数是一类用来简化编写迭代器工作的特殊函数
  ③普通的函数计算并返回一个值，而生成器返回一个能返回数据流的迭代器
  ④当函数到达return表达式时，局部变量会被销毁然后把表达式返回给调用者
  ⑤yield和return最大区别：程序执行到yield时，生成器的执行状态会挂起并保留局部变量，在下一次调用生成器\_\_next\_\_()方法的时候，函数会恢复执行

- 生成器案例==（较难，要掌握！！！）==

  ```python
  def add(n, i):
      return n + i
  
  def test():
  	for i in range(4):
          yield i
          
  g = test()
  for n in [1, 10]: # 注意，这里没有range！！！
      g = (add(n,i) for i in g)
      
  print(list(g)) # [20,21,22,23]
  
  # 生成器具有惰性，当没有真正调用时，不会执行生成器里面的内容，所以这里最终的g是2n+i
  ```

  ```python
  def flatten_list(nested):
      if isinstance(nested, list):
          for sublist in nested:
              print("ccc")
              for item in flatten_list(sublist):
                  print("ppp")
                  yield item
                  print("yyy")
      else:
          print("uuu")
          yield nested
          print("xxx")
  
  raw_list = ["a", ["b", "c", ["d"]]]
  g = flatten_list(raw_list)
  print(next(g), "+++")
  print(next(g), "---")
  """
  ccc
  uuu
  ppp
  a +++
  yyy
  xxx
  ccc
  ccc
  uuu
  ppp
  ppp
  b ---
  """
  ```

- 生成器方法：
  ![image-20231101165811285](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231101165811285.png)
  ![image-20231101165822508](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231101165822508.png)

  方法案例：

  ```python
  def echo(value = None):
      print("Execution starts when 'next()' is called for the first time")
      try:
          while True:
              try:
                  value = (yield value)
                  print("******")
              except Exception as e:
                  value = e
      finally:
          print("Don't forget to clean up when 'close()' is called")
          
  generator = echo(1)
  print(next(generator))
  print(next(generator))
  print(generator.send(2))
  print(generator.throw(TypeError, "spam"))
  generator.close()
  """
  Execution starts when 'next()' is called for the first time
  1
  ******
  None
  ******
  2
  spam
  Don't forget to clean up when 'close()' is called
  """
  ```


### 面向对象编程

#### 1.函数式编程和面向对象编程

(1)函数式编程风格的优点：

①形式证明 
②模块化（把一个问题分解成多个小方面）
③组合性（函数重复使用，甚至可以组合已有的函数来组成新的程序）
④易于调试和测试

(2)面向对象：封装、继承、多态

#### 2.类的定义

(1)类的实例化

```python
class Human:
    pass

# Human()称为实例化，h为类Human的实例/对象
h = Human()
```

> 经典类和新式类：
> ①python 2.x默认都是经典类，只有显式继承object类才是新式类
> ②python 3.x移除经典类，全为新式类

(2)类的属性和方法

```python
class Student:
	university = "Nankai University"
    
    def take_class(self):
        print("having class...")
# right
stu = Student()
print(Student.university)
print(stu.university)
stu.take_class()
Student.take_class(stu)

# error
Student.take_class()
```

(3)类的初始化：

- 区分类的属性和实例属性！

  ```python
  class Student:
      # 类的属性
  	university = "Nankai University"
      
      def __init__(self, name, school):
          # 实例属性
          self.name = name
          self.school = school
      
      def take_class(self):
          # 实例方法
          print(f"{self.name} is having {self.school} class...")
          
  gj = Student("Guo Jing", "G School")
  hr = Student("Huang Rong", "T School")
  gj.take_class()
  hr.take_class()
  
  """
  Guo Jing is having G School class...
  Huang Rong is having T School class...
  """
  ```

- 动态添加属性

  ```python
  # 添加实例属性
  gj.unique_skill = "Nine Yin Kongfu"
  print(gj.unique_skill)
  
  # 添加类的属性
  Student.unique_skill = "Nine Yin Kongfu"
  ```

- 属性的查找顺序和修改

  ```python
  class Student:
      # 类的属性
      capacity = 10
      
      def __init__(self, name, school):
          # 实例属性
          self.name = name
          self.school = school
      
      def take_class(self):
          # 实例方法
          print(f"{self.name} is having {self.school} class...")
          
  gj = Student("Guo Jing", "G School")
  print(gj.capacity)
  gj.capacity = 100
  Student.capacity = 60
  print(gj.capacity)
  print(Student.capacity)
  hr = Student("Huang Rong", "T_school")
  print(hr.capacity)
  
  """
  10
  100
  60
  60
  """
  ```

(4)类的方法：类方法、实例方法、静态方法

- 实例方法：没有任何装饰器，有默认self的普通函数

- 类方法：有classmethod装饰的函数，既可以被类调用，也可以被实例调用

  ```python
  class Student:
      # 类的属性
  	capacity = 10
      
      def __init__(self, name, school):
          # 实例属性
          self.name = name
          self.school = school
      
      def take_class(self):
          # 实例方法
          print(f"{self.name} is having {self.school} class...")
          
      @classmethod
      def show_capacity(cls):
          print(f"{cls.capacity}")
  ```

  例子：

  ```python
  class A:
  
      @classmethod
      def f1(cls):
          print(cls)
  
      def f2(self):
          self.f1()
          A.f1()
  
  a = A()
  a.f2()
  A.f2()
  """
  <class '__main__.A'>
  <class '__main__.A'>
  <class '__main__.A'>
  """
  ```

  

- 静态方法：有staticmethod装饰的函数，无法通过self或者cls访问到实例或类中的属性

  ```python
  class Student:
      # 类的属性
  	capacity = 10
      
      def __init__(self, name, school):
          # 实例属性
          self.name = name
          self.school = school
      
      def take_class(self):
          # 实例方法
          print(f"{self.name} is having {self.school} class...")
          
      @staticmethod
      def show_capacity():
          print("xxx")
  ```

(5)类中的绑定方法

- 类中的方法或函数，默认都是绑定给实例使用的
- 绑定方法都有自动传值的功能，传递的值，就是对象本身
- 类调用实例方法，实例方法仅被视为函数，无自动传值这一功能，有几个参数，就必须传递几个参数
- 通过classmethod装饰器，将绑定给实例的方法，绑定到了类
- 如果一个方法绑定谁，在调用该函数时，自动将该调用者当作第一个参数传递到函数中

(6)类的非绑定方法

- 通过staticmethod装饰器，可以解除绑定关系，将一个类中的方法，变成一个普通函数
- 静态方法中参数传递跟普通函数相同，无需考虑自动传参等问题

(7)方法与函数

![image-20231115163758146](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231115163758146.png)

(8)私有变量与私有方法

- 下划线
  ![image-20231115164611330](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231115164611330.png)

- 类的私有属性（因为私有属性不能再类的外部被直接访问，所以这里`xh.__age`被视为了公有变量，这里其实是创建了一个公有属性）
  ![image-20231115170913801](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231115170913801.png)

  私有属性的内外调用：
  ![image-20231115171823057](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231115171823057.png)

- 私有方法（在类外部调用就会报错）
  ![image-20231115172157092](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231115172157092.png)

- 私有属性和方法的外部访问：类的外部可以通过：\_类名\_\_属性名 或者 \_类名\_\_方法名  来直接访问类中的私有属性和私有方法

- 类之间的关系：继承、组合（关联）、依赖
  ①继承：
  ②组合：将一个类的对象封装到另一个类的对象的属性中，就叫组合（一个类的属性是另一个类）
  ③依赖：将一个类的类名或对象当做参数传递给另一个函数被使用的关系就是依赖关系（一个类的方法中用到了另一个类）

#### 3.类的继承

- 多继承：（谁写在前面，优先级就越高）
  ![image-20231122163300150](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231122163300150.png)

  继承顺序：使用`__mro__`来查询（理解C3算法）

  ![image-20231122163919313](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231122163919313.png)

  > ![image-20231122170412152](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231122170412152.png)
  > ![image-20231226192633727](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231226192633727.png)
  
- 方法的重写：子类定义了与父类同名的方法，则子类的方法会覆盖父类方法
  
- `super()`：主要用于解决多重继承问题。直接用类名调用父类方法，在单继承中没有问题，但如果在多继承，则会涉及查找顺序（MRO）、重复调用（钻石继承）等种种问题

  ①`super()`按照C3算法的继承顺序，`super().__init__()`相当于`BaseClass.__init__(self)`

  ②`super()`和类均可调用父类实例方法，区别在于`super()`后跟的方法不需要传self参数，父类调用实例方法，第一个参数需要传self(方法的绑定)

  ```python
  class Penguin(Bird):
      def __init__(self,name,food):
  		super().__init__(name)
          Animal.__init__(self,food)
  ```

  ③`super()`案例==（要掌握！！！）==
  ![image-20231122172955508](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231122172955508.png)

  > 解释：`super()`按照C3算法的继承顺序`C->A->B>Base`，所以A继承至Base但A的`super()`是B而不是Base

  ![image-20231122173640694](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231122173640694.png)

  > 解释：`super().__init__()`相当于`BaseClass.__init__(self)`，所以所有的`self`一直都是同一个（一直都是kj，谁去调的self就是谁）
  
- 类的Mixin设计模式（接口在python中的设计模式）

  ![image-20231129160904034](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231129160904034.png)

  > python中的三种设计模式：==(很有可能会考！！！)==
  >
  > ① 装饰器设计模式 ② Mixin设计模式 ③ 单例模式

#### 4.类的其他

(1)类的多态

- 定义：Python中，不同的对象调用同一个接口，从而表现出不同状态。(Python是动态语言，使用变量无需为其指定具体类型，故多态是其原始特性)

- 条件：继承+重写
  ![image-20231129163021985](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231129163021985.png)

- 多态的约束方法：
  ![](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231129164443308.png)

(2)抽象基类ABC
![image-20231129170320638](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231129170320638.png)

(3)判断类实例的函数
![image-20231129171629145](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231129171629145.png)

(4)元类


- 元类 metaclass（创建类的类）
  ![image-20231129171925197](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231129171925197.png)

- ==（下图右侧例子会考！！打印了什么！！）==
    ![image-20231129172714125](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231129172714125.png)

  ```txt
  ①上图右侧的输出结果：
    init a <class '__main__.Model'>
  init a ModelMeta
    (<class '__main__.Model'>, <class 'object'>)
  (<class '__main__.ModelMeta'>, <class 'type'>, <class 'object'>)
    <__main__.Model object at 0x0000021FE79FE490>
  <class '__main__.ModelMeta'> <class 'type'>
    
  ②所有的类若不指定metaclass，则metaclass默认为type;除了type，ABC也是元类，声明metaclass=ABC代表该类是抽象类
  ```

(5)协议编程

- Dunders：被双下划线包围的属性名或方法名

- \_\_new\_\_()

  ①定义
  ![image-20231206161125740](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231206161125740.png)

  ②类的实例化时会执行\_\_new\_\_方法，\_\_new\_\_返回的是什么类的实例该类的实例化默认就是那个类![image-20231206161529155](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231206161529155.png)

- 其他：
  ![image-20231206163652705](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231206163652705.png)
  ![image-20231206163713427](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231206163713427.png)

  ![image-20231206163944341](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231206163944341.png)
  ![image-20231206164148550](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231206164148550.png)
  ![image-20231206164532663](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231206164532663.png)

(6)单例模式

- 顾名思义：只有单个实例
  ![image-20231206162127532](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231206162127532.png)

- 单例模式的实现==（考试只需要写一种）==

  ①直接实现
  ![image-20231206162224842](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231206162224842.png)

  ②使用继承的方式实现单例类

  ![image-20231206163059862](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231206163059862.png)

  ③使用元类的方式实现单例类
  ![image-20231206163034473](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231206163034473.png)

  ==(没明白，要搞清楚\_\_call\_\_)==

(7)描述器协议

- 定义
  ![image-20231206170200985](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231206170200985.png)

  > 当你删除一个对象时，会调用`__del__`，当你删除一个对象的属性时，有时会调用`__delete__`
  >
  > 描述器方法的第一个参数是self，第二个参数是调用该描述器的属性所在的类
  
- 用途
  ![image-20231206172455999](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231206172455999.png)

  动态查找、托管属性、定制名称

- 数据描述器与非数据描述器：非数据描述器只实现了\_\_get\_\_方法；数据描述器除了\_get\_\_方法外，实现了\_\_set\_\_或\_\_delete\_\_至少一种方法

- 描述器涉及的一些函数、方法及访问顺序
  ![image-20231206172852828](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231206172852828.png)

  ![image-20231206172837306](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231206172837306.png)

  ①`getattr()`:==（考试要考！！）==
  ![image-20231206172748443](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231206172748443.png)

  ②`__getattribute__()`:
  ![image-20231206173315284](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231206173315284.png)

  > 当存在描述器的时候，一个类实例的查找属性顺序为：先查找类或父类中是否有数据描述器属性，如果有那么，先访问数据描述器，如果没有数据描述器 --> 那么就会查找自己实例的dict属性，如果dict属性里面也没有找到 --> 然后会在类或父类的非数据描述器进行查找。

- \_\_getattribute\_\_和\_\_getattr\_\_例子：
  ==（输出顺序掌握！！！）==
    ![image-20231206173535801](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231206173535801.png)
    ![image-20231206174128984](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231206174128984.png)
    ![image-20231213161540014](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231213161540014.png)

- 赋值语句、函数vars()触发\_\_getattribute\_\_
  ![image-20231213161903861](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231213161903861.png)

- ![](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231213161911647.png)

- \_\_getattribute\_\_陷阱

  ![image-20231213161952230](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231213161952230.png)

- 描述器：点语法调用属性的访问顺序
  ![image-20231213162132458](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231213162132458.png)

- 描述器应用：ORM

  ORM核心思路是：将数据存储在外部数据库中，Python 实例仅持有数据库表中对应的的键，描述器负责对值进行查找或更新

- Property属性描述器

  ①通过属性的方式调用方法
  ②通过property检查实例参数，如属性只读、参数制约等

  样例：
  ![image-20231213170233264](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231213170233264.png)
  ![image-20231213170245066](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231213170245066.png)

### 异常处理

#### 1.错误和异常

(1)错误：语法错误、逻辑错误、异常

- 语法错误：如拼写错误（IDE能给出提示）

- 逻辑错误：代码可执行，但执行结果不符合要求

- 异常（Exception）：
  在python中，异常是一个类，可以处理和使用

  ①异常处理

  ```python
  try:
      pass # do something
  except Exception as error:
      pass # do something
  else:
      pass # do something
  finally:
      pass # do something
  ```

  ②抛出异常：`raise`用于抛出异常

(2)异常处理

- **try**：正常执行的程序，如果执行过程中出现异常，则中断当前的程序执行，跳转到对应的异常处理模块中
  **except**：（可选）如果异常与A/B相匹配，则跳转到对应的except A/B中执行；如果A、B中没有相对应的异常，则跳转到except中执行
  **else**：（可选）如果try中的程序执行过程中没有发生错误，则继续执行else中的程序
  **finally**：无论是否发生异常，只要提供了finally程序，就在执行所有步骤之后执行finally中的程序

- ==注意：==
  ①try无论执行成功失败，都会执行finally
  ②try、else、except中如果有return，当代码执行到return之后，会直接跳转到finally中，开始执行finally中的所有语句，包括return，（敲黑板，是包括return的，return执行完，程序就结束了，不会再执行try、else、except中的return）
  ③当except没有捕获try中抛出的异常时，会直接执行finally，这个时候，如果finally中没有return，则finally中的代码执行完成之后，try中的异常会被抛出，但是如果finally中有return，则不会抛出异常
  ④其余要点见下图
  ![image-20231220172417615](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231220172417615.png)

- 手动触发异常

  ![image-20231220162451919](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231220162451919.png)

- 预定义的清理操作：某些对象定义了不需要该对象时要执行的标准清理操作，无论使用该对象的操作是否成功，都会执行清理操作。常见的是with语句
  ![image-20231220173101000](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231220173101000.png)

  > 与with相关的两个协议：\_\_enter\_\_和\_\_exit\_\_

(3)异常链

- 如果 except 子句有未触发的异常被我们在执行的过程中触发了，该异常的\_\_context\_\_属性会被自动设为已处理的异常，并随 try 子句异常一并抛出

  ![image-20231220171920090](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231220171920090.png)

- `raise new from old`结构，在转换异常时非常有用

  可使用`from None`的方式新异常替换原异常以显示其目的

  ![image-20231220172005221](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231220172005221.png)

(4)异常类层次结构

- 内置异常类层次结构
  ![image-20231220173234933](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231220173234933.png)
- 继承Exception的内置异常（注意：Warning也是）

### 文件I/O

#### 1.文件I/O

(1)文件：由文本I/O、二进制I/O和原始I/O生成的对象

(2)操作文件I/O的流程：①打开文件 ②读写文件 ③关闭文件

(3)类文件对象（file-like object）

- 像open()函数返回的这种有read()方法的对象，称为file-like obejct
- 除了file外，还可以是内存的字节流、网络流、自定义流等等
- file-like object只要实现read方法就行
- StringIO就是在内存中创建的file-like object，常作为临时缓冲

(4)I/O操作

- 文本I/O
  ![image-20231220175851851](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231220175851851.png)
- 二进制I/O
  ![image-20231220175902340](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231220175902340.png)
- 原始I/O（指定没有缓存）
  ![image-20231220175955479](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231220175955479.png)
- 文件I/O打开
  ![image-20231220175941478](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231220175941478.png)

(5)文件/目录

- os模块
  ![image-20231220180006875](https://gitee.com/mianmann/drawing-bed-warehouse/raw/master/img/image-20231220180006875.png)
- python中的4大文件处理库：os、shutil、pathlib、glob

### 模块、包

#### 1.模块

(1)为了使代码更容易维护，可重复使用，可以将一组相关功能的代码写入一个单独的.py文件中。这样的.py文件称为模块

(2)搜索路径：当前目录

(3)\_\_name\_\_

#### 2.包

(1)每一个包都必须要有\_\_init\_\_.py文件

















