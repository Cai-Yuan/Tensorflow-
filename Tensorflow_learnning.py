In [1]:
a = 1.2
aa = tf.constant(1.2) # 创建标量
type(a), type(aa), tf.is_tensor(aa)
Out[1]:
(float, tensorflow.python.framework.ops.EagerTensor, True)

In [2]: x = tf.constant([1,2.,3.3])
x
Out[2]:
<tf.Tensor: id=165, shape=(3,), dtype=float32, numpy=array([1. , 2. , 3.3],
dtype=float32)>

In [3]: x.numpy()
Out[3]:
array([1. , 2. , 3.3], dtype=float32)

In [4]:
a = tf.constant([1.2])
a, a.shape
Out[4]:
(<tf.Tensor: id=8, shape=(1,), dtype=float32, numpy=array([1.2],
dtype=float32)>, TensorShape([1]))


In [5]:
a = tf.constant([1,2, 3.])
a, a.shape
Out[5]:
(<tf.Tensor: id=11, shape=(3,), dtype=float32, numpy=array([1., 2., 3.],
dtype=float32)>, TensorShape([3]))


In [6]:
a = tf.constant([[1,2],[3,4]])
a, a.shape
Out[6]:
(<tf.Tensor: id=13, shape=(2, 2), dtype=int32, numpy=
array([[1, 2], [3, 4]])>, TensorShape([2, 2]))

In [7]:
a = tf.constant([[[1,2],[3,4]],[[5,6],[7,8]]])
Out[7]:
<tf.Tensor: id=15, shape=(2, 2, 2), dtype=int32, numpy=array([[[1, 2],
[3, 4]], [[5, 6], [7, 8]]])>


In [8]:
a = tf.constant('Hello, Deep Learning.')
Out[8]:
<tf.Tensor: id=17, shape=(), dtype=string, numpy=b'Hello, Deep Learning.'>

In [9]:
tf.strings.lower(a)
Out[9]:
<tf.Tensor: id=19, shape=(), dtype=string, numpy=b'hello, deep learning.'>

In [10]: a = tf.constant(True)
Out[10]:
<tf.Tensor: id=22, shape=(), dtype=bool, numpy=True>

In [11]:
a = tf.constant([True, False])
Out[11]:
<tf.Tensor: id=25, shape=(2,), dtype=bool, numpy=array([ True, False])>



In [11]:
a = tf.constant(True) # 创建布尔张量
a == True
Out[11]:
False


In [12]:
tf.constant(123456789, dtype=tf.int16)
tf.constant(123456789, dtype=tf.int32)
Out[12]:
<tf.Tensor: id=33, shape=(), dtype=int16, numpy=-13035>
<tf.Tensor: id=35, shape=(), dtype=int32, numpy=123456789>


In [13]:
import numpy as np
np.pi
tf.constant(np.pi, dtype=tf.float32)
Out[13]:
<tf.Tensor: id=29, shape=(), dtype=float32, numpy=3.1415927>




In [14]:tf.constant(np.pi, dtype=tf.float64)
Out [14]:
<tf.Tensor: id=31, shape=(), dtype=float64, numpy=3.141592653589793>


In [15]:
print('before:',a.dtype)
if a.dtype != tf.float32:
    a = tf.cast(a,tf.float32) # 转换精度
print('after :',a.dtype)
Out[15]:
before: <dtype: 'float16'>
after : <dtype: 'float32'>


In [16]:
a = tf.constant(np.pi, dtype=tf.float16)
tf.cast(a, tf.double)
Out[16]:
<tf.Tensor: id=44, shape=(), dtype=float64, numpy=3.140625>


In [17]:
a = tf.constant(123456789, dtype=tf.int32)
tf.cast(a, tf.int16)
Out[17]:
<tf.Tensor: id=38, shape=(), dtype=int16, numpy=-13035>

In [18]:
a = tf.constant([True, False])
tf.cast(a, tf.int32)
Out[18]:
<tf.Tensor: id=48, shape=(2,), dtype=int32, numpy=array([1, 0])>


In [19]:
a = tf.constant([-1, 0, 1, 2])
tf.cast(a, tf.bool)
Out[19]:
<tf.Tensor: id=51, shape=(4,), dtype=bool, numpy=array([ True, False, True,
True])>


In [20]:
a = tf.constant([-1, 0, 1, 2])
aa = tf.Variable(a)
aa.name, aa.trainable
Out[20]:
('Variable:0', True)



In [21]:
a = tf.Variable([[1,2],[3,4]])
Out[21]:
<tf.Variable 'Variable:0' shape=(2, 2) dtype=int32, numpy=
array([[1, 2],
[3, 4]


In [22]:
tf.convert_to_tensor([1,2.])
Out[22]:
<tf.Tensor: id=86, shape=(2,), dtype=float32, numpy=array([1., 2.],
dtype=float32)> 

In [23]:
tf.convert_to_tensor(np.array([[1,2.],[3,4]]))
Out[23]:
<tf.Tensor: id=88, shape=(2, 2), dtype=float64, numpy=
array([[1., 2.],
[3., 4.]])>



In [24]: tf.zeros([]),tf.ones([])
Out[24]:
(<tf.Tensor: id=90, shape=(), dtype=float32, numpy=0.0>,
<tf.Tensor: id=91, shape=(), dtype=float32, numpy=1.0>)



In [25]: tf.zeros([1]),tf.ones([1])
Out[25]:
(<tf.Tensor: id=96, shape=(1,), dtype=float32, numpy=array([0.],
dtype=float32)>,
<tf.Tensor: id=99, shape=(1,), dtype=float32, numpy=array([1.],
dtype=float32)>)



In [26]: tf.zeros([2,2])
Out[26]:
<tf.Tensor: id=104, shape=(2, 2), dtype=float32, numpy=
array([[0., 0.],[0., 0.]], dtype=float32)>



In [27]: tf.ones([3,2])
Out[27]:
<tf.Tensor: id=108, shape=(3, 2), dtype=float32, numpy=
array([[1., 1.],
[1., 1.],
[1., 1.]], dtype=float32)>


In [28]: a = tf.ones([2,3])
tf.zeros_like(a)
Out[28]:
<tf.Tensor: id=113, shape=(2, 3), dtype=float32, numpy=
array([[0., 0., 0.],
[0., 0., 0.]], dtype=float32)>


In [29]: a = tf.zeros([3,2])
tf.ones_like(a)
Out[29]:
<tf.Tensor: id=120, shape=(3, 2), dtype=float32, numpy=
array([[1., 1.],
[1., 1.],
[1., 1.]], dtype=float32)>



In [30]:tf.fill([], -1)
Out[30]:
<tf.Tensor: id=124, shape=(), dtype=int32, numpy=-1>



In [31]:tf.fill([1], -1)
Out[31]:
<tf.Tensor: id=128, shape=(1,), dtype=int32, numpy=array([-1])>



In [32]:tf.fill([2,2], 99)
Out[32]:
<tf.Tensor: id=136, shape=(2, 2), dtype=int32, numpy=
array([[99, 99],
[99, 99]])>



In [33]: tf.random.normal([2,2])
Out[33]:
<tf.Tensor: id=143, shape=(2, 2), dtype=float32, numpy=
array([[-0.4307344 , 0.44147003],
[-0.6563149 , -0.30100572]], dtype=float32)>



In [34]: tf.random.normal([2,2], mean=1,stddev=2)
Out[34]:
<tf.Tensor: id=150, shape=(2, 2), dtype=float32, numpy=
array([[-2.2687864, -0.7248812],
[ 1.2752185, 2.8625617]], dtype=float32)>


In [35]: tf.random.uniform([2,2])
Out[35]:
<tf.Tensor: id=158, shape=(2, 2), dtype=float32, numpy=
array([[0.65483284, 0.63064325],
[0.008816 , 0.81437767]], dtype=float32)>


In [36]: tf.random.uniform([2,2],maxval=10)
Out[36]:
<tf.Tensor: id=166, shape=(2, 2), dtype=float32, numpy=
array([[4.541913 , 0.26521802],
[2.578913 , 5.126876 ]], dtype=float32)>


In [37]:tf.random.uniform([2,2],maxval=100,dtype=tf.int32)
Out[37]:
<tf.Tensor: id=171, shape=(2, 2), dtype=int32, numpy=
array([[61, 21],
[95, 75]])>


In [38]: tf.range(10)
Out[38]:
<tf.Tensor: id=180, shape=(10,), dtype=int32, numpy=array([0, 1, 2, 3, 4, 5,
6, 7, 8, 9]


In [39]: tf.range(10,delta=2)
Out[39]:
<tf.Tensor: id=185, shape=(5,), dtype=int32, numpy=array([0, 2, 4, 6, 8])>


In [40]: tf.range(1,10,delta=2)
Out[40]:
<tf.Tensor: id=190, shape=(5,), dtype=int32, numpy=array([1, 3, 5, 7, 9])>


In [41]:
out = tf.random.uniform([4,10]) #随机模拟网络输出
y = tf.constant([2,3,2,0]) # 随机构造样本真实标签
y = tf.one_hot(y, depth=10) # one-hot 编码
loss = tf.keras.losses.mse(y, out) # 计算每个样本的 MSE
loss = tf.reduce_mean(loss) # 平均 MSE
print(loss)
Out[41]:
tf.Tensor(0.19950335, shape=(), dtype=float32)


In [42]:
# z=wx,模拟获得激活函数的输入 z
z = tf.random.normal([4,2])
b = tf.zeros([2]) # 模拟偏置向量
z = z + b # 累加偏置
Out[42]:
<tf.Tensor: id=245, shape=(4, 2), dtype=float32, numpy=
array([[ 0.6941646 , 0.4764454 ],
[-0.34862405, -0.26460952],
[ 1.5081744 , -0.6493869 ],
[-0.26224667, -0.78742725]], dtype=float32)>

In [43]:
fc = layers.Dense(3) # 创建一层 Wx+b，输出节点为 3
# 通过 build 函数创建 W,b 张量，输入节点为 4
fc.build(input_shape=(2,4))
fc.bias # 查看偏置
Out[43]:
<tf.Variable 'bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.],
dtype=float32)>


In [44]:
w = tf.ones([4,3]) # 定义 W 张量
b = tf.zeros([3]) # 定义 b 张量
o = x@w+b # X@W+b 运算
Out[44]:
<tf.Tensor: id=291, shape=(2, 3), dtype=float32, numpy=
array([[ 2.3506963, 2.3506963, 2.3506963],
[-1.1724043, -1.1724043, -1.1724043]], dtype=float32)>


In [45]:
fc = layers.Dense(3) # 定义全连接层的输出节点为 3
fc.build(input_shape=(2,4)) # 定义全连接层的输入节点为 4
fc.kernel
Out[45]:
<tf.Variable 'kernel:0' shape=(4, 3) dtype=float32, numpy=
array([[ 0.06468129, -0.5146048 , -0.12036425],
[ 0.71618867, -0.01442951, -0.5891943 ],
[-0.03011459, 0.578704 , 0.7245046 ],
[ 0.73894167, -0.21171576, 0.4820758 ]], dtype=float32)>




In [46]: # 自动加载 IMDB 电影评价数据集
(x_train,y_train),(x_test,y_test)=keras.datasets.imdb.load_data(num_words=10
000)
# 将句子填充、截断为等长 80 个单词的句子
x_train = keras.preprocessing.sequence.pad_sequences(x_train,maxlen=80)
x_train.shape
Out [46]: (25000, 80)



In [47]: # 创建词向量 Embedding 层类
embedding=layers.Embedding(10000, 100)
# 将数字编码的单词转换为词向量
out = embedding(x_train)
out.shape
Out[47]: TensorShape([25000, 80, 100])



In [48]:
# 创建 32x32 的彩色图片输入，个数为 4
x = tf.random.normal([4,32,32,3])
# 创建卷积神经网络
layer = layers.Conv2D(16,kernel_size=3)
out = layer(x) # 前向计算
out.shape # 输出大小
Out[48]: TensorShape([4, 30, 30, 16])
其中卷积核张量也是 4 维张量，可以通过 kernel 成员变量访问：
In [49]: layer.kernel.shape
Out[49]: TensorShape([3, 3, 3, 16])




In [51]: x[0]
Out[51]:<tf.Tensor: id=379, shape=(32, 32, 3), dtype=float32, numpy=
array([[[ 1.3005302 , 1.5301839 , -0.32005513],
[-1.3020388 , 1.7837263 , -1.0747638 ], ...
[-1.1092019 , -1.045254 , -0.4980363 ],
[-0.9099222 , 0.3947732 , -0.10433522]]], dtype=float32)>


In [52]: x[0][1]
Out[52]:
<tf.Tensor: id=388, shape=(32, 3), dtype=float32, numpy=
array([[ 4.2904025e-01, 1.0574218e+00, 3.1540772e-01],
[ 1.5800388e+00, -8.1637271e-02, 6.3147342e-01], ...,
[ 2.8893018e-01, 5.8003378e-01, -1.1444757e+00],
[ 9.6100050e-01, -1.0985689e+00, 1.0827581e+00]], dtype=float32)>


In [53]: x[0][1][2]
Out[53]:
<tf.Tensor: id=401, shape=(3,), dtype=float32, numpy=array([-0.55954427,
0.14497331, 0.46424514], dtype=float32)>

In [54]: x[2][1][0][1]
Out[54]:
<tf.Tensor: id=418, shape=(), dtype=float32, numpy=-0.84922135>


In [55]: x[1,9,2]
Out[55]:
<tf.Tensor: id=436, shape=(3,), dtype=float32, numpy=array([ 1.7487534 , -
0.41491988, -0.2944692 ], dtype=float32)>


In [56]: x[1:3]
Out[56]:
<tf.Tensor: id=441, shape=(2, 32, 32, 3), dtype=float32, numpy=
array([[[[ 0.6920027 , 0.18658352, 0.0568333 ],
[ 0.31422952, 0.75933754, 0.26853144],
[ 2.7898 , -0.4284912 , -0.26247284],..



In [57]: x[0,::]
Out[57]:
<tf.Tensor: id=446, shape=(32, 32, 3), dtype=float32, numpy=
array([[[ 1.3005302 , 1.5301839 , -0.32005513],
[-1.3020388 , 1.7837263 , -1.0747638 ],
[-1.1230233 , -0.35004002, 0.01514002],



In [58]: x[:,0:28:2,0:28:2,:]
Out[58]:
<tf.Tensor: id=451, shape=(4, 14, 14, 3), dtype=float32, numpy=
array([[[[ 1.3005302 , 1.5301839 , -0.32005513],
[-1.1230233 , -0.35004002, 0.01514002],
[ 1.3474811 , 0.639334 , -1.0826371 ],



In [59]: x = tf.range(9)
x[8:0:-1]
Out[59]:
<tf.Tensor: id=466, shape=(8,), dtype=int32, numpy=array([8, 7, 6, 5, 4, 3,
2, 1])>
逆序取全部元素：
In [60]: x[::-1]
Out[60]:
<tf.Tensor: id=471, shape=(9,), dtype=int32, numpy=array([8, 7, 6, 5, 4, 3,
2, 1, 0])>
逆序间隔采样：
In [61]: x[::-2]
Out[61]:
<tf.Tensor: id=476, shape=(5,), dtype=int32, numpy=array([8, 6, 4, 2, 0])>




In [62]: x = tf.random.normal([4,32,32,3])
x[0,::-2,::-2]

Out[62]:
<tf.Tensor: id=487, shape=(16, 16, 3), dtype=float32, numpy=
array([[[ 0.63320625, 0.0655185 , 0.19056146],
[-1.0078577 , -0.61400175, 0.61183935],
[ 0.9230892 , -0.6860094 , -0.01580668],


In [63]: x[:,:,:,1]
Out[63]:
<tf.Tensor: id=492, shape=(4, 32, 32), dtype=float32, numpy=
array([[[ 0.575703 , 0.11028383, -0.9950867 , ..., 0.38083118,
-0.11705163, -0.13746642],
...


In [64]: x[0:2,...,1:]
Out[64]:
<tf.Tensor: id=497, shape=(2, 32, 32, 2), dtype=float32, numpy=
array([[[[ 0.575703 , 0.8872789 ],
[ 0.11028383, -0.27128693],
[-0.9950867 , -1.7737272 ],



In [65]: x[2:,...]
Out[65]:
<tf.Tensor: id=502, shape=(2, 32, 32, 3), dtype=float32, numpy=
array([[[[-8.10753584e-01, 1.10984087e+00, 2.71821529e-01],
[-6.10031188e-01, -6.47952318e-01, -4.07003373e-01],
[ 4.62206364e-01, -1.03655539e-01, -1.18086267e+00],



In [66]: x[...,:2]
Out[66]:
<tf.Tensor: id=507, shape=(4, 32, 32, 2), dtype=float32, numpy=
array([[[[-1.26881 , 0.575703 ],
[ 0.98697686, 0.11028383],
[-0.66420585, -0.9950867 ],



In [67]: x=tf.range(96)
x=tf.reshape(x,[2,4,4,3])
Out[67]:
<tf.Tensor: id=11, shape=(2, 4, 4, 3), dtype=int32, numpy=
array([[[[ 0, 1, 2],
[ 3, 4, 5],
[ 6, 7, 8],
[ 9, 10, 11]],…


In [68]: x.ndim,x.shape
Out[68]:(4, TensorShape([2, 4, 4, 3]))


In [69]: tf.reshape(x,[2,-1])
Out[69]:<tf.Tensor: id=520, shape=(2, 48), dtype=int32, numpy=
array([[ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,…
80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]])>



In [70]: tf.reshape(x,[2,4,12])
Out[70]:<tf.Tensor: id=523, shape=(2, 4, 12), dtype=int32, numpy=
array([[[ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],…
[36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]],
[[48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59], …
[84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]]])>


In [71]: tf.reshape(x,[2,-1,3])
Out[71]:<tf.Tensor: id=526, shape=(2, 16, 3), dtype=int32, numpy=
array([[[ 0, 1, 2], …
[45, 46, 47]],
[[48, 49, 50]


In [72]:
x = tf.random.uniform([28,28],maxval=10,dtype=tf.int32)
Out[72]:
<tf.Tensor: id=552, shape=(28, 28), dtype=int32, numpy=
array([[4, 5, 7, 6, 3, 0, 3, 1, 1, 9, 7, 7, 3, 1, 2, 4, 1, 1, 9, 8, 6, 6,
4, 9, 9, 4, 6, 0],


In [73]: x = tf.expand_dims(x,axis=2)
Out[73]:
<tf.Tensor: id=555, shape=(28, 28, 1), dtype=int32, numpy=
array([[[4],
[5],
[7],
[6],
[3],…



In [74]: x = tf.expand_dims(x,axis=0)
Out[74]:
<tf.Tensor: id=558, shape=(1, 28, 28), dtype=int32, numpy=
array([[[4, 5, 7, 6, 3, 0, 3, 1, 1, 9, 7, 7, 3, 1, 2, 4, 1, 1, 9, 8, 6,
6, 4, 9, 9, 4, 6, 0],
[5, 8, 6, 3, 6, 4, 3, 0, 5, 9, 0, 5, 4, 6, 4, 9, 4, 4, 3, 0, 6,
9, 3, 7, 4, 2, 8, 9],…




In [75]: x = tf.squeeze(x, axis=0)
Out[75]:
<tf.Tensor: id=586, shape=(28, 28, 1), dtype=int32, numpy=
array([[[8],
[2],
[2],
[0]


In [76]: x = tf.squeeze(x, axis=2)
Out[76]:
<tf.Tensor: id=588, shape=(28, 28), dtype=int32, numpy=
array([[8, 2, 2, 0, 7, 0, 1, 4, 9, 1, 7, 4, 8, 2, 7, 4, 8, 2, 9, 8, 8, 0,
9, 9, 7, 5, 9, 7],
[3, 4, 9, 9, 0, 6, 5, 7, 1, 9, 9, 1, 2, 7, 2, 7, 5, 3, 3, 7, 2, 4,
5, 2, 7, 3, 8, 0],…

如果不指定维度参数 axis，即 tf.squeeze(x)， 那么他会默认删除所有长度为 1 的维度：
In [77]:
x = tf.random.uniform([1,28,28,1],maxval=10,dtype=tf.int32)
tf.squeeze(x)
Out[77]:
<tf.Tensor: id=594, shape=(28, 28), dtype=int32, numpy=
array([[9, 1, 4, 6, 4, 9, 0, 0, 1, 4, 0, 8, 5, 2, 5, 0, 0, 8, 9, 4, 5, 0,
1, 1, 4, 3, 9, 9



In [78]: x = tf.random.normal([2,32,32,3])
tf.transpose(x,perm=[0,3,1,2])
Out[78]:
<tf.Tensor: id=603, shape=(2, 3, 32, 32), dtype=float32, numpy=
array([[[[-1.93072677e+00, -4.80163872e-01, -8.85614634e-01, ...,
1.49124235e-01, 1.16427064e+00, -1.47740364e+00],
[-1.94761145e+00, 7.26879001e-01, -4.41877693e-01, ...


如果希望将[𝐬 ℎ, , 𝑝交换为[𝐬 , ℎ, 𝑝，即将行列维度互换，则新维度索引为[0,2,1,3]:
In [79]:
x = tf.random.normal([2,32,32,3])
tf.transpose(x,perm=[0,2,1,3])
Out[79]:
<tf.Tensor: id=612, shape=(2, 32, 32, 3), dtype=float32, numpy=
array([[[[ 2.1266546 , -0.64206547, 0.01311932],
[ 0.918484 , 0.9528751 , 1.1346699 ],
...,


In [80]:
b = tf.constant([1,2])
b = tf.expand_dims(b, axis=0)
b
Out[80]:
<tf.Tensor: id=645, shape=(1, 2), dtype=int32, numpy=array([[1, 2]])>
在 batch 维度上复制数据 1 份：
In [81]: b = tf.tile(b, multiples=[2,1])
Out[81]:
<tf.Tensor: id=648, shape=(2, 2), dtype=int32, numpy=
array([[1, 2],
[1, 2]



In [82]: x = tf.range(4)
x=tf.reshape(x,[2,2])
Out[82]:
<tf.Tensor: id=655, shape=(2, 2), dtype=int32, numpy=
array([[0, 1],
[2, 3]])>

首先在列维度复制 1 份数据：
In [83]: x = tf.tile(x,multiples=[1,2])
Out[83]:
<tf.Tensor: id=658, shape=(2, 4), dtype=int32, numpy=
array([[0, 1, 0, 1],


In [84]: x = tf.tile(x,multiples=[2,1])
Out[84]:
<tf.Tensor: id=672, shape=(4, 4), dtype=int32, numpy=
array([[0, 1, 0, 1],
[2, 3, 2, 3],
[0, 1, 0, 1],
[2, 3, 2, 3]])>

In [87]:
A = tf.random.normal([32,1])
tf.broadcast_to(A, [2,32,32,3])
Out[87]:
<tf.Tensor: id=13, shape=(2, 32, 32, 3), dtype=float32, numpy=
array([[[[-1.7571245 , -1.7571245 , -1.7571245 ],
[ 1.580159 , 1.580159 , 1.580159 ],
[-1.5324328 , -1.5324328 , -1.5324328 ],...

In [88]:
A = tf.random.normal([32,2])
tf.broadcast_to(A, [2,32,32,4])
Out[88]:
InvalidArgumentError: Incompatible shapes: [32,2] vs. [2,32,32,4]
[Op:BroadcastTo]

In [89]:
a = tf.range(5)
b = tf.constant(2)
a//b
Out[89]:
<tf.Tensor: id=115, shape=(5,), dtype=int32, numpy=array([0, 0, 1, 1, 2])>
余除运算：
In [90]: a%b
Out[90]:
<tf.Tensor: id=117, shape=(5,), dtype=int32, numpy=array([0, 1, 0, 1, 0])>



In [91]:
x = tf.range(4)
tf.pow(x,3)
Out[91]:
<tf.Tensor: id=124, shape=(4,), dtype=int32, numpy=array([ 0, 1, 8, 27])>
In [92]: x**2
Out[92]:
<tf.Tensor: id=127, shape=(4,), dtype=int32, numpy=array([0, 1, 4, 9])>
设置指数为1形式即可实现根号运算：

In [93]: x=tf.constant([1.,4.,9.])
x**(0.5)
Out[93]:
<tf.Tensor: id=139, shape=(3,), dtype=float32, numpy=array([1., 2., 3.],
dtype=float32)>


In [94]:x = tf.range(5)
x = tf.cast(x, dtype=tf.float32)
x = tf.square(x)
Out[94]:
<tf.Tensor: id=159, shape=(5,), dtype=float32, numpy=array([ 0., 1., 4.,
9., 16.], dtype=float32)>
平方根运算实现如下：
In [95]:tf.sqrt(x)
Out[95]:
<tf.Tensor: id=161, shape=(5,), dtype=float32, numpy=array([0., 1., 2., 3.,
4.], dtype=float32



In [96]: x = tf.constant([1.,2.,3.])
2**x
Out[96]:
<tf.Tensor: id=179, shape=(3,), dtype=float32, numpy=array([2., 4., 8.],
dtype=float32)>

特别地，对于自然指数𝐵,可以通过 tf.exp(x)实现：
In [97]: tf.exp(1.)
Out[97]:
<tf.Tensor: id=182, shape=(), dtype=float32, numpy=2.7182817>


In [98]: x=tf.exp(3.)
tf.math.log(x)
Out[98]:
<tf.Tensor: id=186, shape=(), dtype=float32, numpy=3.0>


In [99]: x = tf.constant([1.,2.])
x = 10**x
tf.math.log(x)/tf.math.log(10.)
Out[99]:
<tf.Tensor: id=222, shape=(2,), dtype=float32, numpy=array([0. ,
2.3025851], dtype=float32)>


In [100]:
a = tf.random.normal([4,3,23,32])
b = tf.random.normal([4,3,32,2])
a@b
Out[100]:
<tf.Tensor: id=236, shape=(4, 3, 28, 2), dtype=float32, numpy=
array([[[[-1.66706240e+00, -8.32602978e+00],
[ 9.83304405e+00, 8.15909767e+00],
[ 6.31014729e+00, 9.26124632e-01],


In [101]:
a = tf.random.normal([4,28,32])
b = tf.random.normal([32,16])
tf.matmul(a,b)
Out[101]:
<tf.Tensor: id=264, shape=(4, 28, 16), dtype=float32, numpy=
array([[[-1.11323869e+00, -9.48194981e+00, 6.48123884e+00, ...,
6.53280640e+00, -3.10894990e+00, 1.53050375e+00],
[ 4.35898495e+00, -1.03704405e+01, 8.90656471e+00



In [1]:
a = tf.random.normal([4,35,8]) # 模拟成绩册 A
b = tf.random.normal([6,35,8]) # 模拟成绩册 B
tf.concat([a,b],axis=0) # 合并成绩册
Out[1]:
<tf.Tensor: id=13, shape=(10, 35, 8), dtype=float32, numpy=
array([[[ 1.95299834e-01, 6.87859178e-01, -5.80048323e-01, ...,
1.29430830e+00, 2.56610274e-01, -1.27798581e+00],
[ 4.29753691e-01, 9.11329567e-01, -4.47975427e-01, ...,


In [2]:
a = tf.random.normal([10,35,4])
b = tf.random.normal([10,35,4])
tf.concat([a,b],axis=2) # 在科目维度拼接
Out[2]:
<tf.Tensor: id=28, shape=(10, 35, 8), dtype=float32, numpy=
array([[[-5.13509691e-01, -1.79707789e+00, 6.50747120e-01, ...,
2.58447856e-01, 8.47878829e-02, 4.13468748e-01],
[-1.17108583e+00, 1.93961406e+00, 1.27830813e-02, ...,



In [3]:
a = tf.random.normal([4,32,8])
b = tf.random.normal([6,35,8])
tf.concat([a,b],axis=0) # 非法拼接
Out[3]:
InvalidArgumentError: ConcatOp : Dimensions of inputs should match: shape[0]
= [4,32,8] vs. shape[1] = [6,35,8] [Op:ConcatV2] name:


In [4]:
a = tf.random.normal([35,8])
b = tf.random.normal([35,8])
tf.stack([a,b],axis=0) # 堆叠合并为 2 个班级
Out[4]:
<tf.Tensor: id=55, shape=(2, 35, 8), dtype=float32, numpy
array([[[ 3.68728966e-01, -8.54765773e-01, -4.77824420e-01,
-3.83714020e-01, -1.73216307e+00, 2.03872994e-02,
2.63810277e+00, -1.12998331e+00],…


同样可以选择在其他位置插入新维度，如在最末尾插入：
In [5]:
a = tf.random.normal([35,8])
b = tf.random.normal([35,8])
tf.stack([a,b],axis=-1) # 在末尾插入班级维度
Out[5]:
<tf.Tensor: id=69, shape=(35, 8, 2), dtype=float32, numpy=
array([[[ 0.3456724 , -1.7037214 ],
[ 0.41140947, -1.1554345 ],
[ 1.8998919 , 0.56994915],…

In [6]:
a = tf.random.normal([35,8])
b = tf.random.normal([35,8])
tf.concat([a,b],axis=0) # 拼接方式合并，没有 2 个班级的概念
Out[6]:
<tf.Tensor: id=108, shape=(70, 8), dtype=float32, numpy=
array([[-0.5516891 , -1.5031327 , -0.35369992, 0.31304857, 0.13965549,
0.6696881 , -0.50115544, 0.15550546],
[ 0.8622069 , 1.0188094 , 0.18977325, 0.6353301 , 0.05809061,…


In [7]:
a = tf.random.normal([35,4])
b = tf.random.normal([35,8])
tf.stack([a,b],axis=-1) # 非法堆叠操作
Out[7]
InvalidArgumentError: Shapes of all inputs must match: values[0].shape =
[35,4] != values[1].shape = [35,8] [Op:Pack] name: stack


In [8]:
x = tf.random.normal([10,35,8])
# 等长切割
result = tf.split(x,axis=0,num_or_size_splits=10)
len(result)
Out[8]: 10


In [9]: result[0]
Out[9]: <tf.Tensor: id=136, shape=(1, 35, 8), dtype=float32, numpy=
array([[[-1.7786729 , 0.2970506 , 0.02983334, 1.3970423 ,
1.315918 , -0.79110134, -0.8501629 , -1.5549672 ],
[ 0.5398711 , 0.21478991, -0.08685189, 0.7730989 ,…



In [10]: x = tf.random.normal([10,35,8])
# 自定义长度的切割
result = tf.split(x,axis=0,num_or_size_splits=[4,2,2,2])
len(result)
Out[10]: 4


In [11]: x = tf.random.normal([10,35,8])
result = tf.unstack(x,axis=0) # Unstack 为长度为 1
len(result)
Out[11]: 10
查看切割后的张量的形状：
In [12]: result[0]
Out[12]: <tf.Tensor: id=166, shape=(35, 8), dtype=float32, numpy=
array([[-0.2034383 , 1.1851563 , 0.25327438, -0.10160723, 2.094969 ,
-0.8571669 , -0.48985648, 0.5579800


In [13]: x = tf.ones([2,2])
tf.norm(x,ord=1) # 计算 L1 范数
Out[13]: <tf.Tensor: id=183, shape=(), dtype=float32, numpy=4.0>
In [14]: tf.norm(x,ord=2) # 计算 L2 范数
Out[14]: <tf.Tensor: id=189, shape=(), dtype=float32, numpy=2.0>
In [15]: import numpy as np
tf.norm(x,ord=np.inf) # 计算∞范数
Out[15]: <tf.Tensor: id=194, shape=(), dtype=float32


In [16]: x = tf.random.normal([4,10])
tf.reduce_max(x,axis=1) # 统计概率维度上的最大值
Out[16]:<tf.Tensor: id=203, shape=(4,), dtype=float32,
numpy=array([1.2410722 , 0.88495886, 1.4170984 , 0.9550192 ],
dtype=float32)>
同样求出每个样本概率的最小值：
In [17]: tf.reduce_min(x,axis=1) # 统计概率维度上的最小值
Out[17]:<tf.Tensor: id=206, shape=(4,), dtype=float32, numpy=array([-
0.27862206, -2.4480672 , -1.9983795 , -1.5287997 ], dtype=float32)>
求出每个样本的概率的均值：
In [18]: tf.reduce_mean(x,axis=1) # 统计概率维度上的均值
Out[18]:<tf.Tensor: id=209, shape=(4,), dtype=float32,
numpy=array([ 0.39526337, -0.17684573, -0.148988 , -0.43544054],
dtype=float32)>
当不指定 axis 参数时， tf.reduce_*函数会求解出全局元素的最大、最小、 均值、和：
In [19]:x = tf.random.normal([4,10])
# 统计全局的最大、最小、均值、和
tf.reduce_max(x),tf.reduce_min(x),tf.reduce_mean(x)
Out [19]: (<tf.Tensor: id=218, shape=(), dtype=float32, numpy=1.8653786>,
<tf.Tensor: id=220, shape=(), dtype=float32, numpy=-1.9751656>,
<tf.Tensor: id=222, shape=(), dtype=float32, numpy=0.014772797>)


In [20]:
out = tf.random.normal([4,10]) # 网络预测输出
y = tf.constant([1,2,2,0]) # 真实标签
y = tf.one_hot(y,depth=10) # one-hot 编码
loss = keras.losses.mse(y,out) # 计算每个样本的误差
loss = tf.reduce_mean(loss) # 平均误差
loss
Out[20]:
<tf.Tensor: id=241, shape=(), dtype=float32, numpy=1.192118


In [21]:out = tf.random.normal([4,10])
tf.reduce_sum(out,axis=-1) # 求和
Out[21]:<tf.Tensor: id=303, shape=(4,), dtype=float32, numpy=array([-
0.588144 , 2.2382064, 2.1582587, 4.962141 ], dtype=float32)>


In [22]:out = tf.random.normal([2,10])
out = tf.nn.softmax(out, axis=1) # 通过 softmax 转换为概率值
out
Out[22]:<tf.Tensor: id=257, shape=(2, 10), dtype=float32, numpy=
array([[0.18773547, 0.1510464 , 0.09431915, 0.13652141, 0.06579739,
0.02033597, 0.06067333, 0.0666793 , 0.14594753, 0.07094406],
[0.5092072 , 0.03887136, 0.0390687 , 0.01911005, 0.03850609,
0.03442522, 0.08060656, 0.10171875, 0.08244187, 0.05604421]],
dtype=float32)>


In [23]:pred = tf.argmax(out, axis=1) # 选取概率最大的位置
pred
Out[23]:<tf.Tensor: id=262, shape=(2,), dtype=int64, numpy=array([0, 0],
dtype=int64)>


In [24]:out = tf.random.normal([100,10])
out = tf.nn.softmax(out, axis=1) # 输出转换为概率
pred = tf.argmax(out, axis=1) # 选取预测值
Out[24]:<tf.Tensor: id=272, shape=(100,), dtype=int64, numpy=
array([0, 6, 4, 3, 6, 8, 6, 3, 7, 9, 5, 7, 3, 7, 1, 5, 6, 1, 2, 9, 0, 6,
5, 4, 9, 5, 6, 4, 6, 0, 8, 4, 7, 3, 4, 7, 4, 1, 2, 4, 9, 4,…



In [25]: # 真实标签
y = tf.random.uniform([100],dtype=tf.int64,maxval=10)
Out[25]:<tf.Tensor: id=281, shape=(100,), dtype=int64, numpy=
array([0, 9, 8, 4, 9, 7, 2, 7, 6, 7, 3, 4, 2, 6, 5, 0, 9, 4, 5, 8, 4, 2,
5, 5, 5, 3, 8, 5, 2, 0, 3, 6, 0, 7, 1, 1, 7, 0, 6, 1, 2, 1, 3, …
即可获得每个样本是否预测正确。 通过 tf.equal(a, b)(或 tf.math.equal(a, b))函数可以比较这 2
个张量是否相等：
In [26]:out = tf.equal(pred,y) # 预测值与真实值比较
Out[26]:<tf.Tensor: id=288, shape=(100,), dtype=bool, numpy=
array([False, False, False, False, True, False, False, False, False,
False, False, False, False, False, True, False, False, True,…


In [27]:out = tf.cast(out, dtype=tf.float32) # 布尔型转 int 型
correct = tf.reduce_sum(out) # 统计 True 的个数
Out[27]:<tf.Tensor: id=293, shape=(), dtype=float32, numpy=12.0>


In [28]:a = tf.constant([1,2,3,4,5,6])
b = tf.constant([7,8,1,6])
b = tf.pad(b, [[0,2]]) # 填充
b
Out[28]:<tf.Tensor: id=3, shape=(6,), dtype=int32, numpy=array([7, 8, 1, 6,
0, 0])>
填充后句子张量形状一致，再将这 2 句子 Stack 在一起：
In [29]:tf.stack([a,b],axis=0) # 合并
Out[29]:<tf.Tensor: id=5, shape=(2, 6), dtype=int32, numpy=
array([[1, 2, 3, 4, 5, 6],
[7, 8, 1, 6, 0, 0]])>

In [30]:total_words = 10000 # 设定词汇量大小
max_review_len = 80 # 最大句子长度
embedding_len = 100 # 词向量长度
# 加载 IMDB 数据集
(x_train, y_train), (x_test, y_test) =
keras.datasets.imdb.load_data(num_words=total_words)
# 将句子填充或截断到相同长度，设置为末尾填充和末尾截断方式
x_train = keras.preprocessing.sequence.pad_sequences(x_train,
maxlen=max_review_len,truncating='post',padding='post')
x_test = keras.preprocessing.sequence.pad_sequences(x_test,
maxlen=max_review_len,truncating='post',padding='post')
print(x_train.shape, x_test.shape)
Out[30]: (25000, 80) (25000, 80)

