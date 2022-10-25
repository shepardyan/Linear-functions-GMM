# Linear-functions-GMM

利用GMM的线性不变性和线性运算规则，计算GMM的凸组合的PDF/CDF。

· 支持加减法（GMM+GMM / GMM + 常数）、乘法（GMM * 常数）、除法（GMM / 常数）等运算。
· 支持对GMM进行抽样和PDF/CDF计算。
· 设计了基于EM算法的GMM参数拟合方法。
· 可以利用NumPy数组(dtype=GMM)进行矩阵线性变换。
