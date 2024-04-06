


# # 华为校招机试 - 网络保卫战（20240410）
#
# import sys
#
# # 输入获取
# n = int(input())
# matrix = [list(map(int, input().split())) for _ in range(n)]
# exposed = list(map(int, input().split()))
#
# next = {}
#
#
# def dfs(i, permission, visited):
#     """
#     :param i: 攻击者进入的节点编号
#     :param permission: 攻击者此时的权限
#     :param visited: 记录已攻击过的节点
#     :return: 从节点i进入，攻击者所能攻击的节点数量
#     """
#     # 可以访问i点，此时被攻击的节点数为1
#     count = 1
#
#     # 找到i节点的下游节点j
#     for j in next[i]:
#         # 如果j已经被访问过，则跳过
#         if j in visited:
#             continue
#
#         # 如果j未被访问过，且攻击者当前权限大于等于（i访问j所需的权限值），则可以进入j节点
#         if permission >= matrix[i][j]:
#             # 递归处理
#             visited.add(j)
#             count += dfs(j, matrix[i][j], visited)
#
#     # 返回攻击者从i节点进入后，所能攻击的节点数量count
#     return count
#
#
# # 算法入口
# def solution():
#     # 找到i节点可以访问的下游节点
#     for i in range(n):
#         next[i] = []
#         for j in range(n):
#             if i != j and matrix[i][j] > 0:
#                 next[i].append(j)
#
#     # 将公网暴露的节点编号升序
#     exposed.sort()
#
#     # R[i] 表示攻击者从节点exposed[i]进入，所能攻击的节点数量
#     R = [0] * len(exposed)
#     for i in range(len(exposed)):
#         nodeIdx = exposed[i]
#
#         if matrix[nodeIdx][nodeIdx] == 0:
#             continue
#
#         visited = set()
#         visited.add(nodeIdx)
#
#         R[i] = dfs(nodeIdx, 10, visited)
#
#     minR = sys.maxsize
#     idx = -1
#
#     # 下线exposed[i]节点
#     for i in range(len(exposed)):
#         # maxR记录下线exposed[i]节点后，攻击者从其他暴露节点进入所能产生的最大攻击数量
#         maxR = 0
#
#         # 攻击者从exposed[j]进入，则能攻击节点数量为R[j]
#         for j in range(len(exposed)):
#             # 如果 i == j，则exposed[j]为下线节点，不可进入
#             if j != i:
#                 maxR = max(maxR, R[j])
#
#         # 下线exposed[i]节点，则攻击者最多可以攻击maxR个节点，如果maxR < min，则下线exposed[i]节点是更优策略
#         if maxR < minR:
#             minR = maxR
#             idx = exposed[i]
#
#     return idx
#
#
# # 算法调用
# print(solution())



## 华为校招机试 - 相似图片分类（20240410）
# import re
#
#
# class UnionFindSet:  # 并查集实现
#     def __init__(self, n):
#         self.fa = [i for i in range(n)]
#         # sum[i]表示以i为根的相似类中所有图片的相似度之和
#         self.sum = [0 for i in range(n)]
#
#     def find(self, x):
#         if x != self.fa[x]:
#             self.fa[x] = self.find(self.fa[x])
#             return self.fa[x]
#         return x
#
#     def union(self, x, y, val):
#         x_fa = self.find(x)
#         y_fa = self.find(y)
#
#         # 本次新增的相似度，归到x_fa根或者y_fa根都可以
#         self.sum[x_fa] += val
#
#         if x_fa != y_fa:
#             # 让Y_fa指向x_fa, 即x_fa成为新根
#             self.fa[y_fa] = x_fa
#
#             # 此时y_fa上的相似度之和累计到新根x_fa上
#             self.sum[x_fa] += self.sum[y_fa]
#             # 累计完后，y_fa不在记录相似度
#             self.sum[y_fa] = 0
#
#
# # 算法入口
# def solution():
#     n = int(input())
#
#     matrix = []
#     for _ in range(n):
#         # 每行有 N 列数据，空格分隔（为了显示整弃，空格可能为多个）
#         matrix.append(list(map(int, re.split(r"\s+", input()))))
#
#     ufs = UnionFindSet(n)
#
#     for i in range(n):
#         for j in range(i + 1, n):
#             similar = matrix[i][j]
#
#             if similar > 0:
#                 # 合并两个子连通图为一个，将所有相似度之和（包括本次similar）全部转移到新连通图的根节点上
#                 ufs.union(i, j, similar)
#
#     ans = []
#
#     for i in range(n):
#         # 找根节点
#         if ufs.find(i) == i:
#             # 相似类是一个连通子图，连通子图中所有图片的相似度之和记录在根上
#             ans.append(ufs.sum[i])
#
#     # 按照 “从大到小” 的顺序返回每个相似类中所有图片的相似度之和
#     ans.sort(reverse=True)
#
#     return " ".join(map(str, ans))
#
#
# # 算法调用
# print(solution())

# # 华为校招机试 - 云服务计费（20240410） 逻辑判断
# # 算法入口
# def solution():
#     # 计费日志的条数
#     n = int(input())
#
#     # key是客户id，val是客户的计费日志容器
#     logs = {}
#     for _ in range(n):
#         # 时间戳,客户标识,计费因子,计费时长
#         timestamp, custId, factor, duration = input().split(",")
#
#         # 初始化客户的日志容器（对象作为容器）
#         if custId not in logs:
#             logs[custId] = {}
#
#         key = timestamp + factor
#
#         # 若日志的客户标识、时间戳、计费因子三要素相同，则认为日志重复，不加入重复日志
#         if key in logs[custId]:
#             continue
#
#         # 日志不重复，则记录日志信息[计费因子，计费时长]
#         logs[custId][key] = (factor, int(duration))
#     # 计费因子的数量
#     m = int(input())
#
#     # key是计费因子，val是计费因子单价
#     unit_price = {}
#     for _ in range(m):
#         # 计费因子,单价
#         factor, price = input().split(",")
#         unit_price[factor] = int(price)
#
#     # 客户花费,key是客户id，val是客户所有log的花费之和
#     fees = {}
#
#     for custId in logs:
#         for factor, duration in logs[custId].values():
#             # 计费时长（范围为0-100), 当计费时长不在范围内要认为是计费日志有问题，当成计费为 0 处理
#             if duration > 100 or duration < 0:
#                 duration = 0
#
#             # 计费因子查不到时认为计费因子单价是0
#             price = unit_price.get(factor, 0)
#
#             fees[custId] = fees.get(custId, 0) + price * duration
#
#     # 结果按照客户标识（fees的key）进行升序
#     for custId in sorted(fees.keys()):
#         print(f"{custId},{fees[custId]}")
#
#
# # 算法调用
# solution()


# # 华为OD机试 - 部门人力分配
#
# M = int(input())
# requirements = list(map(int,input().split(' ')))
# requirements.sort()
# def cal(mid:int):
#     l=0
#     r=len(requirements)-1
#     count=0
#     while l<=r:
#         if requirements[l] + requirements[r]<=mid:
#             l+=1
#         r-=1
#         count += 1
#     #print(f'count={count}')
#     return M >= count
#
#
# def getResult():
#
#     low = requirements[-1]
#     high = requirements[-1]+requirements[-2]
#     ans = high
#     while low<=high:
#         mid = (low + high) // 2
#         #print(mid,low,high)
#         if cal(mid):
#             ans = mid
#             high = mid-1
#         else:
#             low = mid+1
#     return mid
#
# print(getResult())


# # 华为校招机试 - 通话不中断的最短路径（20231220）bfs + 动态规划
# # 输入获取
# th = int(input())  # 信号不中断传播的门限Th
# m, n = map(int, input().split())  # 矩阵行数，矩阵列数
# k = int(input())  # 基站个数
# queue = [tuple(map(int, input().split())) for _ in range(k)]  # [(基站所在行号,基站所在列号,基站的初始信号强度)]
#
# grid = [[0] * n for _ in range(m)]  # grid[i][j]代表(i,j)位置的信号强度
# offsets = ((0, -1), (0, 1), (-1, 0), (1, 0))
#
#
# def bfs():  # 广度搜索初始化网格矩阵
#     while len(queue) > 0:
#         x, y, strong = queue.pop(0)
#
#         grid[x][y] = strong
#
#         for offsetX, offsetY in offsets:
#             newX = x + offsetX
#             newY = y + offsetY
#
#             # 如果新位置越界，或者新位置的信号强度大于strong-1，则无法进入新位置
#             if newX < 0 or newX >= m or newY < 0 or newY >= n or grid[newX][newY] >= strong - 1:
#                 continue
#
#             grid[newX][newY] = strong - 1
#
#             queue.append((newX, newY, grid[newX][newY]))
#
#
# def getMinStep():
#     # 如果(0,0)到(i,j)位置不可达，则初始化dp[i][j] = MAX_STEP
#     MAX_STEP = m * n
#
#     # dp[i][j] 表示从(0,0)到(i,j)的最短路径距离
#     # 初始时，假设(0,0)到所有位置都不可达
#     dp = [[MAX_STEP] * n for _ in range(m)]
#
#     # (0,0)到自身的可达距离为0
#     dp[0][0] = 0
#
#     for i in range(m):
#         for j in range(n):
#             # 如果(i,j)位置的信号强度<门限th, 则(0,0)到(i,j)不可达
#             if grid[i][j] < th:
#                 dp[i][j] = MAX_STEP
#                 continue
#
#             # (0,0)->(i,j)的路径相当于(0,0)->(i-1,j)的路径基础上增加(i-1,j)->(i,j)的1个距离
#             if i > 0:
#                 dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j])
#
#             # (0,0)->(i,j)的路径相当于(0,0)->(i,j-1)的路径基础上增加(i,j-1)->(i,j)的1个距离
#             if j > 0:
#                 dp[i][j] = min(dp[i][j - 1] + 1, dp[i][j])
#
#     # (0,0)到(m-1,n-1)的最短距离为dp[m-1][n-1]
#     ans = dp[m - 1][n - 1]
#
#     if ans >= MAX_STEP:
#         return 0
#     else:
#         return ans
#
#
# # 算法入口
# def solution():
#     # 多源bfs求出矩阵每个位置的信号强度
#     bfs()
#
#     if grid[0][0] < th or grid[m - 1][n - 1] < th:
#         # 如果左上角位置或者右下角位置的信号强度小于门限th, 则:不存在信号不中断的最短路径
#         return 0
#     else:
#         return getMinStep()
#
#
# # 算法调用
# print(solution())

# # 华为校招机试 - 畅玩迪士尼（20210407）dfs
# row, col, t = list(map(int, input().split(' ')))
# path = []
# for i in range(row):
#     path.append(list(map(int, input().split(' '))))
# result = -1
#
# offsets = ((1, 0), (0, 1)) # 深搜的方向
#
#
# def dfs(m, n, count):
#     global result
#
#     if m == row - 1 and n == col - 1:
#         if count <= t:
#             result = max(result, count)
#         return
#     for offset in offsets:
#         new_m = m + offset[0]
#         new_n = n + offset[1]
#         if new_m < row and new_n < col and count + path[new_m][new_n] <= t:
#             dfs(new_m, new_n, count + path[new_m][new_n])
#             if result >= t:
#                 return
#
# dfs(0, 0, path[0][0])
# print(result)

# # 华为校招机试 - 任务实际执行时间计算（20210407）拓扑排序
# task_runtimes = list(map(int, input().split(',')))
# relations = []
# for re in input().split(','):
#     relations.append(list(map(int, re.split('->'))))
#
#
# def getResult():
#     n = len(task_runtimes)
#     inDegree = [0] * n  # 每个任务的入度
#     chs = {}  # 每个任务的子任务（完成该任务才能执行的任务）
#     for ch, fa in relations:
#         inDegree[ch] += 1
#         chs.setdefault(fa, [])  # 收集每个任务的子任务
#         chs[fa].append(ch)
#     queue = [i for i in range(n)]  # 任务执行队列，按照初始顺序执行，将各个任务加入队列
#     time = 0  # 记录时间
#     task_complete_times = [0] * n
#     while len(queue) > 0:
#         taskId = queue.pop(0)
#         if inDegree[taskId] > 0:
#             queue.append(taskId)  # 回到队列尾重新排队
#         else:
#             time += task_runtimes[taskId]  # 加上该任务的执行时间
#             task_complete_times[taskId] = time  # 记录任务实际执行时间
#             if chs.get(taskId) is not None:  # 若该任务的子任务不是空的，那么子任务的入度都要减去1
#                 for ch in chs[taskId]:
#                     inDegree[ch] -= 1
#
#     return ','.join(map(str, task_complete_times))
#
#
# print(getResult())

# 华为校招机试 - 游戏分组（20210407） 方法：dfs

# def max_groups(n, preferences):
#     # 使用字典来存储每个小朋友的分组意愿
#     groups = {}
#     for preference in preferences:
#         names = preference.split()
#         #print(names)
#         for name in names:
#             if name not in groups:
#                 groups[name] = set(names) - {name}
#             else:
#                 groups[name] |= set(names) - {name}
#     # 使用深度优先搜索来计算最多可以分成多少组
#     def dfs(person):
#         if person not in visited:
#             visited.add(person)
#             for friend in groups[person]:
#                 dfs(friend)
#
#     visited = set()
#     max_group_count = 0
#     for person in groups:
#         if person not in visited:
#             dfs(person)
#             max_group_count += 1
#
#     return max_group_count
#
#
# if __name__ == "__main__":
#     N = int(input())
#     preferences = []
#     for _ in range(N):
#         preference = input()#.strip()
#         preferences.append(preference)
#         #print(preferences)
#     max_group_count = max_groups(N, preferences)
#     print(max_group_count)

# N = int(input())
# name = {}
# partner = []
# count = 0
# for i in range(N):
#     a,b = input().split(' ')
#     name[a] = name.get(a,'')+b
# relations = list(name.items())
# a= list(name.keys())
# b= list(name.values())
# def part(key,name,count):
#
#     first = key
#     last = name[key]
#
#     if first not in partner:
#         partner.append(first)
#         part(last,name,count)
#
# def summ(count,name):
#     if len(name) == 1:
#         return 1
#     for key in name:
#         if key not in partner:
#             if name[key] in partner:
#                 count -=1
#             part(key,name,count)
#             #print(partner)
#             count+=1
#     return count
# print(summ(count,name))
#


# # 华为校招机试 - 游标匹配问题（20210331）
# s = input()  # 搜索串
# t = input()  # 目标串
# curIndex = int(input())  # 搜索指针当前的位置
# s_char_idxs = {}
#
#
# def recursive(s_idx, t_idx):
#     if t_idx == len(t):
#         return 0
#     t_char = t[t_idx]  # 目标字符
#     minStep = float('inf')
#     for idx in s_char_idxs[t_char]:
#         noCycle = abs(idx - s_idx) + recursive(idx, t_idx + 1)  # 循环递归，不循环的向右寻找
#         cycle = len(s) - abs(idx - s_idx) + recursive(idx, t_idx + 1)  # 向左查找，查到第一位再向左可以跳到最后一位
#         minStep = min(minStep, cycle, noCycle)
#     return minStep
#
#
# def solution():
#     for i, c in enumerate(s):
#         s_char_idxs.setdefault(c, set())
#         s_char_idxs[c].add(i)
#     return recursive(curIndex, 0)
#
#
# print(solution())


# #  华为校招机试 - 猜帽子数量（20210331） 贪心算法
# import math
#
# hat = eval(input())
#
#
# def total():
#     ans = 0
#     dex = {}
#     for num in hat:
#         dex[num] = dex.get(num, 0) + 1
#     for key in dex.keys():
#         count = key + 1  # count是戴相同帽子的员工数量
#         ans += math.ceil(dex[key] / count) * count  # ceil为向上取整,ans取值为,如果有x个人都有y个相同的帽子，那么分组，比如4个人都有其他2人有相同帽子，
#         # 那就3个人为一组，剩下一个人为一组，这个一个人需要再补充两个人，也就相当于一共有两组三个人的，所以是向上取整
#     return ans
#
#
# print(total())

## 华为校招机试 - 足球比赛排名（20210331）
# import re
#
# regExp = re.compile(r"^([a-z])-([a-z]) (\d):(\d)$")
# scores = {}
#
# while True:
#     try:
#         matcher = regExp.match(input())
#
#         team1 = matcher.group(1)
#         team2 = matcher.group(2)
#
#         score1 = int(matcher.group(3))
#         score2 = int(matcher.group(4))
#
#         res1 = 0
#         res2 = 0
#
#         if score1 > score2:
#             res1 += 3
#         elif score1 < score2:
#             res2 += 3
#         else:
#             res1 += 1
#             res2 += 1
#
#         scores[team1] = scores.get(team1, 0) + res1
#         scores[team2] = scores.get(team2, 0) + res2
#     except:
#         break
#
# print(",".join(map(lambda x: x[0] + " " + str(x[1]), sorted(list(scores.items()), key=lambda x: (-x[1], x[0])))))
#

# # 华为校招机试 - 发广播（20210310） 并查集实现或者dfs
# def dfs(matrix, visited, node):
#     visited[node] = True
#     for i in range(len(matrix[node])):
#         if matrix[node][i] == '1' and not visited[i]:
#             dfs(matrix, visited, i)
#
# def min_broadcast_stations(matrix):
#     n = len(matrix)
#     visited = [False] * n
#     count = 0
#     for i in range(n):
#         if not visited[i]:
#             dfs(matrix, visited, i)
#             count += 1
#     return count
#
# if __name__ == "__main__":
#     matrix_str = input().strip().split(',')
#     matrix = [list(row) for row in matrix_str]  # 其实这句没必要
#     min_stations = min_broadcast_stations(matrix)
#     print(min_stations)


# # gpt优化 华为校招机试 - 挑选货物（20210310）
# def count_selections(N, K, M):
#     total_selections = 0
#     prefix_sum = 0
#     prefix_sum_frequency = {0: 1}  # 初始前缀和为0的频率为1
#
#     for i in range(N):
#         prefix_sum += M[i]
#         remainder = prefix_sum % K
#         total_selections += prefix_sum_frequency.get(remainder, 0)
#         prefix_sum_frequency[remainder] = prefix_sum_frequency.get(remainder, 0) + 1
#
#     return total_selections
#
# # 读取输入
# N, K = map(int, input().split())
# M = list(map(int, input().split()))
#
# # 计算挑选方式并输出结果
# result = count_selections(N, K, M)
# print(result)

# 暴力算法，超时
# def count_selections(N, K, M):
#     total_selections = 0
#     for i in range(len(M)):
#         for j in range(1, len(M)+1):
#             if sum(M[i:i+j]) % K == 0 and i+j<=len(M):
#                 print(f'i,j={i,i+j}')
#                 total_selections += 1
#     return total_selections
#
# # 读取输入
# N, K = map(int, input().split())
# M = list(map(int, input().split()))
#
# # 计算挑选方式并输出结果
# result = count_selections(N, K, M)
# print(result)


# #  华为校招机试 - 字母替换（20210310）
# ming = input()
# hong = input()
# max_count = int(input())
#
#
# def getResult():
#     min_count = max_count + 1
#     for i in range(len(hong) - len(ming) + 1):
#         count = 0
#         for j in range(len(ming)):
#             if ming[j] != hong[j + i]:
#                 count += 1
#                 if count > min_count:
#                     break
#         #if count<min_count:
#             #min_count = count
#         min_count = min(count,min_count)
#     if min_count > max_count:
#         return 0
#     return min_count
#
#
# print(getResult())
#


# # 华为校招机试 - 找磨损度最高和最低的硬盘（20231220）
# endurances = list(enumerate(map(int, input().split())))
# endurances.sort(key=lambda x: x[1])
# result = []
# for i in range(len(endurances)):
#     result.append(endurances[i][0])
# real=[]
# for i in range(len(result)-1,len(result)-4,-1):
#     real.append(result[i])
#     real.sort()
# print(real[0],real[1],real[2])
# real=[]
# for i in range(0,3):
#     real.append(result[i])
#     real.sort()
#
# print(real[0],real[1],real[2])


# print(e[:3])


# # 华为校招机试 - 循环依赖（20240320）
# inDegree = {}  # 记录每个点的入度值
# outDegree = {}  # 记录每个点的出度值
# next = {}  # 记录每个点的下游点（一个元素可以依赖于多个元素）
# prev = {}  # 记录每个点的上游点（一个元素也可被多个元素依赖）
#
# # 输入获取
# N = int(input())  # 依赖关系的个数
#
# for _ in range(N):
#     tmp = list(map(int, input().split()))
#
#     n = tmp[0]  # 第一个数 n 表示后面有 n 个元素
#     a = tmp[1]  # 第二个数为元素编号 a
#
#     for i in range(2, n + 1):  # 后面n-1个数为 a 依赖的元素编号
#         # a 依赖于 b, 根据题目用例图示来看，a和b关系为： a -> b
#         b = tmp[i]
#
#         # 初始化a,b的出度
#         outDegree.setdefault(a, 0)
#         outDegree.setdefault(b, 0)
#
#         # 初始化a,b的入度
#         inDegree.setdefault(a, 0)
#         inDegree.setdefault(b, 0)
#
#         # a的出度+1, b的入度+1
#         outDegree[a] += 1
#         inDegree[b] += 1
#
#         # a的next加入b, b的prev加入a
#         next.setdefault(a, set())
#         next[a].add(b)
#
#         prev.setdefault(b, set())
#         prev[b].add(a)
#
#
# # 拓扑排序
# def removeZeroDegree(degree, rela):
#     """
#     拓扑排序
#     :param degree: 入度或出度
#     :param rela: next或prev
#     :return:
#     """
#     queue = []
#
#     for k in degree:
#         if degree[k] == 0:
#             queue.append(k)
#
#     while len(queue) > 0:
#         ch = queue.pop(0)
#
#         if ch in rela:
#             fathers = rela[ch]
#
#             for fa in fathers:
#                 degree[fa] -= 1
#
#                 if degree[fa] == 0:
#                     queue.append(fa)
#
#
# # 算法入口
# def solution():
#     cycle = set()
#
#     # 剥离入度为0的点
#     removeZeroDegree(inDegree, next)
#     # 剥离出度为0的点
#     removeZeroDegree(outDegree, prev)
#
#     for k in inDegree:
#         # 最后的循环依赖中的点，必然入度、出度都不为0
#         if inDegree[k] != 0 and outDegree[k] != 0:
#             cycle.add(k)
#
#     # 找出循环依赖中最小值点
#     start = min(cycle)
#
#     # 记录循环依赖顺序
#     res = [start]
#
#     cur = start
#     isFirst = True
#     while isFirst or cur != start:
#         isFirst = False
#
#         # 找出当前点的下一个点k
#         for k in next[cur]:
#             # 下一个点可能有多个，但是必然只有一个点是循环依赖中的点，因为题目说：假定总是存在唯一的循环依赖
#             if k in cycle:
#                 # k就是循环依赖的下一个点
#                 res.append(k)
#                 cur = k
#                 break
#
#     return " ".join(map(str, res))
#
#
# # 算法调用
# print(solution())

# 华为校招机试 - 栈数据合并（20240320）
# list1 = list(map(int,input().split()))
# stack = [list1[0]]
# def push(num,stack):
#     total = num
#     for i in range(len(stack)-1,-1,-1):
#         total -= stack[i]
#         if total == 0:
#             del stack[i:]
#             push(num*2,stack)
#             return
#         elif total <0:
#             break
#     stack.append(num)
# def solution():
#     for i in range(1,len(list1)):
#         push(list1[i],stack)
#     stack.reverse()
#     return ' '.join(map(str,stack))
#
# print(solution())

# # 华为校招机试 - 计算座位最大利用数（20240320）
# m, n, x = list(map(int, input().split()))  # 列车座位数量、停靠站点数量和预定乘客数量
# reserve = [list(map(int, input().split())) for _ in range(x)]
# ans = 0
#
#
# def check(path):  # 检查最大座位利用数
#     useRatio = 0
#     seats = [0] * m
#     for i in path:
#         start, end = reserve[i]
#         is_Find = False
#         for j in range(m):
#             if start >= seats[j]:
#                 seats[j] = end
#                 useRatio += end - start
#                 is_Find = True
#                 break
#         if not is_Find:  # 一个乘客没找到位置，该组合不成立
#             return 0
#     return useRatio
#
#
# def dfs(index, path: list):  # 找出所有乘客组合
#     global ans
#     if len(path) > 0:
#         ans = max(ans, check(path))
#     if index >= x:  # 注意这一步
#         return
#     for i in range(index, x):
#         path.append(i)
#         dfs(i + 1, path)
#         path.pop()
#     return ans
#
#
# def solution():
#     reserve.sort(key=lambda x: x[0])  # 注意这一步，预定记录按照上车站编号升序
#     return dfs(0, [])
#
#
# print(solution())

# 华为校招机试 - 整数分解结果的枚举（20240131）(回溯算法)
# n = int(input())
# list1 = []  # list1为n的所有因数（包括自身）
# for i in range(2, n ):
#     if n % i == 0:
#         list1.append(i)
# def backtracking(n, list1, result: list, path: list, index,total):
#     if total>n:
#         return
#     if total == n:
#         result.append(path[:])
#         return
#     for i in range(index, len(list1)):
#         total*=list1[i]
#         path.append(list1[i])
#         backtracking(n, list1, result, path, i,total)
#         path.pop()
#         total = total/list1[i]
#
#
# result = []
# backtracking(n, list1, result, [], 0,1)
# for i in range(len(result)):
#     a = ''
#     for num in result[i]:
#         a = a+str(num)+'*'
#     a = a[:-1]
#     print(f'{n}={a}')
# print(f'{n}={n}')

# 华为校招机试 - 大礼包（20240131） 滑动窗口解法
# # 输入获取
# n, x = map(int, input().split())
# d = list(map(int, input().split()))
#
#
# # 等差数列求和公式，首项为1，公差为1，n为数列长度（项数）
# def diffSum(an):
#     return (1 + an) * an // 2
#
#
# # 算法入口
# def solution():
#     # gold[i]表示第i个月一共有多少金币
#     gold = list(map(lambda x: diffSum(x), d))
#
#     # 记录题解
#     ans = 0
#
#     # l,r指针用于扫描d数组
#     l = 0
#     r = 0
#
#     days = 0  # 第l月 ~ 第r月总共天数
#     total = 0  # 第l月 ~ 第r月总共金币数
#
#     while r < n * 2:
#         # r % n 用于保证r指针第二轮循环时从头开始扫描
#         days += d[r % n]
#         total += gold[r % n]
#
#         # 如果第l月 ~ 第r月总共天数days 超过了 需要的连续登录天数x
#         if days >= x:
#             # 由于我们只需要连续登录x天，因此有days-x天是无用的，这days-x天我们应该尽可能选择金币数少的时间段，而金币数少的的时间段就是第l个月的前days-x天
#             # 但是我们需要保证第l月的天数d[l] >= days-x，否则我们无法在d[l]天内挖去前days-x天
#             while days - x > d[l]:
#                 # 如果d[l]不足days-x，则去除第l月
#                 days -= d[l]
#                 total -= gold[l]
#
#                 l += 1
#
#                 # 防止l越界
#                 if l >= n:
#                     return ans
#
#             # 我们只需要连续登录x天，而days-x就是多出来的天数，这部分天数对应的金币要减去，为了保证获得最大金币数，则days-x天应该选择金币数少的日子，即第l个月的前days-x天
#             min_remove = diffSum(days - x)
#
#             ans = max(ans, total - min_remove)
#
#         r += 1
#     return ans
#
#
# # 算法调用
# print(solution())


# 华为校招机试 - 大礼包（20240131） 暴力解法
# n,x = list(map(int,input().split())) #连续登录x天
# date = list(map(int,input().split()))
# list_1=[] # 我们所需要的list，最终用来寻找最大值
# for i in range(n):
#     for j in range(1,date[i]+1):
#         list_1.append(j)
# result=0
# print(list_1)
# for i in range(len(list_1)):
#     result1=0
#     for j in range(i,i+x):
#         if j >= len(list_1):
#             j = j%len(list_1)
#         result1+=list_1[j]
#     result = max(result1,result)
# print(result)


# 华为校招机试 - 找出最可疑的嫌疑人（20240131）
# people = list(map(int, input().split(',')))
# def solution():
#     target = len(people)//2 + len(people)%2
#     a = dict()
#     for person in people:
#         if person in a:
#             a[person]+=1
#         else:
#             a[person]=1
#     for key in a:
#         if a[key]>=target:
#             return key
#     return 0
#
# print(solution())

# 华为校招机试 - 新能源汽车充电桩建设策略（20240124）
# 输入获取
# import sys
#
# n = int(input())
# station = list(map(int, input().split()))
# r = int(input())
# k = int(input())
#
#
# # 树状数组实现类
# class BIT:
#     def __init__(self, n):
#         self.c = [0] * n  # 树状数组
#
#     # lowbit返回的是c[i]控制的区间长度
#     def lowbit(self, i):
#         # 区间长度就是 i 的二进制表示下最低位的1以及它后面的0构成的数值
#         # 例如 i = 20, 其二进制表示位 10100，末尾有2个0，区间长度即为2^2
#         # 而 i & (-i) 公式计算可得 i 二进制表示下最低位的1以及它后面的0构成的数值
#         return i & (-i)
#
#     # 点更新
#     def add(self, i, z):
#         # 更新c[i]及其后继
#         while i < len(self.c):
#             self.c[i] += z
#             # c[i]的直接后继为 c[i + lowbit(i)]
#             i += self.lowbit(i)
#
#     # 前缀和
#     def query(self, i):
#         preSum = 0
#
#         # 累加c[i]及其前驱
#         while i > 0:
#             preSum += self.c[i]
#             #  c[i]的直接前驱为 c[i - lowbit(i)]
#             i -= self.lowbit(i)
#
#         return preSum
#
#
# def check(limit, remain):
#     """
#     只有k个充电桩用来分配，是否能保证分配后的每个区域得充电桩数量都>=limit
#     :param limit: 每个区域的充电桩最少数量
#     :param remain: 剩余可用充电桩数量, 初始值为k
#     :return: 是否能保证每个区域得充电桩数量都>=limit
#     """
#
#     # 创建树状数组
#     bit = BIT(n + 1)
#
#     # 初始化树状数组
#     for i in range(1, n + 1):
#         # 树状数组需要创建为n+1长度，因为要预留为c[i]中i需要大于0，否则add操作会产生死循环
#         # 原因是add操作过程会不断向前找前驱，而前驱节点得索引为 i - lowbit(i)，而 lowbit(0) == 0,因此i - lowbit(i)结果还是i，无法向前
#         bit.add(i, station[i - 1])
#
#     # 注意，此时是基于树状数组来进行：点更新和区间和求解
#     for i in range(1, n + 1):
#         # 越界检查
#         left = max(1, i - r)
#         right = min(n, i + r)
#
#         # 求解[left, right]区间和sum，即当前区域i的充电桩数量
#         rangeRum = bit.query(right) - bit.query(left - 1)
#
#         # 如果当前区域的充电桩数量sum少于limit
#         if rangeRum < limit:
#             # 则应该给当前区域新增limit - sum个充电桩
#             increment = limit - rangeRum
#
#             # 如果不够分配，则分配失败
#             if remain < increment:
#                 return False
#
#             # 否则，剩余可分配充电桩remain需要减去这部分分配出去的
#             remain -= increment
#
#             # 且我们应该将increment个充电桩都分配到当前区域的右边界充电站上，这样才能保证尽可能后面的区域都能共享到此次新增充电桩
#             bit.add(right, increment)
#
#     return True
#
#
#
# def solution():
#     low = sys.maxsize
#     high = k
#     for s in station:
#         low = min(low, s)
#         high += s
#
#     ans = 0
#
#     while low <= high:
#         mid = (low + high) // 2
#
#         if check(mid, k):
#             ans = mid
#             low = mid + 1
#         else:
#             high = mid - 1
#
#     return ans
#
#
# print(solution())


# 华为校招机试 - 大模型训练
# N, T = map(int, input().split())
# tasks = list(map(int, input().split()))
#
# def check(limit):
#     time = 1
#     total = 0
#     for i in range(N):
#         if total+tasks[i]<=limit:
#             total+=tasks[i]
#         else:
#             total = tasks[i]
#             time += 1
#     return time<=T
#
# def main():
# # 读取输入
#     low = max(tasks)
#     high = sum(tasks)
#     ans = low
#     while low<=high:
#         mid = (low + high) // 2
#         if check(mid):
#             ans = mid
#             high = mid-1
#         else:
#             low=mid+1
#     return ans
#
# print(main())

# 华为校招1 计算小球积分
# balls = input()
# grade = 0
# for i in range(len(balls)):
#     count = 0
#     if balls[i]=='r':
#         grade +=1
#     if balls[i]=='g':
#         grade+=2
#     if balls[i]=='b':
#         grade+=3
#     for j in range(i-1,-1,-1):
#         if balls[j]!=balls[i]:
#             break
#         else:
#             count+=1
#     grade+=count
#
# print(grade)


# # 华为OD机试 - 智能驾驶
# import sys
# m, n = map(int, input().split(","))
# matrix = [list(map(int, input().split(","))) for _ in range(m)]
# offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
#
#
# # 记录路径中位置的几个状态
# class Node:
#     def __init__(self, x, y):
#         self.x = x  # 位置横坐标
#         self.y = y  # 位置纵坐标
#         self.init = 0  # 到达此位置所需的最少初始油量
#         self.remain = 0  # 到达此位置时剩余可用油量
#         self.flag = False  # 到达此位置前有没有加过油
#
#
# # 算法入口
# def bfs():
#     # 如果左上角和右下角不可达，则直接返回-1
#     if matrix[0][0] == 0 or matrix[m - 1][n - 1] == 0:
#         return -1
#
#     # 广搜队列
#     queue = []
#
#     # 起始位置
#     src = Node(0, 0)
#
#     if matrix[0][0] == -1:
#         # 如果起始位置就是加油站，则到达(0,0)位置所需初始油量为0，且剩余可用油量为100，且需要标记已加油
#         src.init = 0
#         src.remain = 100
#         src.flag = True
#     else:
#         # 如果起始位置不是加油站，则到达(0,0)位置所需的初始油量至少为matrix[0][0], 剩余可用油量为0，未加油状态
#         src.init = matrix[0][0]
#         src.remain = 0
#         src.flag = False
#
#     queue.append(src)
#
#     # dist_init[x][y] 用于记录起点 (0, 0) 到达 (x, y) 的所有可达路径中最优路径（即初始油量需求最少的路径）的初始油量
#     # 由于需要记录每个位置的最少需要的初始油量，因此每个位置所需的初始油量初始化为一个较大值
#     dist_init = [[sys.maxsize] * n for _ in range(m)]
#
#     # dist_remain 用于记录起点 (0,0) 到达 (x,y) 的所有可达路径中最优路径（即初始油量需求最少的路径）的最大剩余可用油量
#     # 即如果存在多条最优路径，我们应该选这些路径中到达此位置剩余油量最多的
#     dist_remain = [[0] * n for _ in range(m)]
#
#     # 起点（0,0）到达自身位置（0,0）所需的最少初始油量和最多剩余油量
#     dist_init[0][0] = src.init
#     dist_remain[0][0] = src.remain
#
#     # 广搜
#     while len(queue) > 0:
#         cur = queue.pop(0)
#
#         # 从当前位置cur开始向上下左右四个方向探路
#         for offsetX, offsetY in offsets:
#             # 新位置
#             newX = cur.x + offsetX
#             newY = cur.y + offsetY
#
#             # 新位置越界 或者 新位置是障碍，则新位置不可达，继续探索其他方向
#             if newX < 0 or newX >= m or newY < 0 or newY >= n or matrix[newX][newY] == 0:
#                 continue
#
#             # 如果新位置可达，则计算到达新位置的三个状态数据
#             init = cur.init  # 到达新位置所需的最少初始油量
#             remain = cur.remain  # 到达新位置时还剩余的最多可用油量
#             flag = cur.flag  # 是否加油了
#
#             if matrix[newX][newY] == -1:
#                 # 如果新位置是加油站，则加满油
#                 remain = 100
#                 # 标记加过油了
#                 flag = True
#             else:
#                 # 如果新位置不是加油站，则需要消耗matrix[newX][newY]个油
#                 remain -= matrix[newX][newY]
#
#             # 如果到达新位置后，剩余油量为负数
#             if remain < 0:
#                 if flag:
#                     # 如果之前已经加过油了，则说明到达此路径前是满油状态，因此我们无法从初始油量里面"借"油
#                     continue
#                 else:
#                     # 如果之前没有加过油，则超出的油量（-remain），可以从初始油量里面"借"，即需要初始油量 init + (-remain) 才能到达新位置
#                     init -= remain
#                     # 由于初始油量 init + (-remain) 刚好只能支持汽车到达新位置，因此汽车到达新位置后剩余可用油量为0
#                     remain = 0
#
#             # 如果到达新位置所需的初始油量超过了满油100，则无法到达新位置
#             if init > 100:
#                 continue
#
#             # 如果可达新位置，则继续检查当前路径策略到达新位置(newX, newY)所需的初始油量init是否比其他路径策略更少
#             if init > dist_init[newX][newY]:
#                 # 如果不是，则无需探索新位置(newX, newY)
#                 continue
#
#             # 当前路径策略到达新位置(newX,newY)所需初始油量init更少，或者，init和前面路径策略相同，但是当前路径策略剩余可用油量remain更多
#             if init < dist_init[newX][newY] or remain > dist_remain[newX][newY]:
#                 # 则当前路径策略更优，记录更优路径的状态
#                 dist_init[newX][newY] = init
#                 dist_remain[newX][newY] = remain
#
#                 # 将当前新位置加入BFS队列
#                 nxt = Node(newX, newY)
#                 nxt.init = init
#                 nxt.remain = remain
#                 nxt.flag = flag
#
#                 queue.append(nxt)
#
#     # dist_init[m - 1][n - 1] 记录的是到达右下角终点位置所需的最少初始油量
#     if dist_init[m - 1][n - 1] == sys.maxsize:
#         return -1
#     else:
#         return dist_init[m - 1][n - 1]
#
#
# # 算法调用
# print(bfs())


# # 华为OD机试 - 亲子游戏
# n = int(input())
# queue = []
# candy = [[-1] * n for _ in range(n)]
# matrix = []
# for i in range(n):
#     matrix.append(list(map(int, input().split())))
#
#     for j in range(n):
#         # 妈妈的位置
#         if matrix[i][j] == -3:
#             candy[i][j] = 0
#             queue.append((i, j))
#
# offsets = ((0, -1), (0, 1), (-1, 0), (1, 0))
#
#
# # 算法入口
# def bfs():
#     global queue
#
#     # 记录题解
#     ans = -1
#
#     # bfs 按层扩散
#     while len(queue) > 0:
#         # 记录当前扩散层的点
#         newQueue = []
#
#         # 当前层是否有宝宝所在的点
#         flag = False
#
#         # 源点坐标
#         for x, y in queue:
#             # 向四个方向扩散
#             for offsetX, offsetY in offsets:
#                 # 当前扩散点坐标
#                 newX = x + offsetX
#                 newY = y + offsetY
#
#                 # 当前扩散点坐标越界，或者扩散点是墙，则无法扩散
#                 if newX < 0 or newX >= n or newY < 0 or newY >= n or matrix[newX][newY] == -1:
#                     continue
#
#                 # 当前扩散点坐标对应的糖果数量为-1，说明对应扩散点坐标位置还没有加入到当前扩散层
#                 if candy[newX][newY] == -1:
#                     newQueue.append((newX, newY))  # 加入当前扩散层
#
#                 # 当前扩散点可能会被多个源点扩散到，因此比较保留扩散过程中带来的较大糖果数
#                 # candy[newX][newY] 记录的是当前扩散点获得的糖果数
#                 # candy[x][y] + max(0, matrix[newX][newY]) 记录的是从源点(x,y)带来的糖果数 + (newX,newY)位置原本的糖果数
#                 candy[newX][newY] = max(candy[newX][newY], candy[x][y] + max(0, matrix[newX][newY]))
#
#                 # 如果当前扩散点是宝宝位置，则可以停止后续层级的bfs扩散，因为已经找到宝宝的最短路径长度（即扩散层数）
#                 if matrix[newX][newY] == -2:
#                     ans = candy[newX][newY]
#                     flag = True
#         # 已经找到去宝宝位置的最短路径和最大糖果数，则终止bfs
#         if flag:
#             break
#         # 否则继续
#         queue = newQueue
#     return ans
# # 算法调用
# print(bfs())


# # 华为OD机试 - 跳马 bfs
# import sys
# m, n = map(int, input().split())  # 棋盘行数, 棋盘列数
# grid = [input() for _ in range(m)]  # 棋盘矩阵
# stepGrid = [[0] * n for _ in range(m)]  # 最小步数和矩阵，stepMap[i][j]记录各个马走到棋盘(i,j)位置的最小步数之和
#
# # 记录所有马都可达的公共位置坐标,初始为整个棋盘
# reach = set()
# for i in range(m):
#     for j in range(n):
#         reach.add(i * n + j)
#
# # 马走日的偏移量
# offsets = ((1, 2), (1, -2), (2, 1), (2, -1), (-1, 2), (-1, -2), (-2, 1), (-2, -1))
#
#
# # 广搜
# def bfs(sx, sy, k):
#     global reach
#
#     # 广搜队列
#     # (sx,sy)为马所在初始位置，马到达初始位置需要0步
#     queue = [(sx, sy, 0)]
#
#     # 记录该马可以访问(sx,sy)位置
#     vis = set()
#     vis.add(sx * n + sy)  # 二维坐标一维化
#
#     # k记录该马剩余可走步数
#     while len(queue) > 0 and k > 0:
#         # newQueue记录该马花费相同步数的可达的位置（即BFS按层遍历的层）
#         newQueue = []
#
#         # 按层BFS
#         for x, y, step in queue:
#             for offsetX, offsetY in offsets:
#                 # 马走日到达的新位置
#                 newX = x + offsetX
#                 newY = y + offsetY
#
#                 pos = newX * n + newY
#
#                 # 如果新位置越界或者已访问过，则不能访问
#                 if newX < 0 or newX >= m or newY < 0 or newY >= n or (pos in vis):
#                     continue
#
#                 # 将新位置加入新层
#                 newQueue.append((newX, newY, step + 1))
#
#                 # 该马到达(newX, newY)位置最小步数为step+1, 由于该马首次到达(newX, newY)位置，因此step+1就是最小步数
#                 stepGrid[newX][newY] += step + 1
#
#                 # 记录该马访问过该位置，后续如果该马再次访问该位置，则不是最小步数
#                 vis.add(pos)
#
#         queue = newQueue
#         k -= 1  # 剩余步数减1
#
#     # BFS完后，将公共可达位置reach和当前马可达位置vis取交集，交集部分就是新的公共可达位置
#     reach &= vis
#
#
# # 算法入口
# def getResult():
#     # 遍历棋盘
#     for i in range(m):
#         for j in range(n):
#             # 如果棋盘(i,j)位置是马
#             if grid[i][j] != '.':
#                 # 马的等级
#                 k = int(grid[i][j])
#                 # 对该马进行BFS走日
#                 bfs(i, j, k)
#
#     # 如果所有马走完，发现没有公共可达位置
#     if len(reach) == 0:
#         return -1
#
#     # 记录所有马都可达位置的最小步数和
#     minStep = sys.maxsize
#
#     for pos in reach:
#         x = pos // n
#         y = pos % n
#         # (x,y)是所有马都可达的位置，stepMap[x][y]记录所有马到达此位置的步数和
#         minStep = min(minStep, stepGrid[x][y])
#
#     return minStep
#
#
# # 算法调用
# print(getResult())


#  #  华为OD机试 - 路口最短时间问题 dfs
# import sys
# # 根据三点坐标，确定拐弯方向
# def getDirection(preX, preY, curX, curY, nextX, nextY):
#     """
#     :param preX: 前一个点横坐标
#     :param preY: 前一个点纵坐标
#     :param curX: 当前点横坐标
#     :param curY: 当前点纵坐标
#     :param nextX: 下一个点横坐标
#     :param nextY: 下一个点纵坐标
#     :return: cur到next的拐弯方向， >0 表示向左拐， ==0 表示直行（含调头）， <0 表示向右拐
#     """
#     # 向量 pre->cur
#     dx1 = curX - preX
#     dy1 = curY - preY
#
#     # 向量 cur->next
#     dx2 = nextX - curX
#     dy2 = nextY - curY
#
#     #  两个向量的叉积 >0 表示向左拐， ==0 表示直行（含调头）， <0 表示向右拐
#     return dx1 * dy2 - dx2 * dy1
#
#
# class Solution:
#     def calcTime(self, lights, timePerRoad, rowStart, colStart, rowEnd, colEnd):
#         n = len(lights)
#         m = len(lights[0])
#
#         # 到达位置(i,j)的路径有四个来源方向
#         # dist[i][j][k] 表示从来源方向k到达位置(i,j)所需要的时间，初始化INT_MAX
#         dist = [[[sys.maxsize] * 4 for _ in range(m)] for _ in range(n)]
#
#         # 小顶堆，堆中元素是数组 [前一个位置行号，前一个位置列号，当前位置行号，当前位置列号，到达当前位置需要的时间]
#         # 到达当前位置的时间越小，优先级越高
#         pq = []
#
#         # 四个来源方向到达出发点位置 (rowStart, colStart) 所需时间均为 0
#         for k in range(4):
#             dist[rowStart][colStart][k] = 0
#             # 出发点位置没有前一个位置，因此前一个位置设为(-1,-1)
#             pq.append((-1, -1, rowStart, colStart, 0))
#
#         offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
#
#         # 每次取出最短路
#         while len(pq) > 0:
#             pq.sort(key=lambda x: -x[4])
#             preX, preY, curX, curY, cost = pq.pop()
#
#             # 向四个方向探索
#             for k in range(4):
#                 # 新位置
#                 newX = curX + offsets[k][0]
#                 newY = curY + offsets[k][1]
#
#                 # 新位置越界，则不可进入
#                 if newX < 0 or newX >= n or newY < 0 or newY >= m:
#                     continue
#
#                 # 本题不允许掉头，因此新位置处于掉头位置的话，不可进入
#                 if newX == preX and newY == preY:
#                     continue
#
#                 # 每走一步都要花费 timePerRoad 单位时间
#                 newCost = cost + timePerRoad
#
#                 # 出发的第一步，或者右拐，不需要等待红绿灯，其他情况需要等待红绿灯 lights[curX][curY] 单位时间
#                 if preX != -1 and preY != -1 and getDirection(preX, preY, curX, curY, newX, newY) >= 0:
#                     newCost += lights[curX][curY]
#
#                 # 如果以来源方向k到达位置（newX, newY）花费的时间 newCost 并非更优，则终止对应路径探索
#                 if newCost >= dist[newX][newY][k]:
#                     continue
#
#                 # 否则更新为更优时间
#                 dist[newX][newY][k] = newCost
#                 # 并继续探索该路径
#                 pq.append((curX, curY, newX, newY, newCost))
#
#         # 最终取(rowEnd, colEnd)终点位置的四个来源方向路径中最短时间的作为题解
#         return min(dist[rowEnd][colEnd])
#
#
# # 实际考试时，本题为核心代码模式，即无需我们解析输入输出，因此只需要写出上面代码即可
# if __name__ == '__main__':
#     n, m = map(int, input().split())
#     lights = [list(map(int, input().split())) for _ in range(n)]
#     timePerRoad = int(input())
#     rowStart, colStart = map(int, input().split())
#     rowEnd, colEnd = map(int, input().split())
#     print(Solution().calcTime(lights, timePerRoad, rowStart, colStart, rowEnd, colEnd))


# # 华为OD机试 - 可以组成网络的服务器
# n, m = map(int, input().split())
# matrix = [list(map(int, input().split())) for _ in range(n)]
# offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
#
#
# def bfs(i, j):
#     count = 1
#     matrix[i][j] = 0
#
#     queue = [[i, j]]
#
#     while len(queue) > 0:
#         x, y = queue.pop(0)
#
#         for offsetX, offsetY in offsets:
#             newX = x + offsetX
#             newY = y + offsetY
#
#             if n > newX >= 0 and m > newY >= 0 and matrix[newX][newY] == 1:
#                 count += 1
#                 matrix[newX][newY] = 0
#                 queue.append([newX, newY])
#
#     return count
#
# # 算法入口
# def getResult():
#     ans = 0
#
#     for i in range(n):
#         for j in range(m):
#             if matrix[i][j] == 1:
#                 ans = max(ans, bfs(i, j))
#
#     return ans
#
#
# # 算法调用
# print(getResult())


# # 华为OD机试 - 考古学家 dfs
# # 输入获取
# n = int(input())
# arr = input().split()
#
# # 全局变量
# path = []
# used = [False] * n
# cache = set() # 集合里不能有重复元素
#
#
# # 全排列求解
# def dfs():
#     if len(path) == n:
#         cache.add("".join(path))
#         return
#
#     for i in range(n):
#         if used[i]:
#             continue
#
#         # 树层去重
#         if i > 0 and arr[i] == arr[i - 1] and not used[i - 1]:
#             continue
#
#         path.append(arr[i])
#         used[i] = True
#         dfs()
#         used[i] = False
#         path.pop()
#
#
# # 算法入口
# def getResult():
#     # 排序是为了让相同元素相邻，方便后面树层去重
#     arr.sort()
#     dfs()
#
#     # 输出石碑文字的组合（按照升序排列）
#     for v in sorted(list(cache)):
#         print(v)
#
#
# # 算法调用
# getResult()


# # 华为OD机试 - 解密犯罪时间 dfs+正则
# import re
#
# reg = re.compile("(([01][0-9])|([2][0-3]))[0-5][0-9]")
#
# # 输入获取
# hour, minute = input().split(":")
#
#
# def dfs(arr, path, res):
#     if len(path) == 4:
#         timeStr = "".join(path)
#         if reg.search(timeStr) is not None:
#             res.append(timeStr)
#         return
#
#     for i in range(len(arr)):
#         path.append(arr[i])
#         dfs(arr, path, res)
#         path.pop()
#
#
# # 算法入口
# def getResult():
#     arr = list(hour)
#     arr.extend(list(minute))
#
#     arr = list(set(arr))
#
#     res = []
#     dfs(arr, [], res)
#     res.sort()
#
#     index = res.index(hour + minute)
#
#     if index == len(res) - 1:
#         recentTime = res[0]
#     else:
#         recentTime = res[index + 1]
#
#     ans = list(recentTime)
#     ans[1] += ":"
#     return "".join(ans)
#
#
# # 调用算法
# print(getResult())


# # 华为OD机试 - 小华地图寻宝 dfs/bfs -- dfs
# import sys
#
# sys.setrecursionlimit(5000)
# m, n, k = list(map(int, input().split(' ')))
#
# offsets = ((-1, 0), (1, 0), (0, 1), (0, -1))
# flags = [[False] * n for _ in range(m)]
# ans = 0
# digitSums = [0] * (max(m, n))  # 该数组索引是原始数，值是原始数对应的数位和
# for i in range(len(digitSums)):
#     num = i
#     while num > 0:
#         digitSums[i] += num % 10
#         num //= 10
#
#
# def dfs(x: int, y: int):
#     global ans
#     if x < 0 or x >= m or y < 0 or y >= n:  # 对应位置越界
#         return
#     if digitSums[x] + digitSums[y] > k:  # 数位和超过k则不能进入
#         return
#     if flags[x][y]:
#         return
#     flags[x][y] = True
#
#     ans += 1
#     for offset in offsets:
#         new_x = x + offset[0]
#         new_y = y + offset[1]
#         dfs(new_x, new_y)
#
#
# dfs(0, 0)
# print(ans)

# # 华为OD机试 - 连续出牌数量
# nums = list(map(int, input().split(' ')))
# color = list(input().split(' '))
#
# a = list(zip(nums, color))
# a.sort()
# print(a)
# ans = 0
#
#
# def backtracking(a, selected: tuple, count):
#     global ans
#     count += 1
#     ans = max(count, ans)
#     if count >= len(a):
#         return
#     for i in range(len(a)):
#
#         if (a[i][0] == selected[0] or a[i][1] == selected[1]) and not flags[i]:
#             flags[i] = True
#             backtracking(a, a[i], count)
#             flags[i] = False
#             #count -= 1  #不需要count-1
#
#
# for i in range(len(a)):
#     flags = [False] * len(a)
#     flags[i] = True
#     backtracking(a, a[i], 1)
# print(ans)

# # 华为OD机试 - 项目排期 回溯算法+二分法
# works = list(map(int, input().split(' ')))  # 任务量
# n = int(input())  # 项目组人员
#
#
# def check(index, buckets, limit):
#     if index == len(works):  # 如果任务被取完了，代表可以完成全部任务
#         return True
#
#     selected = works[index]  # 当前进行的任务
#     for i in range(len(buckets)):
#         if i > 0 and buckets[i] == buckets[i - 1]:
#             # 剪枝优化
#             continue
#         if selected + buckets[i] <= limit:
#             buckets[i] += selected
#             if check(index + 1, buckets, limit):
#                 return True
#             buckets[i] -= selected
#     return False
#
#
# def solution():
#     works.sort(reverse=True)
#     low = max(works)
#     high = sum(works)
#     ans = high  # 记录题解
#     while low <= high:
#         mid = (low + high) // 2
#         if check(0, [0] * n, mid):
#             ans = mid  # 此时是一个可行解，但不一定是最优解，去尝试更优解
#             high = mid - 1
#         else:
#             low = mid + 1
#     return ans
#
#
# print(solution())

# ## 华为OD机试 - 字符串拼接 回溯算法
# chr, n = list(input().split(' '))
# n = int(n)
# result = []
# flags = [False] * len(chr)
#
#
# def backtracking(chr, path: list, result: list, pre):
#     if len(path) == n:
#         result.append(path[:])
#         # print(path)
#         return
#     for i in range(len(chr)):
#         if pre >= 0 and chr[i] == chr[pre]:  # 相同的字符不能相邻，pre指向前面一个被选择的字符在path中的位置，i指向当前的
#             continue
#         if flags[i]:
#             continue
#         if i > 0 and chr[i] == chr[i - 1] and not flags[i - 1]:  # 树层去重
#             continue
#         path.append(chr[i])
#         flags[i] = True
#         # print(path)
#         backtracking(chr, path, result, i)
#         path.pop()
#         flags[i] = False
#
#
# chr = list(chr)
# chr.sort()
# backtracking(chr, [], result, -1)
#
#
# # print(result)
# def solution():
#     for c in chr:
#         if c < 'a' or c > 'z':
#             return 0
#     return 1
#
#
# if solution() == 1:
#     print(len(result))
# else:
#     print(0)

# # 华为OD机试 - 田忌赛马 回溯算法
# a = list(map(int, input().split(' ')))
# b = list(map(int, input().split(' ')))
# result = []
# path = []
# used = [False] * len(a)
# a.sort()
#
#
# def backtracking(a: list, path: list, result: list, used: list):
#     if len(path) == len(a):
#         result.append(path[:])
#
#     for i in range(len(a)):
#         if used[i]:
#             continue
#         if i > 0 and a[i] == a[i - 1] and not used[i - 1]:  # 树层去重，关键关键关键*************
#             continue
#         path.append(a[i])
#         used[i] = True
#         backtracking(a, path, result, used)
#         path.pop()
#         used[i] = False
#
#
# if len(a) > 1:
#     backtracking(a, path, result, used)
#     ans = dict()
#
#     for i in range(len(result)):
#         count = 0
#         for j in range(len(b)):
#             if result[i][j] > b[j]:
#                 count += 1
#         ans[count] = ans.get(count, 0) + 1
#     m = []
#     for key in ans:
#         m.append(key)
#     m.sort()
#     print(ans[m[-1]])
# else:
#     print(1)

# # 华为OD机试 - 数字排列 回溯算法
# nums = list(map(int,input().split(',')))
# def if_right(nums:list):  #  判断输入4个数字是否符合要求
#     if len(nums)!=4:
#         return False
#     if 2 in nums and 5 in nums:
#         return False
#     if 6 in nums and 9 in nums:
#         return False
#     if 0 in nums:
#         return False
#     for i in range(3):
#         for j in range(i+1,4):
#             if nums[i]==nums[j]:
#                 return False
#     return True
#
#
# def backtracking(nums,path,result:list):
#     if len(path)>0:
#         result.append(path[:])
#     for i in range(len(nums)):
#         if nums[i] not in path:
#             if 2 in path and nums[i]==5:
#                 continue
#             if 5 in path and nums[i] == 2:
#                 continue
#             if 6 in path and nums[i]==9:
#                 continue
#             if 9 in path and nums[i]==6:
#                 continue
#             path.append(nums[i])
#             backtracking(nums,path,result)
#             path.pop()
#
# max_num = max(nums)
#
# result = []
# ans = []
# if if_right(nums):
#     if 2 in nums and nums.index(2) <= 3:
#         nums.append(5)
#     if 5 in nums and nums.index(5) <= 3:
#         nums.append(2)
#     if 6 in nums and nums.index(6) <= 3:
#         nums.append(9)
#     if 9 in nums and nums.index(9) <= 3:
#         nums.append(6)
#     nums.sort()
#     backtracking(nums,[],result)
#     for i in range(len(result)):
#         if len(result[i]) == 1:
#             ans.append(result[i][0])
#         if len(result[i]) == 2:
#
#             ans.append(result[i][0]*10+result[i][1])
#         if len(result[i]) == 3:
#             ans.append(result[i][0]*100+result[i][1]*10+result[i][2])
#         if len(result[i]) == 4:
#             ans.append(result[i][0]*1000+result[i][1]*100+result[i][2]*10+result[i][3])
#     ans.sort()
#     print(ans)
#     print(ans[max_num-1])
# else:
#     print(-1)


# #  华为OD机试 - 数据单元的变化替换 正则匹配，递归
# import re
#
# regexp = re.compile(r"(<.*?>)")  #  只要遇到以<开始，以>结束的就匹配，实际上这两个符号可以替换成任意符号
#
# # 输入获取
# cells = input().split(",")
#
#
# def changeCell(index):
#     # 通过正则匹配出单元格内容中"引用字符串"
#     matchers = regexp.findall(cells[index])
#
#     # reference记录引用字符串
#     for reference in matchers:
#         # 引用单元格编号只能是A~Z的字母，即引用引用字符串长度只能是3，比如"<A>"
#         if len(reference) != 3:
#             return False
#
#         # 引用单元格的编号
#         reference_cellNum = reference[1]
#         # 当前单元格的编号,A的ASCll编码是65
#         self_cellNum = chr(65 + index)
#
#         # 引用单元格编号只能是A~Z的字母，且不能自引用
#         if reference_cellNum < 'A' or reference_cellNum > 'Z' or reference_cellNum == self_cellNum:
#             return False
#         # ord（A）=65
#         # 引用单元格的数组索引， 'A' -> 0  ... 'Z' -> 25
#         reference_index = ord(reference_cellNum) - 65
#
#         # 引用单元格编号不存在
#         if reference_index >= len(cells):
#             return False
#
#         if not changeCell(reference_index):
#             return False
#
#         # 将单元格内容中的引用部分，替换为被引用的单元格的内容
#         cells[index] = cells[index].replace(reference, cells[reference_index])
#
#     return True
#
#
# # 算法入口
# def getResult():
#     if len(cells) > 26:
#         # 最多26个单元格，对应编号A~Z
#         return "-1"
#
#     for i in range(len(cells)):
#         # 替换单元格中的引用
#         if not changeCell(i):
#             # 替换失败，则返回-1
#             return "-1"
#
#         if len(cells[i]) > 100:
#             # 每个单元格的内容，在替换前和替换后均不超过100个字符
#             return "-1"
#
#         if not re.match(r"^[a-zA-Z0-9]+$", cells[i]):
#             # 每个单元格的内容包含字母和数字
#             return "-1"
#     return ",".join(cells)
#
#
# # 算法调用
# print(getResult())


# # 华为OD机试 - 计算三叉搜索树的高度 树的定义
# class TreeNode:
#     def __init__(self, val):
#         self.val = val  # 节点值
#         self.height = None  # 节点所在高度
#         self.left = None  # 左子树
#         self.mid = None  # 中子树
#         self.right = None  # 右子树
#
#
# class Tree:
#     def __init__(self):
#         self.root = None  # 树的根节点
#         self.height = 0  # 树的高度
#
#     def add(self, val):
#         node = TreeNode(val)
#
#         if self.root is None:
#             # 如果树是空的，则当前创建的节点将作为根节点
#             node.height = 1  # 根节点的高度为1
#             self.root = node
#             self.height = 1
#         else:
#             # 如果树不是空的，则从根节点开始比较
#             cur = self.root
#
#             while True:
#                 # 假设创建的节点node是当前节点cur的子节点，则node节点高度值=cur节点高度值+1
#                 node.height = cur.height + 1
#                 # 如果创建的node进入新层，则更新树的高度
#                 self.height = max(node.height, self.height)
#
#                 if val < cur.val - 500:
#                     # 如果数小于节点的数减去500，则将数插入cur节点的左子树
#                     if cur.left is None:
#                         # 如果cur节点没有左子树，则node作为cur节点的左子树
#                         cur.left = node
#                         # 停止探索
#                         break
#                     else:
#                         # 否则继续探索
#                         cur = cur.left
#                 elif val > cur.val + 500:
#                     # 如果数大于节点的数加上500，则将数插入节点的右子树
#                     if cur.right is None:
#                         cur.right = node
#                         break
#                     else:
#                         cur = cur.right
#                 else:
#                     # 如果数大于节点的数加上500，则将数插入节点的中子树
#                     if cur.mid is None:
#                         cur.mid = node
#                         break
#                     else:
#                         cur = cur.mid
#
#
# while True:
#     try:
#         n = int(input())
#         nums = list(map(int, input().split()))
#
#         tree = Tree()
#         for num in nums:
#             tree.add(num)
#
#         print(tree.height)
#     except:
#         break


# # 华为OD机试 - 特殊的加密算法 dfs
# # 输入获取
# n = int(input())  # 明文数字个数
# datas = list(map(int, input().split()))  # 明文
#
# m = int(input())  # 密码本矩阵大小
# secrets = []  # 密码本
#
# # 记录密码本中元素值等于“明文第一个数字”的所有元素的位置
# starts = []
#
# for i in range(m):
#     secrets.append(list(map(int, input().split())))
#     for j in range(m):
#         # 如果密码本(i,j)位置元素指等于明文第一个数字值，则记录(i,j)作为一个出发位置
#         if secrets[i][j] == datas[0]:
#             starts.append((i, j))
#
# # 上，左，右，下偏移量，注意这里的顺序是有影响的，即下一步偏移后产生的密文的字符序必然是：上 < 左 < 右 < 下
# offsets = ((-1, 0), (0, -1), (0, 1), (1, 0))
#
#
# def dfs(x, y, index, path, used):
#     """
#     :param x: 当前位置横坐标
#     :param y: 当前位置纵坐标
#     :param index: datas[index]是将要匹配的明文数字
#     :param path: 路径
#     :param used: 密码本各元素使用情况
#     :return: 是否找到符合要求的路径
#     """
#     if index == n:
#         # 已找到明文最后一个数字，则找到符合要求的路径
#         return True
#     # 否则，进行上、左、右、下四个方向偏移，注意这里的顺序是有影响的，即下一步偏移后产生的密文的字符序必然是：上 < 左 < 右 < 下
#     for offsetX, offsetY in offsets:
#         # 新位置
#         newX = x + offsetX
#         newY = y + offsetY
#
#         # 新位置越界，或者新位置已使用，或者新位置不是目标值，则跳过
#         if newX < 0 or newX >= m or newY < 0 or newY >= m or used[newX][newY] or secrets[newX][newY] != datas[index]:
#             continue
#
#         # 递归进入新位置
#         path.append(f"{newX} {newY}")
#         used[newX][newY] = True
#
#         # 如果当前分支可以找到符合要求的路径，则返回
#         if dfs(newX, newY, index + 1, path, used):
#             return True
#
#         # 否则，回溯
#         used[newX][newY] = False
#         path.pop()
#
#     return False
#
#
# # 算法入口
# def getResult():
#     # 出发位置(x,y)
#     for x, y in starts:
#         # used[i][j]用于记录密码本(i,j)元素是否已使用
#         used = [[False] * m for _ in range(m)]
#         # 出发点位置元素已使用
#         used[x][y] = True
#
#         # 记录结果路径各节点位置
#         # 出发点位置记录
#         path = [f"{x} {y}"]
#
#         # 开始深搜
#         if dfs(x, y, 1, path, used):
#             return " ".join(path)
#
#     return "error"
#
#
# # 算法调用
# print(getResult())


#  背包问题
# def test_2_wei_bag_problem1(weight, value, bagweight):
#     # 二维数组
#     dp = [[0] * (bagweight + 1) for _ in range(len(weight))]
#
#     # 初始化
#     for j in range(weight[0], bagweight + 1):
#         dp[0][j] = value[0]
#
#     # weight数组的大小就是物品个数
#     for i in range(1, len(weight)):  # 遍历物品
#         for j in range(bagweight + 1):  # 遍历背包容量
#             if j < weight[i]:
#                 dp[i][j] = dp[i - 1][j]
#             else:
#                 dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight[i]] + value[i])
#
#     return dp[len(weight) - 1][bagweight]
#
#
# if __name__ == "__main__":
#     bagweight, weight, vals = [int(x) for x in input().split()], [int(x) for x in input().split()], [int(x) for x in input().split()]
#     # weight = [1, 3, 4]
#     # value = [15, 20, 30]
#     # bagweight = 4
#
#     result = test_2_wei_bag_problem1(weight, vals, bagweight[1])
#     print(result)


#  动态规划
# class Solution:
#     def uniquePathsWithObstacles(self, obstacleGrid: list[list[int]]) -> int:
#         m,n = len(obstacleGrid),len(obstacleGrid[0])
#         dp = [[0]*n for _ in range(m)]
#
#         for i in range(m):
#             if obstacleGrid[i][0] == 1:
#                 continue
#             dp[i][0] = 1
#         for j in range(n):
#             if obstacleGrid[0][i] == 1:
#                 continue
#             dp[0][j] = 1
#         for i in range(1,m):
#             for j in range(1,n):
#                 if obstacleGrid[i][j]==1:
#                     continue
#                 dp[i][j] = dp[i-1][j] + dp[i][j-1]
#         return dp[m-1][n-1]


# 深度搜索飞地的数量 力扣 1020题,下有优化方法
# class Solution:
#     def __init__(self):
#         self.position = [[-1, 0], [0, 1], [1, 0], [0, -1]]  # 四个方向
#
#     # 深度优先遍历，把可以通向边缘部分的1全部标记成true
#     def dfs(self, grid: list[list[int]], row: int, col: int, visited: list[list[bool]]):
#         for current in self.position:
#             newRow, newCol = row + current[0], col + current[1]
#             #  索引下标越界
#             if newRow < 0 or newRow >= len(grid) or newCol < 0 or newCol >= len(grid[0]):
#                 continue
#             #  当前位置不是1(陆地)或者已经被访问过了
#             if grid[newRow][newCol] == 0 or visited[newRow][newCol]: continue
#             visited[newRow][newCol] = True
#             self.dfs(grid, newRow, newCol, visited)
#
#     def numEnclaves(self, grid: list[list[int]]) -> int:
#         rowSize, colSize, ans = len(grid), len(grid[0]), 0
#         # 标记数组记录每个值为1的位置是否可以到达边界，可以为true，反之为false
#         visited = [[False] * colSize for _ in range(rowSize)]
#         # 搜索左边界和右边界，对值为1的位置进行深度优先遍历
#         for row in range(rowSize):
#             if grid[row][0] == 1:
#                 visited[row][0] = True
#                 self.dfs(grid, row, 0, visited)
#             if grid[row][colSize - 1] == 1:
#                 visited[row][colSize - 1] = True
#                 self.dfs(grid, row, colSize - 1, visited)
#         # 搜索上边界和下边界，对值为1的位置进行深度优先遍历，但是四个角不需要，上面已经遍历过了
#         for col in range(1, colSize - 1):
#             if grid[0][col] == 1:
#                 visited[0][col] = True
#                 self.dfs(grid, 0, col, visited)
#             if grid[rowSize - 1][col] == 1:
#                 visited[rowSize - 1][col] = True
#                 self.dfs(grid, rowSize - 1, col, visited)
#         # 找出矩阵中值为1但是没有被标记过的位置记录答案
#         for row in range(rowSize):
#             for col in range(colSize):
#                 if grid[row][col] == 1 and not visited[row][col]:
#                     ans += 1
#         return ans

# 深度搜索飞地的数量 leetcode 1020题（优化方法）
# class Solution:
#     def numEnclaves(self, grid: List[List[int]]) -> int:
#         m = len(grid)
#         n = len(grid[0])
#
#         def dfs(r, c):
#             if r >= m or r < 0 or c >= n or c < 0 or grid[r][c] == 0: #将与边界连通的陆地全部转化为0（海洋）
#                 return
#             grid[r][c] = 0
#             dfs(r + 1, c)
#             dfs(r - 1, c)
#             dfs(r, c + 1)
#             dfs(r, c - 1)
#
#         for i in range(n):
#             dfs(0, i)
#             dfs(m - 1, i)
#         for i in range(m):
#             dfs(i, 0)
#             dfs(i, n - 1)
#         return sum(row.count(1) for row in grid)

# 岛屿数量，dfs方法（深度搜索）
# class Solution:
#     def numIslands(self, grid: list[list[str]]) -> int:
#         m, n = len(grid), len(grid[0])
#         visited = [[False] * n for _ in range(m)]
#         dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # 四个方向 dfs
#         result = 0
#
#         def dfs(x, y):
#             for d in dirs:
#                 nextx = x + d[0]
#                 nexty = y + d[1]
#                 if nextx < 0 or nextx >= m or nexty < 0 or nexty >= n:  # 越界了，跳过
#                     continue
#                 if not visited[nextx][nexty] and grid[nextx][nexty] == '1':  # 没有访问过的同时是陆地的
#                     visited[nextx][nexty] = True   # 把未访问过的陆地标记为True
#                     dfs(nextx, nexty)
#
#         for i in range(m):
#             for j in range(n):
#                 if not visited[i][j] and grid[i][j] == '1':
#                     visited[i][j] = True
#                     result += 1  # 遇到没访问过的陆地，+1
#                     dfs(i, j)  # 将与其链接的陆地都标记上true
#         return result

#  柱状图中最大的矩形，单调栈
# class Solution:
#     def largestRectangleArea(self, heights: list[int]) -> int:
#         # Monotonic Stack
#         '''
#         找每个柱子左右侧的第一个高度值小于该柱子的柱子
#         单调栈：栈顶到栈底：从大到小（每插入一个新的小数值时，都要弹出先前的大数值）
#         栈顶，栈顶的下一个元素，即将入栈的元素：这三个元素组成了最大面积的高度和宽度
#         情况一：当前遍历的元素heights[i]大于栈顶元素的情况
#         情况二：当前遍历的元素heights[i]等于栈顶元素的情况
#         情况三：当前遍历的元素heights[i]小于栈顶元素的情况
#         '''
#         # 输入数组首尾各补上一个0（与42.接雨水不同的是，本题原首尾的两个柱子可以作为核心柱进行最大面积尝试
#         heights.insert(0, 0)
#         heights.append(0)
#         stack = [0]
#         result = 0
#         for i in range(1, len(heights)):
#             # 情况一
#             if heights[i] > heights[stack[-1]]:
#                 stack.append(i)
#             # 情况二
#             elif heights[i] == heights[stack[-1]]:
#                 #stack.pop()
#                 stack.append(i)
#             # 情况三
#             else:
#                 # 抛出所有较高的柱子
#                 while stack and heights[i] < heights[stack[-1]]:
#                     # 栈顶就是中间的柱子，主心骨
#                     mid_index = stack[-1]
#                     stack.pop()
#                     if stack:
#                         left_index = stack[-1]
#                         right_index = i
#                         width = right_index - left_index - 1
#                         height = heights[mid_index]
#                         result = max(result, width * height)
#                 stack.append(i)
#         return result


#  接雨水
# class Solution:
#     def trap(self, height: list[int]) -> int:
#         stack = [0]
#         result = 0
#         for i in range(1, len(height)):
#             while stack and height[i] > height[stack[-1]]:
#                 mid_height = stack.pop()
#                 if stack:
#                     # 雨水高度是 min(凹槽左侧高度, 凹槽右侧高度) - 凹槽底部高度
#                     h = min(height[stack[-1]], height[i]) - height[mid_height]
#                     # 雨水宽度是 凹槽右侧的下标 - 凹槽左侧的下标 - 1
#                     w = i - stack[-1] - 1
#                     # 累计总雨水体积
#                     result += h * w
#             stack.append(i)
#         return result


# 分发糖果问题（贪心算法）
# class Solution:
#     def candy(self, ratings: list[int]) -> int:
#         count = len(ratings)
#         bull = [1] * len(ratings)
#         if len(ratings) == 1:
#             return 1
#         # 从前向后遍历，处理右侧比左侧评分高的情况
#         for j in range(1, len(ratings)):
#             if ratings[j] > ratings[j-1]:
#                 bull[j] = bull[j-1]+1
#         # 从后向前遍历，处理左侧比右侧评分高的情况
#         for i in range(len(ratings)-2,-1,-1):
#             if ratings[i] > ratings[i+1]:
#                 bull[i] = max(bull[i],bull[i+1]+1) #记住观察此处处理，不需要再+1了如果已经更大的话
#         return sum(bull[:])

# class Solution:
#     def largestSumAfterKNegations(self, nums: list[int], k: int) -> int:
#         nums.sort()
#         for i in range(len(nums)):
#             if nums[i]<0 and k>0:
#                 nums[i] = -nums[i]
#                 k -= 1
#         nums.sort()
#         if k%2==1:
#             nums[0] = -nums[0]
#         return sum(nums)

# 贪心算法
# class Solution:
#     def jump(self, nums: list[int]) -> int:
#         if len(nums)==1:return 0
#         cover = 0 #最大覆盖范围
#         i = 0
#         count = 0 #计数
#         while i <= cover:
#             for j in range(i,cover+1):
#                 cover = max(nums[i]+i,cover)
#                 if cover >= len(nums)-1:
#                     return count+1
#             count += 1
#         return False

# import sys
# for line in sys.stdin:
#     a = line.split()
#     if(int(a[0]) == 0 and int(a[1]) == 0):
#         break
#     print(int(a[0]) + int(a[1]))

# 解数独 二维递归
# class Solution:
#     def solveSudoku(self, board: list[list[str]]) -> None:
#         self.backtracking(board)
#
#     def backtracking(self, board):
#         for i in range(len(board)):  # 遍历行
#             for j in range(len(board[0])):  # 遍历列
#                 if board[i][j] != '.':
#                     continue
#                 for k in range(1, 10):
#                     if self.is_valid(i, j, k, board):  # 判断是否满足条件
#                         board[i][j] = str(k)
#                         if self.backtracking(board): return True  # 递归
#                         board[i][j] = '.'  # 回溯
#                 return False
#         return True
#
#     def is_valid(self, row, col, val, board):
#         # 判断行
#         for i in range(len(board)):
#             if str(val) == board[row][i]:
#                 return False
#         # 判断列
#         for i in range(len(board[0])):
#             if str(val) == board[i][col]:
#                 return False
#         # 判断九宫格
#         start_row, start_col = (row // 3) * 3, (col // 3) * 3
#         for i in range(start_row, start_row + 3):
#             for j in range(start_col, start_col + 3):
#                 if str(val) == board[i][j]:
#                     return False
#         return True

#  N皇后问题
# class Solution:
#     def solveNQueens(self, n: int) -> list[list[str]]:
#         result = []  # 存储最终结果的二维字符串数组
#         chessboard = ['.' * n for _ in range(n)]  # 初始化棋盘
#         self.backtracking(n, 0, chessboard, result)  # 回溯求解
#         return [[''.join(row) for row in solution] for solution in result]
#
#     def backtracking(self, n, row, chessboard, result):
#         if row == n:
#             result.append(chessboard[:])  # 棋盘填满，将当前解加入结果集
#             return
#         for col in range(n):
#             if self.is_valid(row, col, chessboard):
#                 chessboard[row] = chessboard[row][:col] + 'Q' + chessboard[row][col + 1:]  # 放置皇后
#                 self.backtracking(n, row + 1, chessboard, result)  # 递归到下一行
#                 chessboard[row] = chessboard[row][:col] + '.' + chessboard[row][col + 1:]  # 回溯，撤销当前位置皇后
#
#     def is_valid(self, row: int, col: int, chessboard: list[str]):
#         for i in range(row):  # 检查列
#             if chessboard[i][col] == 'Q':  # 剪枝
#                 return False
#         # 检查45度(左上角)
#         i, j = row - 1, col - 1
#         while i >= 0 and j >= 0:
#             if chessboard[i][j] == 'Q':
#                 return False  # 左上方向存在皇后，不合法
#             i -= 1
#             j -= 1
#         # 检查135度
#         i, j = row - 1, col + 1
#         while i >= 0 and j < len(chessboard):
#             if chessboard[i][j] == 'Q':
#                 return False  # 右上方向存在皇后，不合法
#             i -= 1
#             j += 1
#         return True  # 当前位置合法

#  重新安排行程，leet code 332
# class Solution:
#     def findItinerary(self, tickets: list[list[str]]) -> list[str]:
#         tickets.sort()  # 先排序，这样一旦找到第一个可行路径，一定是字母排序最小的
#         results = []
#         used = [0] * len(tickets)
#         path = ['JFK']  # 从JFK出发
#         self.backtracking(tickets, used, path, 'JFK', results)
#         return results[0]
#
#     def backtracking(self, tickets, used, path, cur, results):
#         if len(path) == len(tickets) + 1:  # 终止条件，路径长度等于机票数+1
#             results.append(path[:])  # 只需找到唯一的一个行程，找到后就直接返回True就可以了
#             return True
#         ignore_set = set()
#         for i, ticket in enumerate(tickets):  # 枚举函数，用于将一个可遍历的数据对象组合为一个索引序列，同时列出数据和数据下标
#             if ticket[0] != path[-1] or used[i] or ticket[0] + ticket[1] in ignore_set:
#                 continue
#             if used[i] == 0 and ticket[0] == cur:  # 找到起始机场为cur且未使用过的机票
#                 used[i] = 1  # 记录票使用过了
#                 path.append(ticket[1])  # 处理节点
#                 state = self.backtracking(tickets, used, path, ticket[1], results)
#                 path.pop()
#                 used[i] = 0
#                 if state:
#                     return True  # 只要找到一个可行路径就返回，不继续搜索
#                 else:
#                     ignore_set.add(ticket[0] + ticket[1])  # 剪枝，如果有不满足的相同行程的机票则剪掉
#

#  利用used的含有set去重的全排列问题（有顺序）
# class Solution:
#     def backtracking(self, nums, result, path, used):
#         if len(path) == len(nums):
#             result.append(path[:])
#             return
#         uSet = set()
#         for i in range(0, len(nums)):
#             if nums[i] in uSet:
#                 continue
#             if used[i]:
#                 continue
#             used[i] = True
#             path.append(nums[i])
#             uSet.add(nums[i])
#             self.backtracking(nums, result, path, used)
#             path.pop()
#             used[i] = False
#
#     def permuteUnique(self, nums: list[int]) -> list[list[int]]:
#         result = []
#         self.backtracking(nums, result, [], [False]*len(nums))
#         return result

# 利用set去重
# class Solution:
#     def backtracking(self, nums, result, path, index):
#         if len(path) > 1:
#             result.append(path[:])
#         used = set()
#         for i in range(index, len(nums)):
#             if (path and nums[i] < path[-1]) or nums[i] in used:  # 保证取非递减
#                 continue
#             path.append(nums[i])
#             used.add(nums[i])
#             self.backtracking(nums, result, path, i + 1)
#             path.pop()
#
#     def findSubsequences(self, nums: list[int]) -> list[list[int]]:
#         result = []
#         self.backtracking(nums, result, [], 0)
#         return result

# 有去重的子集问题
# class Solution:
#     def backtracking(self, nums, result, path, index):
#         result.append(path[:])
#         for i in range(index, len(nums)):
#             if nums[i] == nums[i-1] and i>index:# 去重，树往下可以重复，往右不能重复
#                 continue
#             path.append(nums[i])
#             self.backtracking(nums, result, path, i + 1)
#             path.pop()
#
#     def subsetsWithDup(self, nums: list[int]) -> list[list[int]]:
#         nums.sort()
#         result = []
#         self.backtracking(nums, result, [], 0)
#         return result

# 子集问题（回溯算法）
# class Solution:
#     def backtracking(self, nums, path, result, index):
#         result.append(path[:])
#         for i in range(index, len(nums)):
#             path.append(nums[i])
#             self.backtracking(nums, path, result, i+1)
#             path.pop()
#
#     def subsets(self, nums: list[int]) -> list[list[int]]:
#         result = []
#         self.backtracking(nums, [], result, 0)
#         return result

# 回溯解决分组问题
# class Solution:
#     def backtracking(self, s, index, pointNum, result, current):
#         if pointNum == 3:  # 逗号等于3分隔结束
#             if self.is_valid(s, index, len(s) - 1):  # 判断第四段字符串是否合法
#                 current += s[index:]
#                 result.append(current)
#             return
#         for i in range(index, len(s)):
#             if self.is_valid(s, index, i):
#                 sub = s[index:i + 1]
#                 self.backtracking(s, i + 1, pointNum + 1, result, current + sub + '.')
#             else:
#                 break
#
#     def restoreIpAddresses(self, s: str) -> list[str]:
#         result = []
#         self.backtracking(s, 0, 0, result, "")
#         return result
#
#     def is_valid(self, s, start, end):
#         if start > end:
#             return False
#         if s[start] == '0' and start != end:
#             return False
#         num = 0
#         for i in range(start, end + 1):
#             if not s[i].isdigit():
#                 return False
#             num = num * 10 + int(s[i])
#             if num > 255:
#                 return False
#         return True

# # 回溯算法解决组合问题
# class Solution:
#     def combine(self, n: int, k: int) -> list[list[int]]:
#         result = []  # 存放结果集
#         self.backtracking(n, k, 1, [], result)  # 回溯函数
#         return result
#
#     def backtracking(self, n, k, startIndex, path, result):
#         if len(path) == k:
#             result.append(path[:])
#             return
#         for i in range(startIndex, n + 1):  # 可以优化
#             path.append(i)  # 处理节点
#             self.backtracking(n, k, i + 1, path, result)
#             path.pop()  # 回溯，撤销处理的节点

# 滑动窗口最大值
# class Solution:
#     def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
#         queue,res=collections.deque(),[]
#         for i in range(k):
#             while queue and queue[-1]<nums[i]:
#                 queue.pop()
#             queue.append(nums[i])
#         res.append(queue[0])
#         for i in range(k,len(nums)):
#             if queue[0]==nums[i-k]:
#                 queue.popleft()
#             while queue and queue[-1]<nums[i]:
#                 queue.pop()
#             queue.append(nums[i])
#             res.append(queue[0])
#         return res

# 利用栈以及eval来计算逆波兰表达式
# class Solution:
#     def evalRPN(self, tokens: list[str]) -> int:
#         stack = []
#         for item in tokens:
#             if item not in {"+", "-", "*", "/"}:
#                 stack.append(item)
#             else:
#                 first_num, second_num = stack.pop(), stack.pop()
#                 stack.append(int(eval(f'{second_num}{item}{first_num}')))
#         return int(stack.pop())

# 四数之和，双指针法
# class Solution:
#     def fourSum(self, nums: list[int], target: int) -> list[list[int]]:
#         nums.sort()
#         n = len(nums)
#         result = []
#         for i in range(n):
#             if nums[i] > target > 0 and nums[i] > 0:
#                 break  # 剪枝
#             if i > 0 and nums[i] == nums[i - 1]:
#                 continue  # a去重
#             for j in range(i + 1, n):
#                 if nums[i] + nums[j] > target > 0:
#                     break  # 剪枝（可省）
#                 if j > i + 1 and nums[j] == nums[j - 1]:  # 去重
#                     continue
#                 left = j + 1
#                 right = n - 1
#
#                 while left < right:
#                     s = nums[i] + nums[j] + nums[left] + nums[right]
#                     if s == target:
#                         result.append([nums[i], nums[j], nums[left], nums[right]])
#                         while left<right and nums[left]==nums[left+1]:
#                             left +=1
#                         while left < right and nums[right] == nums[right - 1]:
#                             right -= 1
#                         left+=1
#                         right-=1
#                     elif s < target:
#                         left += 1
#                     elif s> target:
#                         right -=1
#         return result
