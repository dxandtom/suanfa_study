# 华为校招机试 - 循环依赖（20240320）
<<<<<<< HEAD

=======
#111
>>>>>>> 33b4b4b (Initial commit)




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

# 华为1 计算小球积分
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

# 回溯算法解决组合问题
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
