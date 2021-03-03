## 7.1 算法总结
有两种常见的动态规划方法，以一维为例：
1. 截止到i的最优状态（可以不包含i）保存到dp[i]
2. 以i为结尾的（子串）的最优状态（必须包含i）保存到dp[i]

## 7.2 基本动态规划：二维
### 413 Arithmetic Slices
比如对于数组[1, 2, 3, 4, 5], 其dp = [0, 0, 1, 2, 3], 表示到该位为止的3位等差子数组的个数，dp之和恰好相当于3位、4位、5位等差子数组个数之和。
```cpp
int numberOfArithmeticSlices(vector<int>& A) {
    int n = A.size();
    if (n < 3) return 0;
    //dp表示截止到i为止连续的元素个数为3的等差子数组的个数
    vector<int> dp(n, 0);
    for (int i = 2; i < n; ++i) {
        if (A[i] - A[i-1] == A[i-1] - A[i-2]) {
            dp[i] = dp[i-1] + 1;
        }
    }
    return accumulate(dp.begin(), dp.end(), 0);
}
```

### 542 01 Matrix
如果对每个点进行广度优先搜索，会导致超时：
```cpp
//对每个点进行广度优先搜索
vector<int> directions{-1, 0, 1, 0, -1};
int m, n;
vector<vector<int>> updateMatrix(vector<vector<int>>& matrix) {
    m = matrix.size();
    n = matrix[0].size();
    vector<vector<int>> ans(m, vector<int>(n, 0));
    for (int i = 0; i < m; ++i) {
        for (int j = 0;j < n; ++j) {
            if (matrix[i][j] == 0) continue;
            ans[i][j] = bfs(matrix, i, j);
        }
    }
    return ans;
}

int bfs(vector<vector<int>>& matrix, int i, int j) {
    vector<vector<bool>> visited(m, vector<bool>(n, false));
    queue<pair<int, int>> q;
    q.push({i, j});
    visited[i][j] = true;
    int dis = 0;
    while (!q.empty()) {
        ++dis;
        int size = q.size();
        while(size--) {
            pair<int, int> node = q.front();
            q.pop();
            for (int k = 0; k < 4; ++k) {
                int x = node.first + directions[k], y = node.second + directions[k + 1];
                if (x < 0 || x >= m || y < 0 || y >= n || visited[x][y]) {
                    continue;
                }
                if (matrix[x][y] == 0) {
                    return dis;
                } else {
                    q.push({x, y});
                    visited[x][y] = true;
                }
            }
        }
    }
    return dis;
}
```

## 7.6 背包问题
有 N 种物品和容量为 W 的背包，想要使得背包装下总价值最高的物品
1. 0-1背包问题： 每种物品仅能选择装0个或者1个，假设第 i 件物品体积为 w，价值为 v：
   1. dp[i][j]表示前i种物品体积不超过j时所能装下的最大价值， dp[i][j] = max(dp[i-1][j], dp[i-1][j-w] + v)
   2. 空间压缩：**逆向遍历** dp[j] = max(dp[j], dp[j-w] + v)
2. 完全背包问题：每种物品均可以拿多次，假设第 i 件物品体积为 w，价值为 v：
   1. dp[i][j]表示前i种物品体积不超过j时所能装下的最大价值， dp[i][j] = max(dp[i-1][j], dp[i][j-w] + v)
   2. 空间压缩：**正向遍历** dp[j] = max(dp[j], dp[j-w] + v)

### 474 Ones and Zeros
空间压缩前：
```cpp
pair<int, int> count(const string& s) {
    int count0 = s.length(), count1 = 0;
    for (const char& c : s) {
        if (c == '1') {
            ++count1;
            --count0;
        }
    }
    return make_pair(count0, count1);
}
int findMaxForm(vector<string>& strs, int m, int n) {
    int size = strs.size();
    //dp表示截止到第strNum个字符串，光用 i 个0 和 j 个1最多能组成几个
    vector<vector<int>> temp(m + 1, vector<int>(n + 1, 0));
    vector<vector<vector<int>>> dp(size + 1, temp);
    int strNum = 0;                                       
    for (const string& str : strs) {
        ++strNum;
        auto [count0, count1] = count(str);
        for (int i = 0; i <= m; ++i) {
            for (int j = 0; j <= n; ++j) {
                if (i >= count0 && j >= count1) {
                    dp[strNum][i][j] = max(dp[strNum-1][i][j],
                                            1 + dp[strNum-1][i-count0][j-count1]);
                } else {
                    dp[strNum][i][j] = dp[strNum-1][i][j];
                }
            }
        }
    }
    return dp[size][m][n];
}
```
## 7.7 字符串编辑 

### 650 2 Keys Keyboard
可能这样更容易理解
```cpp
int minSteps(int n) {
    //dp表示延展到长度i所需要的最小操作数
    vector<int> dp(n + 1);
    for (int i = 2; i <= n; ++i) {
        dp[i] = i;
        for (int j = 2; j < i; ++j) {
            if (i % j == 0) {
                //获得j长度后，操作i/j次可得到i长度，类似从1到i/j
                dp[i] = dp[j] + dp[i/j];
                //没必要继续循环下去
                break;
            }
        }
    }
    return dp[n];
}
```
## 7.8 股票交易
注意状态转移图每个圈代表当天进行了什么操作，每条边代表当天进行**箭头终点**操作后的收益。
## 7.9 练习
### 213 House Robber II
所有房子首尾相接，且不允许连续偷两家，求最多能偷多少。

思路1：用两个vector分别表示抢第一家(不可以抢最后一家)和不抢第一家(可以抢最后一家)
```cpp
int rob(vector<int>& nums) {
    int n = nums.size();
    if (n == 0) return 0;
    if (n == 1) return nums[0];
    //抢第一间房
    vector<int> dp1(n + 1, 0);
    dp1[1] = nums[0];
    //不抢第一间房
    vector<int> dp2(n + 1, 0);
    for (int i = 2; i < n; ++i) {
        dp1[i] = max(dp1[i-1], dp1[i-2] + nums[i-1]);
        dp2[i] = max(dp2[i-1], dp2[i-2] + nums[i-1]);
    }
    //抢第一间房就不能抢最后一间房
    dp1[n] = dp1[n-1];
    //不抢第一间房，还可以抢第二间房
    dp2[n] = max(dp2[n-1], dp2[n-2] + nums[n-1]);
    return max(dp1[n], dp2[n]);
}
```

思路2：空间压缩
```cpp
int rob(vector<int>& nums) {
    int n = nums.size();
    if (n == 0) return 0;
    if (n == 1) return nums[0];
    //抢第一间房
    int pre1_1 = nums[0], pre1_2 = 0, cur1 = pre1_1;
    //不抢第一间房
    int pre2_1 = 0, pre2_2 = 0, cur2 = 0;
    for (int i = 1; i < n - 1; ++i) {
        cur1 = max(pre1_1, pre1_2 + nums[i]);
        pre1_2 = pre1_1;
        pre1_1 = cur1;
        cur2 = max(pre2_1, pre2_2 + nums[i]);
        pre2_2 = pre2_1;
        pre2_1 = cur2;
    }
    //抢第一间房就不能抢最后一间房
    //不抢第一间房，还可以抢第二间房
    cur2 = max(pre2_1, pre2_2 + nums[n-1]);
    return max(cur1, cur2);
}
```

### 53 Maximum Subarray
找出整数串中最大的连续子串之和。

思路1：用dp表示以i为结尾的子串最大和
```cpp
int maxSubArray(vector<int>& nums) {
    int n = nums.size();
    //dp表示以i为结尾的子串最大和
    vector<int> dp(n, 0);
    dp[0] = nums[0];
    for (int i = 1; i < n; ++i) {
        dp[i] = nums[i] + max(dp[i-1], 0);
    }
    return *max_element(dp.begin(), dp.end());
}
```

思路2：空间压缩
```cpp
int maxSubArray(vector<int>& nums) {
    int n = nums.size();
    int sum = nums[0], maxSum = nums[0];
    for (int i = 1; i < n; ++i) {
        sum = nums[i] + max(sum, 0);
        maxSum = max(maxSum, sum);
    }
    return maxSum;
}
```

### 343 Integer Break
给定整数 n， 找出和为 n 的至少两个正整数，使得他们积最大。

思路：对从2到n的所有数计算最大积。
```cpp
int integerBreak(int n) {
    //dp表示截止到i的最大积
    vector<int> dp(n + 1, 0);
    dp[1] = 1;
    dp[2] = 1;
    for (int i = 3; i <= n; ++i) {
        for (int j = 1; j < i; ++j) {
            //dp[2] = 1,但对于后续数来说也可以直接用2
            dp[i] = max(dp[i], (i - j) * j);
            dp[i] = max(dp[i], (i - j) * dp[j]);
        }
    }
    return dp[n];
}
```
### 583 Delete Operation for Two Strings
通过删除字符使得两字符串最终一致所需要的最少操作数

思路：用dp[i][j]记录word1前i个字符想和word2前j个字符一致所需要的最少操作数。要注意修改边缘值
```cpp
int minDistance(string word1, string word2) {
    if (word1.empty()) return word2.length();
    if (word2.empty()) return word1.length();
    int m = word1.length(), n = word2.length();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
    //对边缘部分进行赋值，即word1前0个字符要与word2前j个字符变成一致，需要j步
    for (int i = 0; i <= m; ++i) {
        dp[i][0] = i;
    }
    for (int j = 0; j <= n; ++j) {
        dp[0][j] = j;
    }
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            //各自新增的字符相同，不用删
            if (word1[i-1] == word2[j-1]) {
                dp[i][j] = dp[i-1][j-1];
            } else {
                dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + 1;
            }
        }
    }
    return dp[m][n];
}
```

### 646 Maximum Length of Pair Chain
给定一个数对集合，找出能够形成的最长数对链的长度。可以以任何顺序排列数对集合，但要求chain中的相邻数对(a,b),(c,d)满足c>b

思路1：类似最长递增子序列。首先对pairs进行先右后左的升序排序，并用dp[i]表示以索引i为结尾的chain的最长长度。
```cpp
int findLongestChain(vector<vector<int>>& pairs) {
    int max_length = 1, n = pairs.size();
    if (n <= 1) return n;
    //dp表示以i为结尾的最长链长度
    vector<int> dp(n, 1);
    //以右元素为第一关键字，左元素为第二关键字进行升序排列
    sort (pairs.begin(), pairs.end(), [](vector<int>& a, vector<int>& b) {
        if (a[1] != b[1]) return a[1] < b[1];
        else
            return a[0] < b[0];
    });
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            if (pairs[i][0] > pairs[j][1]) {
                dp[i] = max(dp[i], dp[j] + 1);
            }
        }
        max_length = max(max_length, dp[i]);
    }
    return max_length;
}
```

思路2：首先以左元素为第一关键字升序排列，对于每一个数对，找到dp中合适的位置（第一个右元素比该数对左元素大的索引），如果dp中当前索引恰好能包含该数对，更新当前索引为该数对。如果不能包含(即dp[left][0]<pairs[i][0]且dp[left][1]<pairs[i][1])，则无需更新，因为当前的dp[left]比pairs[i]更适合。最后输出dp长度
```cpp
int findLongestChain(vector<vector<int>>& pairs) {
    int n = pairs.size();
    if (n <= 1) return n;
    //以左元素为第一关键字，右元素为第二关键字进行升序排列
    sort (pairs.begin(), pairs.end(), [](vector<int>& a, vector<int>& b) {
        if (a[0] != b[0]) return a[0] < b[0];
        else
            return a[1] < b[1];
    });
    //dp表示当前最长链
    vector<vector<int>> dp;
    dp.push_back(pairs[0]);
    //由于pairs左元素升序，后入的左元素必定不小于所有已经进入dp的左元素
    for (int i = 1; i < n; ++i) {
        //如果比所有的都大，直接放右边
        if (pairs[i][0] > dp.back()[1]) {
            dp.push_back(pairs[i]);
        } 
        //用二分查找出dp中第一个右元素比pairs[i]左元素大的索引
        else {
            int left = 0, right = dp.size();
            while (left < right) {
            int mid = (left + right) / 2;
                if (dp[mid][1] >= pairs[i][0]) right = mid;
                else left = mid + 1;
            }
            //如果pairs[i]完全被dp[left]包含，更新dp[left]
            if (dp[left][1] > pairs[i][1]) dp[left] = pairs[i];
        }
    }
    return dp.size();
}
```

### 376 Wiggle Subsequence
求摆动子序列的最长长度

思路1：定义结尾元素比前一个元素大的摆动子序列为上升摆动子序列；结尾元素比前一个元素小的摆动子序列为下降摆动子序列。记录到某个索引处为止的up down长度
```cpp
int wiggleMaxLength(vector<int>& nums) {
    int n = nums.size();
    if (n <= 1) return n;
    //分别表示截止到当前索引为止最后一个元素分别为升、降的摆动序列的长度
    vector<int> up(n), down(n);
    up[0] = down[0] = 1;
    for (int i = 1; i < n; ++i) {
        //最后一个元素升，可能导致降->升，up+1
        if (nums[i] > nums[i-1]) {
            up[i] = max(up[i-1], down[i-1] + 1);
            down[i] = down[i-1];
        }
        //最后一个元素降，可能导致升->降，down+1
        else if (nums[i] < nums[i-1]) {
            up[i] = up[i-1];
            down[i] = max(up[i-1] + 1, down[i-1]);
        } 
        else {
            up[i] = up[i-1];
            down[i] = down[i-1];
        }
    }
    return max(up[n-1], down[n-1]);
}
```

思路2：空间压缩
```cpp
int wiggleMaxLength(vector<int>& nums) {
    int n = nums.size();
    if (n <= 1) return n;
    //分别表示截止到当前索引为止最后一个元素分别为升、降的摆动序列的长度
    int up(1), down(1);
    for (int i = 1; i < n; ++i) {
        //最后一个元素升，可能导致降->升，up+1
        if (nums[i] > nums[i-1]) {
            up = max(up, down + 1);
        }
        //最后一个元素降，可能导致升->降，down+1
        else if (nums[i] < nums[i-1]) {
            down = max(up + 1, down);
        } 
    }
    return max(up, down);
}
```

### 494 Target Sum
给定系列整数，求共有多少种不同方法使得通过+/-将所有整数连起来后可以得到目标

思路1：主要问题点在于不允许负索引，所以对所有索引加1000。
```cpp
int findTargetSumWays(vector<int>& nums, int S) {
    int n = nums.size();
    //dp[i][j]表示前i个数得出j的可能性数目
    //由于不能出现负索引，且sum不超过1000，所以索引从-1000取到1000
    //再对所有索引加1000
    vector<vector<int>> dp(n + 1, vector<int>(2001, 0));
    dp[0][1000] = 1;
    for (int i = 1; i <= n; ++i) {
        for (int j = -1000 ; j <= 1000; ++j) {
            if (dp[i-1][j+1000] > 0) {
                dp[i][j+nums[i-1]+1000] += dp[i-1][j+1000];
                dp[i][j-nums[i-1]+1000] += dp[i-1][j+1000];
            }
        }
    }
    return S>1000 ? 0 : dp[n][S+1000];
}
```

思路2：空间压缩
```cpp
int findTargetSumWays(vector<int>& nums, int S) {
    int n = nums.size();
    //dp[i][j]表示前i个数得出j的可能性数目
    //由于不能出现负索引，且sum不超过1000，所以索引从-1000取到1000
    //再对所有索引加1000
    vector<int> dp(2001, 0);
    dp[1000] = 1;
    for (int i = 1; i <= n; ++i) {
            vector<int> next(2001, 0);
        for (int j = -1000 ; j <= 1000; ++j) {
            if (dp[j+1000] > 0) {
                next[j+nums[i-1]+1000] += dp[j+1000];
                next[j-nums[i-1]+1000] += dp[j+1000];
            }
        }
        dp = next;
    }
    return S>1000 ? 0 : dp[S+1000];
}
```

### 714 Best Time to Buy and Sell Stock with Transaction Fee
有手续费，不用冷却的买卖股票

思路：画出状态转移图，注意每条边代表当天进行**箭头终点**操作后的收益。
```cpp
int maxProfit(vector<int>& prices, int fee) {
    int n = prices.size();
    //均表示当前最后一天进行该操作的最大获利
    vector<int> buy(n), sell(n), s1(n), s2(n);
    s2[0] = sell[0] = 0;
    s1[0] = buy[0] = -prices[0];
    for (int i = 1; i < n; ++i) {
        buy[i] = max(s2[i-1], sell[i-1]) - prices[i];
        s1[i] = max(buy[i-1], s1[i-1]);
        sell[i] = max(buy[i-1], s1[i-1]) + prices[i] - fee;
        s2[i] = max(sell[i-1], s2[i-1]);
    }
    //最后一天必须保证手头没有尚未卖出的股票
    return max(sell[n-1], s2[n-1]);
}
```
