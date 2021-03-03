## 6.2 深度优先搜索
对书中树进行深度优先搜索：
1. 栈写法
```cpp
struct Node {
    int val;
    vector<Node *> children;
    Node(int v) {
        val = v;
        children = vector<Node *>{};
    }
    Node(int v, vector<Node*> ch) {
        val = v;
        children = ch;
    }
};

int main() {
    Node n4(4), n3(3), n2(2, vector<Node *>{&n4}), n1(1, vector<Node *>{&n2, &n3});
    stack<Node *> stk;
    stk.push(&n1);
    while (!stk.empty()) {
        Node* node = stk.top();
        stk.pop();
        cout << node->val << "  ";
        for (int i = node->children.size() - 1; i >= 0; --i) {
            stk.push(node->children[i]);
        }
    }
    return 0;
}
```
2. 递归写法
```
struct Node {
    int val;
    vector<Node *> children;
    Node(int v) {
        val = v;
        children = vector<Node *>{};
    }
    Node(int v, vector<Node*> ch) {
        val = v;
        children = ch;
    }
};
void dfs(Node* root) {
    if (root == nullptr) return;
    cout << root->val << "  ";
    for (int i = 0; i < root->children.size(); ++i) {
        dfs(root->children[i]);
    }
    return;
}

int main() {
    Node n4(4), n3(3), n2(2, vector<Node *>{&n4}), n1(1, vector<Node *>{&n2, &n3});
    dfs(&n1);
    return 0;
}
```
3. 对于矩阵的递归写法
```cpp
int m = mat.size(), n = mat[0].size();
void dfs(vector<vector<int>>& mat, int i, int j) {
    if (i < 0 || i >= m || j < 0 || j >= n || visited[i][j]) {
        return;
    }
    ...
    dfs(mat, i - 1, j);
    dfs(mat, i + 1, j);
    dfs(mat, i, j - 1);
    dfs(mat, i, j + 1);
} 
```
### 695 Max Area of Island
栈写法
```cpp
vector<int> direction {-1, 0, 1, 0, -1};

int maxAreaOfIsland(vector<vector<int>>& grid) {
    //m行 n列
    int m = grid.size(), n = m ? grid[0].size() : 0, local_area, area = 0, x, y;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            //找到一个陆地点，对该点进行深度优先搜索
            if (grid[i][j]) {
                local_area = 1;
                grid[i][j] = 0;
                stack<pair<int, int>> island;
                //陆地部分入栈
                island.push({i, j});
                while (!island.empty()) {
                    //搜索栈顶元素的四个邻位
                    auto [r, c] = island.top();
                    island.pop();
                    //依次表示一个邻位,左上右下
                    for (int k = 0; k < 4; ++k) {
                        x = r + direction[k];
                        y = c + direction[k+1];
                        if (x >= 0 && x < m && y >= 0 && y < n && grid[x][y]) {
                            grid[x][y] = 0;
                            ++local_area;
                            island.push({x, y});
                        }
                    }
                }
                area = max(area, local_area);
            }
        }
        
    }
    return area;
}
```
## 6.3 回溯法
如果当前节点（及其子节点）不是需求目标时，回退到原来节点继续搜索。

- 按引用传递状态
- 所有状态修改在递归完成后回改

### 46 Permutations
思路：任意交换第i位与本身及之后的任意元素，并在每一种交换情况下，递归交换第i+1位。注意递归后回改好进入下一次循环并继续交换第i位
```cpp
vector<vector<int>> permute(vector<int>& nums) {
    vector<vector<int>> ans;
    backtracking(nums, 0, ans);
    return ans;
}

void backtracking(vector<int>& nums, int level, vector<vector<int>>& ans) {
    //当调换到最后一位时结束
    if (level == nums.size() - 1){
        ans.push_back(nums);
        return;
    }
    for (int i = level; i < nums.size(); ++i) {
        //将level位与任一位进行调换。level=0时：123,213,321
        swap(nums[i], nums[level]);
        //将level+1位与本身及其后任一位调换
        //本身包含回改操作，故不会改变nums顺序
        backtracking(nums, level + 1, ans);
        //回改
        swap(nums[i], nums[level]);
    }
}
```

### 77 Combinations
因为组合无序，所以count+1位可以直接在 comb[count]+1 = i + 1到 n 中选

### 51 N-Queens
所有左斜、右斜各2n-1个

## 6.4 广度优先搜索
### 934 Shortest Bridge
```cpp
vector<int> direction{-1, 0, 1, 0, -1};
int shortestBridge(vector<vector<int>>& A) {
    int m = A.size(), n = A[0].size();
    queue<pair<int, int>> points;
    bool flipped = false;
    //寻找第一个1, 
    for (int i = 0; i < m; ++i) {
        if (flipped) break;
        for (int j = 0; j < n; ++j) {
            if (A[i][j] == 1) {
                //将所有与输入点相连的1利用dfs变成2，并将所有与该岛相邻的0入队points
                dfs(points, A, m, n, i, j);
                flipped = true;
                break;
            }
        }
    }
    //bfs寻找第二个岛屿
    int x, y;
    int level = 0;
    while(!points.empty()){
        ++level;
        int n_points = points.size();
        while(n_points--) {
            auto [r, c] = points.front();
            points.pop();
            for (int k = 0; k < 4; ++k) {
                x = r + direction[k], y = c + direction[k+1];
                if (x >= 0 && y >= 0 && x < m && y < n) {
                    if (A[x][y] == 2) {
                        continue;
                    }
                    //找到了第二个岛屿
                    if (A[x][y] == 1) {
                        return level;
                    }
                    //将离第一座岛屿level距离的0全部赋值为2，并入队，等待下一轮循环
                    points.push({x, y});
                    A[x][y] = 2;
                }
            }  
        }
    }
    return 0;
}

//将所有与输入点相连的1利用dfs变成2，并将所有与该岛相邻的0入队points
void dfs(queue<pair<int, int>>& points, vector<vector<int>>& grid, int m, int n,
        int i, int j) {
    if (i < 0 || j < 0 || i == m || j == n || grid[i][j] == 2){
        return;
    }
    if (grid[i][j] == 0) {
        points.push({i, j});
        return;
    }
    grid[i][j] = 2;
    dfs(points, grid, m, n, i - 1, j);
    dfs(points, grid, m, n, i + 1, j);
    dfs(points, grid, m, n, i, j - 1);
    dfs(points, grid, m, n, i, j + 1);
}
```

## 6.5 练习
### 130 Surrounded Regions
所有边缘上的 O 或与这些 O 相邻的 O 不变，其余被 X 包围的 O 全部赋值 X

思路1：深度优先搜索

对边缘上的 O 结点进行深度优先搜索，将边缘 O 及与其相邻的 O 赋值为 N。最后遍历整个矩阵，将所有 N 赋值为 O， 所有 O 赋值 X。
```cpp
int m, n;
void solve(vector<vector<char>>& board) {
    m = board.size();
    if (m == 0)
        return;
    n = board[0].size();
    for (int j = 0; j < n; ++j) {
        dfs(board, 0, j);
        dfs(board, m - 1, j);  
    }
    for (int i = 0; i < m; ++i) {
        dfs(board, i, 0);
        dfs(board, i, n - 1);  
    }
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if(board[i][j] == 'O') {
                board[i][j] = 'X';
            } else if(board[i][j] == 'N') {
                board[i][j] = 'O';
            }
        }
    }
}

void dfs(vector<vector<char>>& board, int i, int j) {
    //访问过的结点要么是X，要么是N
    if (i < 0 || i >= m || j < 0 || j >= n || board[i][j] != 'O') {
        return;
    }
    board[i][j] = 'N';
    dfs(board, i - 1, j);
    dfs(board, i + 1, j);
    dfs(board, i, j - 1);
    dfs(board, i, j + 1);
}
```
思路2：广度优先搜索

类似思路1，只不过采用广度优先搜索
```cpp
vector<int> direction{-1, 0, 1, 0, -1};
void solve(vector<vector<char>>& board) {
    int m = board.size();
    if (m == 0) return;
    int n = board[0].size();
    queue<pair<int, int>> points;
    for (int j = 0; j < n; ++j) {
        if (board[0][j] == 'O') {
            points.push({0, j});
        }
        if (board[m-1][j] == 'O') {
            points.push({m - 1, j});
        }
    }
    for (int i = 0; i < m; ++i) {
        if (board[i][0] == 'O') {
            points.push({i, 0});
        }
        if (board[i][n-1] == 'O') {
            points.push({i, n - 1});
        }
    }
    int x, y;
    while (!points.empty()) {
        int r = points.front().first, c = points.front().second;
        board[r][c] = 'N';
        points.pop();
        for (int k = 0; k < 4; ++k) {
            x = r + direction[k], y = c + direction[k+1];
            if (x >= 0 && y >= 0 && x < m && y < n && board[x][y] == 'O') {
                points.push({x, y});
            }
        }
    }
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if(board[i][j] == 'O') {
                board[i][j] = 'X';
            } else if(board[i][j] == 'N') {
                board[i][j] = 'O';
            }
        }
    }
}
```

### 257 Binary Tree Paths
Given a binary tree, return all root-to-leaf paths.

思路1： 不用回溯法的深度遍历搜索
```cpp
vector<string> binaryTreePaths(TreeNode* root) {
    vector<string> paths;
    string p;
    dfs(root, paths, p);
    return paths;
}
//注意如果不用回溯法，string p不能加引用，否则遍历右子树时会出错
void dfs(TreeNode* root, vector<string>& paths, string p) {
    if (root == nullptr) {
        return;
    }
    p += to_string(root->val);
    if (root->left == nullptr && root->right == nullptr) {
        paths.push_back(p);
        return;
    }
    p += "->";
    dfs(root->left, paths, p);
    //因为p是传值，此时依旧以“->”结尾
    dfs(root->right, paths, p);
}
```

思路2：使用回溯法的深度优先搜索
```cpp
vector<string> binaryTreePaths(TreeNode* root) {
    vector<string> paths;
    string p;
    dfs(root, paths, p);
    return paths;
}
//使用回溯法，则需要引用p
void dfs(TreeNode* root, vector<string>& paths, string& p) {
    if (root == nullptr) {
        return;
    }
    p += to_string(root->val);
    if (root->left == nullptr && root->right == nullptr) {
        paths.push_back(p);
        return;
    }
    //回溯法需要保证遍历右子树与遍历左子树之前的p不变
    string l = p + "->";
    string r = p + "->";
    dfs(root->left, paths, l);
    dfs(root->right, paths, r);
}
```
思路3：广度优先搜索
```cpp
vector<string> binaryTreePaths(TreeNode* root) {
    vector<string> paths;
    if (root == nullptr) {
        return paths;
    }
    //单条path，但不是字符串类型
    queue<TreeNode*> node_queue;
    queue<string> path_queue;

    node_queue.push(root);
    path_queue.push(to_string(root->val));

    while (!node_queue.empty()) {
        TreeNode* node = node_queue.front(); 
        string path = path_queue.front();
        node_queue.pop();
        path_queue.pop();

        if (node->left == nullptr && node->right == nullptr) {
            paths.push_back(path);
        } else {
            if (node->left != nullptr) {
                node_queue.push(node->left);
                path_queue.push(path + "->" + to_string(node->left->val));
            }

            if (node->right != nullptr) {
                node_queue.push(node->right);
                path_queue.push(path + "->" + to_string(node->right->val));
            }
        }
    }
    return paths;
}
```

### 47 Permutations II
Given a collection of numbers, nums, that might contain duplicates, return all possible unique permutations in any order.

思路：建立一个perm向量用来存储单一排列。建立visited向量来表示在当前索引**之前**是否用过该元素。每次遍历nums中所有元素，对于每个可用元素，将其放在当前索引处，再利用回溯法，完成当前排列，再回改，继续将下一个可用元素放于当前索引出。

不可用元素定义（满足任何一条）：
1. 在当前索引之前被用过
2. 在当前索引之前没被用过，但是有等价元素在当前索引处被用了（但是visited仍为false）
```cpp
vector<vector<int>> permuteUnique(vector<int>& nums) {
    //使得重复数字相邻
    sort(nums.begin(), nums.end());
    vector<vector<int>> ans;
    //某个单一排列组合，不能用nums表示是为了避免白排序了
    vector<int> perm;
    vector<bool> visited(nums.size(), false);
    backtracking(nums, 0, ans, visited, perm);
    return ans;
}
void backtracking(vector<int>& nums, int level, vector<vector<int>>& ans,
                    vector<bool>& visited, vector<int>& perm) {
    //level = 0时，在向perm中安插索引为0的元素
    if (level == nums.size()) {
        ans.push_back(perm);
    }
    //此处i从0开始，是因为我们利用perm表示单次排列组合，而不是直接修改nums
    //遍历所有元素，找能在level索引放的没用过的元素，并对每种情况进行回溯
    for (int i = 0; i < nums.size(); ++i) {
        //注意visited只能表示level索引之前是否用过该元素，不能表示level索引处是否用过！！
        //如果当前元素在level索引之前已经被用过了进入下一循环
        //如果当前元素与nums中前若干(>=2)个元素相同，且这些元素在level索引之前都还没用过
        //则在level索引处必然已经用过了等价元素，不能重复，也进入下一循环
        //若若干(>=2)等价元素中已经用过几个了，则没用过的第一个用来填充level索引
        //之后的等价元素则可直接略过
        if (visited[i] || (i > 0 && nums[i] == nums[i-1] && !visited[i-1])) {
            continue;
        }
        perm.push_back(nums[i]);
        visited[i] = true;
        backtracking(nums, level + 1, ans, visited, perm);
        //回改
        visited[i] = false;
        perm.pop_back();
    }
}
```
### 40 Combination Sum II
Given a collection of candidate numbers (candidates) and a target number (target), find all unique combinations in candidates where the candidate numbers sum to target.

Each number in candidates may only be used once in the combination.

思路1：类似第47题，差别在于输出条件是组合的和为target，且组合必须unique。但需要注意的是仅在比当前元素大的元素中找单个组合的下一元素。总体上升序寻找。

```cpp
vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
    sort(candidates.begin(), candidates.end());
    vector<vector<int>> ans;
    vector<int> comb;
    vector<bool> visited(candidates.size(), false);
    backtracking(candidates, comb, ans, target, 0, visited);
    return ans;
}

//此处level表示当前被插入的元素在candidates中索引为level
void backtracking(vector<int>& candidates, vector<int>& comb, vector<vector<int>>& ans,
                    int target, int level, vector<bool>& visited) {
    int sum = accumulate(comb.begin(), comb.end(), 0);
    if (sum > target || level > candidates.size()) {
        return;
    }
    if (sum == target) {
        ans.push_back(comb);
        return;
    }
    for (int i = level; i < candidates.size(); ++i) {
        if (visited[i] || (i > 0 && candidates[i] == candidates[i-1] && !visited[i-1])){
            continue;
        }
        visited[i] = true;
        comb.push_back(candidates[i]);
        //仅搜索当选元素后续元素
        backtracking(candidates, comb, ans, target, i + 1, visited);
        visited[i] = false;
        comb.pop_back();
    }
}
```

思路2：排序后首先寻找 使得target减去该元素后的差值可以由比该元素更大的元素相加所得 的元素（即所有组合中 单个组合最小元素 最大的）。尽可能多次重复该元素，组合剩余元素。再减小单个组合最小元素。
```cpp
//用来存放每个数出现的次数
vector<pair<int, int>> freq;
vector<vector<int>> ans;
vector<int> comb;

vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
    sort(candidates.begin(), candidates.end());
    for (const int num : candidates) {
        if(freq.empty() || num != freq.back().first) {
            freq.emplace_back(num, 1);
        }
        else {
            ++freq.back().second;
        }
    }
    dfs(0, target);
    return ans;
}
//pos表示当前遍历的是freq[pos].first
void dfs(int pos, int rest) {
    if (rest == 0) {
        ans.push_back(comb);
        return;
    }
    //如果遍历完所有unique元素或rest比所有剩余元素都小
    if (pos == freq.size() || rest < freq[pos].first) {
        return;
    }
    dfs(pos + 1, rest);
    //如果rest不能由下一个大数组成，则需要倒序考虑前面重复的小数
    int most = min(rest/freq[pos].first, freq[pos].second);
    for (int i = 1; i <= most; ++i) {
        comb.push_back(freq[pos].first);
        //pos+1表示此处最先满足条件的元素是与target差值仍比后续元素大的
        dfs(pos + 1, rest - i * freq[pos].first);
    }
    for (int i = 1; i <= most; ++i) {
        comb.pop_back();
    }
}
```
### 37 Sudoku Solver （类似N-Queens）

### 310 Minimum Height Trees
Given a tree of n nodes labelled from 0 to n - 1, and an array of n - 1 edges where edges[i] = [ai, bi] indicates that there is an undirected edge between the two nodes ai and bi in the tree, you can choose any node of the tree as the root. When you select a node x as the root, the result tree has height h. Among all possible rooted trees, those with minimum height (i.e. min(h))  are called minimum height trees (MHTs).

Return a list of all MHTs' root labels. You can return the answer in any order.

思路1：不断修减树，每次减去仅与1个其他点相连的点，最后剩下的单个或两个点即为结果。原理类似BFS。
```cpp
vector<int> findMinHeightTrees(int n, vector<vector<int>>& edges) {
    if (n == 1)
        return {0};
    else if (n == 2)
        return {0, 1};
    
    //子节点的数目
    vector<int> indegree(n, 0);
    vector<int> v;
    //expression of the graph
    vector<vector<int>> graph(n, v);
    for (int i = 0; i < edges.size(); i++){
        graph[edges[i][0]].push_back(edges[i][1]);
        graph[edges[i][1]].push_back(edges[i][0]);
        indegree[edges[i][0]]++;
        indegree[edges[i][1]]++;
    }
    queue<int> myqueue;
    //删除所有仅有1个子节点的点，其不可能为最小高度树的根结点
    for (int i = 0; i < n; i++){
        if (indegree[i] == 1){
            myqueue.push(i);
        }
    }
    int cnt = myqueue.size();
    //对于刚刚入队的所有点，将其连接点的indegree-1
    while (n > 2){
        n -= cnt;
        while (cnt--){
            int temp = myqueue.front();
            myqueue.pop();
            indegree[temp] = 0;
            for(int i = 0; i < graph[temp].size(); i++){
                if (indegree[graph[temp][i]] != 0){
                    indegree[graph[temp][i]] --;
                    if (indegree[graph[temp][i]] == 1){
                        myqueue.push(graph[temp][i]);
                    }
                }
            }
        }
        cnt = myqueue.size();
    }
    vector<int> result;
    //最后队内剩下的是两个相连的点或者单个点
    while (!myqueue.empty()){
        result.push_back(myqueue.front());
        myqueue.pop();
    }
    return result;
}
```
