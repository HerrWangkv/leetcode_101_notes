## 5.1 常用排序算法
(参考 《数据结构与算法分析 C++描述》 #7 排序)
### 快速排序(平均情况下O(n logn)，最坏情形O(n^2))
快速排序是一种分治的递归算法：
1. 如果数组 S 中元素个数为 0 或 1 ，则返回本身
2. 取 S 中任意元素 v，称为 pivot
3. 对于 S 中除 pivot 外元素划分成 小于等于 pivot 的集合S1 和 大于 pivot 的集合S2
4. 把 v 置于中间，再对S1 和 S2 分别快速排序。
   
```cpp
void quickSort(vector<int>& nums, int l, int r) {
    if (l + 1 >= r) {
        return;
    }
    //找到一个比较合适的pivot,设为第一个元素
    int first = l, last = r - 1, key = nums[first];
    //双向遍历
    while (first < last) {
        //从后往前找第一个比key小的元素
        while (first < last && nums[last] >= key) {
            --last;
        }
        //把这个本该属于S1集合(<key)的值移到第一位，原来位于第一位的key因为要放到 S1 和 S2中间，直接覆盖掉
        nums[first] = nums[last];
        //从前往后找第一个比key大的元素
        while (first < last && nums[first] <= key) {
            ++first;
        }
        //因为last处的值已经移到了第一位，先把本属于S2集合(>key)的元素移到last处
        nums[last] = nums[first];
    }
    //此时first=last，位于S1和S2之间
    nums[first] = key;
    quickSort(nums, l, first);
    quickSort(nums, first + 1, r);
}

//调用时l = 0, r = nums.size()
quickSort(nums, 0, nums.size());
```

### 归并排序(O(n logn))
同样是分治策略。对两个已排好序的表，将较小的首元素移进空白总表。

```cpp
void mergeSort(vector<int>&nums, int l, int r, vector<int> &temp) {
    if (l + 1 >= r) {
        return;
    }
    //divide
    int m = l + (r - l) / 2;
    mergeSort(nums, l, m, temp);
    mergeSort(nums, m, r, temp);

    //conquer
    int p = l, q = m, i = l;
    //最开始进入的是两个由单个元素组成的集合，肯定是排好序的。
    //对两个已排好序的表取小的放在前面
    while (p < m || q < r) {
        
        if (q >= r || (p < m && nums[p] <= nums[q])) {
            temp[i++] = nums[p++];
        }
        else {
            temp[i++] = nums[q++];
        }
    }
    //对排好序的部分进行赋值
    for (i = l; i < r; ++i) {
        nums[i] = temp[i];
    }
}

//调用时l = 0, r = nums.size()
mergeSort(nums, 0, nums.size());
```

### 插入排序
将元素依次插入到排好序的数组中的相应位置中

### 冒泡排序
比较相邻元素，小的置前，大的置后，每次将遍历到的最大元素置于最后

### 选择排序
每次循环中选择最小元素，并与当前上为排好序的首位元素换位

## 5.2 快速选择
```cpp
//找出nums[l]在排好序的数组中的索引
//并保证该返回值前均比nums[l]小，该返回值后均比nums[l]大
int quickSelection (vector<int>& nums, int l, int r) {
    //l r 都是索引
    int i = l, j = r, pivot = nums[l];
    while (i < j) {
        //找到最后一个小于nums[l]的元素索引
        while (j > i && nums[j] >= pivot) {
            --j;
        }
        nums[i] = nums[j];
        //找到第一个大于nums[l]的元素索引
        while (i < j && nums[i] <= pivot) {
            ++i;
        }
        nums[j] = nums[i];
    }
    //将nums[l]置于中间
    nums[i] = pivot;
    return i;
}
int findKthLargest(vector<int>& nums, int k) {
    //随机打乱以避免最坏情况
    random_shuffle(nums.begin(), nums.end());
    int l = 0, r = nums.size() - 1, target = nums.size() - k;
    while (l < r) {
        int mid = quickSelection(nums, l, r);
        if (mid == target) {
            return nums[mid];
        }
        //通过比较排好序数组中的索引判断需要增大还是减小目标
        if (mid < target) {
            l = mid + 1;
        } else{
            r = mid - 1;
        }
    }
    return nums[l];
}
```

## 5.4 练习
### 451 Sort Characters By Frequency
Given a string, sort it in decreasing order based on the frequency of characters.

```cpp
string frequencySort(string s) {
    unordered_map<char, int> counts;
    int max_count = 0;
    for(const char& c : s) {
        max_count = max(max_count, ++counts[c]);
    }
    
    vector<string> buckets(max_count + 1);
    for (const auto& p : counts) {
        buckets[p.second].push_back(p.first);
    }
    
    string ret;
    for (int i = max_count; i >= 0; --i){
        for (const char& ch : buckets[i]) {
            ret.append(i, ch);
        }
    }
    return ret;
}
```

### 75 Sort Colors
Given an array nums with n objects colored red, white, or blue, sort them in-place so that objects of the same color are adjacent, with the colors in the order red, white, and blue.

Here, we will use the integers 0, 1, and 2 to represent the color red, white, and blue respectively.
```cpp
void sortColors(vector<int>& nums) {
    int n = nums.size();
    if (n < 2)
        return;
    int p = 0, q = n - 1;
    //遇见0，置于p处；遇见2，置于q处。保证p之前全为0， q之后全为2
    for (int i = 0; i <= q; ++i) {
        if (nums[i] == 0) {
            swap(nums[i], nums[p]);
            ++p;
            //不用--i是因为换回来的nums[p]已遍历过，不会是2
        }
        if (nums[i] == 2) {
            swap(nums[i], nums[q]);
            --q;
            //防止换回来的不是1
            --i;
        }
    }
}
```
