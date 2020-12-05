## 3.1 C++ 花样
```cpp
//第一个const：int恒定，即指向的值不可改变
//第二个const：指针恒定，即指向的地址不可改变
const int * const p = &x;

//函数中新建空间，new返回新空间地址，可用于初始化指向链表结点的指针
类型* sum = new 类型(值);

//函数指针，指向函数的指针
返回类型 (*函数指针名)(参数1类型,...) = 函数名;
//函数指针在函数参数表中表示为：
返回类型 (*函数指针名)(参数1类型,...)
//调用时
(*函数指针名)(参数1,...);
```

## 3.3 归并两个有序数组
### 88 Merge Sorted Array
```cpp
void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
    //运行完之后m, n分别为nums1和nums2最后一位有数元素的索引
    int pos = m-- + n-- - 1;
    //直接在nums1从后往前添加两个指向元素中较大的
    //若最后m=0, n!=0表示nums2剩余元素最小，可直接赋值给nums1的前几个元素
    //若最后m!=0, n=0表示nums1剩余元素最小，则不用变了
    while (m >= 0 && n >= 0) {
        //条件运算符
        nums1[pos--] = nums1[m] > nums2[n] ? nums1[m--] : nums2[n--];
    }
    while (n >= 0) {
        nums1[pos--] = nums2[n--];
    }
}
```
注： 在进行累加时，如果不需要返回值，++a运行速度较快。

## 3.4 快慢指针
### 142 Linked List Cycle II
思路详解：假设从1开始一直到n共n个结点，n->next指向m。则fast进入循环后结点数与时间的函数：(2t - m) % (n - m) + m; slow进入循环后函数为：(t - m) % (n - m) + m。当fast与slow第一次重合时，t = n - m, slow位于(-m) % (n - m) + m。

此时将fast重新置于head，则fast进入循环时间为 t = m; slow进入循环后函数为：(t) % (n - m) + (-m) % (n - m) + m = (t - m) % (n - m) + m, t = m时，位于m，即n->next

C++细节：do while 和 while 区别： do while至少进入循环一次，而while可完全不进入循环。

## 3.5 滑动窗口
### 76 Minimum Window Substring
```cpp
string minWindow(string s, string t) {
    //表示还需要几个、哪些字符
    vector<int> chars(52, 0);
    //表示target中是否有这个字符
    vector<bool> flag(52, false);
    int index;
    //统计target字符串
    for (int i = 0; i < t.size(); ++i) {
        index = t[i] >= 'a' ? 26 + t[i] - 'a' : t[i] - 'A';
        flag[index] = true;
        ++chars[index];
    }
    int cnt = 0, l = 0, minL = 0, minSize = s.size() + 1, indexL, indexR;
    for (int r = 0; r < s.size(); ++r) {
        indexR = s[r] >= 'a' ? 26 + s[r] - 'a' : s[r] - 'A';
        if (flag[indexR]) {
            if (--chars[indexR] >= 0) {
                //cnt加1表示t中某一字符已满足
                ++cnt;
            }
        }
        //当从l到r这段字符包含所有所需字符时，移动l找最小
        while (cnt == t.size()) {
            if (r - l + 1 < minSize) {
                minL = l;
                minSize = r - l + 1;
            }
            indexL = s[l] >= 'a' ? 26 + s[l] - 'a' : s[l] - 'A';
            if (flag[indexL] && ++chars[indexL] > 0)
                --cnt;
            ++l;
        }
    }
    return minSize > s.size() ?  "" : s.substr(minL,minSize);
} 
```
注意取子字符串操作：
```cpp
str.substr(左索引, 区间大小)
```

## 3.6 练习
### 633 Sum of Square Numbers
Given a non-negative integer c, decide whether there're two integers a and b such that a2 + b2 = c.
```cpp
//利用二分查找到窗口右端，即平方小于c的最大整数
long binarySearch (long c) {
    long left = 0, right = c, mid = (left + right) / 2;
    while (left <= right) {
        if (mid * mid < c) {
            left = mid + 1;
            mid = (left + right) / 2;
        }
        else if (mid * mid > c) {
            right = mid - 1;
            mid = (left + right) / 2;
        }
        else
            return mid;
    }
    return mid;
} 
//类似 Two Sum 
bool judgeSquareSum(int c) {
    long l = 0, r = binarySearch(static_cast<long>(c));
    long sum;
    while (l <= r) {
        sum = l * l + r * r;
        if (sum == c)
            return true;
        else if (sum <= c)
            ++l;
        else
            --r;
    }
    return false;
}
```

### 680 Valid Palindrome II
Given a non-empty string s, you may delete at most one character. Judge whether you can make it a palindrome(回文).
```cpp
bool checkPanlindrome (const string &s, int l, int r) {
    while (l < r) {
        if (s[l] != s[r])
            return false;
        ++l;
        --r;
    }
    return true;
}

bool validPalindrome(string s) {
    int n = s.size(), l = 0, r = n - 1;
    if (n < 3)
        return true;
    while (l < r) {
        if (s[l] == s[r]) {
            ++l;
            --r;
        }
        else {
            return (checkPanlindrome(s, l + 1, r) || checkPanlindrome(s, l, r - 1));
        }
    }  
    return true;     
}
```

### 524 Longest Word in Dictionary through Deleting
Given a string and a string dictionary, find the longest string in the dictionary that can be formed by deleting some characters of the given string. If there are more than one possible results, return the longest word with the smallest lexicographical(字典中的) order. If there is no possible result, return the empty string.

思路：首先将string dictionary按照字符串长度为主关键字，字母顺序为第二关键词排序，然后依次用双指针法判断是否可以组成
```cpp
bool isSubstr(const string& s, const string& temp) {
    int i = 0, j = 0;
    while(i < s.size() && j < temp.size()) {
        if (s[i] == temp[j])
            ++j;
        ++i;
    }
    return (j == temp.size());
}
string findLongestWord(string s, vector<string>& d) {
    //禁止C++输入输出流兼容C的输入输出
    ios::sync_with_stdio(false);
    sort(d.begin(), d.end(), [](string& a, string& b) {
        return (a.size() > b.size() || (a.size() == b.size() && a < b));
    });
    for (int i = 0; i < d.size(); ++i) {
        if (isSubstr(s, d[i]))
            return d[i];
    }
    return "";
}
```
