#Remove duplicates
#70%
i = 1
for b in range(len(nums) - 1):
    if nums[b] != nums[b + 1]:
        nums[i] = nums[b + 1]
        i += 1
return (i)

#Buy Time to Sell Stock

#10% faster
prof = []
for i in range(len(prices) - 1):
    prof.append(max(prices[i + 1] - prices[i], 0))
return (sum(prof))

#98% faster
prof = [0] * (len(prices) - 1)
for i in range(len(prices) - 1):
    prof[i] = (max(prices[i + 1] - prices[i], 0))
return (sum(prof)

#from right side
if not prices or len(prices) is 1:
    return 0
profit = 0
for i in range(1, len(prices)):
    if prices[i] > prices[i - 1]:
        profit += prices[i] - prices[i - 1]
    return profit

#Rotate Array
#80%
k = k % len(nums)
nums[:] = nums[-k:] + nums[:-k]

#Contains duplicate
#41%
return(len(set(nums))!=len(nums))

#Single Number
#72%
return(2*(sum(set(nums)))-sum(nums))

#Intersection of Two Arrays
#Dict method 92% big o(n)
d = {}
ans = []
for i in nums2:
    d[i] = d.get(i, 0) + 1
for i in nums1:
    if d.get(i, 0) != 0:
        ans.append(i)
        d[i] -= 1
return (ans)

#two pointers method 74% , sorted makes slower
nums1, nums2 = sorted(nums1), sorted(nums2)
pt1 = pt2 = 0
res = []
while True:
    try:
        if nums1[pt1] > nums2[pt2]:
            pt2 += 1
        elif nums1[pt1] < nums2[pt2]:
            pt1 += 1
        else:
            res.append(nums1[pt1])
            pt1 += 1
            pt2 += 1
    except IndexError:
        break
return res

#Counter method 45%
counts = collections.Counter(nums1)
res = []
for num in nums2:
    if counts[num] > 0:
        res += num,
        counts[num] -= 1
return res

#PlusOne
#book method 60%
digits[-1] += 1

for i in reversed(range(1, len(digits))):
    if digits[i] != 10:
        break
    digits[i] = 0
    digits[i - 1] += 1

if digits[0] == 10:
    digits[0] = 1
    digits.append(0)
return (digits)

#Move Zeroes
#62%
pt = 0
for i in range(len(nums)):
    if nums[i] != 0:
        nums[pt], nums[i] = nums[i], nums[pt]
        pt += 1

#Two Sums
#46%
d={}
for i,n in enumerate(nums):
    des=target-n
    if des in d:
        return([d[des],i])
    else:
        d[n]=i

#Valid Sudoku
def isUnitValid(self, unit):
    check = [x for x in unit if x != '.']
    return (len(set(check)) == len(check))


def isRowValid(self, board):
    for unit in board:
        if self.isUnitValid(unit) == False:
            return (False)
    return (True)


def isColValid(self, board):
    for unit in zip(*board):
        if self.isUnitValid(unit) == False:
            return (False)
    return (True)


def isSquareValid(self, board):
    for i in (0, 3, 6):
        for j in (0, 3, 6):
            unit = [board[x][y] for x in range(i, i + 3) for y in range(j, j + 3)]
            if self.isUnitValid(unit) == False:
                return (False)
    return (True)


def isValidSudoku(self, board):
    return (self.isRowValid(board) and self.isColValid(board) and self.isSquareValid(board))
#Rotate Image
#54% if apply list, 98% if not
#reverse and then transpote
matrix[:]=map(list,zip(*matrix[::-1]))

#Reverse String
#51% returns an iterator ready to traverse the list in reversed order
s[:]=reversed(s)
#93% does it inplace, modifies the list itself and does not return a value
s.reversed()
repeat('aabccddefg')
#15% does not change list but returns reversed slice
s[:]=s[::-1]

#Reverse Integer
#96%
sign = (x > 0) - (x < 0)
nums = sign * int((str(abs(x))[::-1]))
if -(2) ** 31 <= nums < 2 ** 31:
    return (nums)
else:
    return (0)

#First Unique Character in a String
#14% -> 65%
d = {}
seen = [] #change to set() since we knwo each value gon be unique
for i, n in enumerate(s):
    if n not in seen: #have to use seen so if letter shows up 3 times, doesnt add,delete,then add= only want to add each element once or else rmeove from dict if thers duplicates, keep new dict to onlyh ave single showne elements
        d[n] = i
        seen.append(n) #change to set if
    elif n in d:
        del d[n]
if len(d.values()) > 0: #change to just If d: brings from 65% to 87%
    return (min(d.values()))
else:
    return (-1)

#Valid Anagram
# 1 dic 70%
d = {}
for i in s:
    d[i] = d.get(i, 0) + 1

for i in t:
    if i in d:
        d[i] -= 1
    else:
        return (False)
for i in (d.values()):
    if i != 0:
        return (False)

return (True)

#sorted method 40%
return(sorted(s)==sorted(t))

#two dict method
d = {}
for i in s:
    d[i] = d.get(i, 0) + 1

d2 = {}
for i in t:
    d2[i] = d2.get(i, 0) + 1

if d == d2:
    return (True)
else:
    return (False)
#Valid Palindrome
#81% takes Big O n space
s = s.lower()
s = [x for x in s if x.isalnum()]
return (s == s[::-1])

#40% takes big o constant in place
l, r = 0, len(s) - 1
while l < r:
    while l < r and not s[l].isalnum():
        l += 1
    while l < r and not s[r].isalnum():
        r -= 1
    if s[l].lower() != s[r].lower():
        return False
    l += 1;
    r -= 1
return True

# String to Integer
#49%
str = list(str.strip())
if len(str) == 0:
    return (0)

if str[0] == '-':
    sign = -1
else:
    sign = 1

if str[0] in ['-', '+']:
    del str[0]

res, i = 0, 0
while i < len(str) and str[i].isnumeric():
    res = res * 10 + int(str[i])
    i += 1

return (max(-2 ** 31, min(res * sign, 2 ** 31 - 1)))

#Implement strStr()
#80%, time complexity big o n * m tho during hay==needle chunk
if needle == '':
    return (0)
nl = len(needle)
for i in range(len(haystack) - nl + 1):
    if haystack[i:i + nl] == needle:
        return (i)
return (-1)

#Count and Say
def cns(str_):
    res = ''
    str_ += '#'
    c = 1
    for i in range(len(str_) - 1):
        if str_[i] == str_[i + 1]:
            c += 1
            continue
        else:
            res += str(c) + str_[i]
            c = 1

    return res


start = '1'
for i in range(n - 1):
    start = cns(start)
return start

#Longest Common Prefix
#73%
if not strs:
    return ("")
shortest = min(strs, key=len)
for i, l in enumerate(shortest):
    for others in strs:
        if others[i] != l:
            return (shortest[:i])
return (shortest)

#delete node in linked list
#36ms, 13mb ,97%
node.val = node.next.val
node.next = node.next.next

#remove n-th node from linked list
#20ms,12.7mb, 98%
fast = head
slow = head

for i in range(n):
    fast = fast.next

if fast == None:#this means that if fast already = None, it means the head is what we want to remove, otherwise we are trying to land on the node before the one we want to remove
    return (head.next)

while fast.next != None:
    fast = fast.next
    slow = slow.next

slow.next = slow.next.next
return (head)

#reverse node in linked list
#32 ms, 13.8mb , 97%
prev = None
nextt = None
curr = head

while (curr != None):
    nextt = curr.next #need to save next node beacuse about to overwrite the curr.next to point to previous node(reversing direction)
    curr.next = prev
    prev = curr
    curr = nextt

return (prev)

#merge two sorted lists
#pointer method 36ms, 12.8mb, 94%
dummy = cur = ListNode(0)
while l1 and l2:
    if l1.val < l2.val:
        cur.next = l1
        l1 = l1.next
    else:
        cur.next = l2
        l2 = l2.next
    cur = cur.next
cur.next = l1 or l2
return (dummy.next)

#palindrome linked list
#56 ms, 22.8mb 99%
fast = slow = head
while fast and fast.next:#if # of nodes is even, then slow will land right at the node right after the midpoint since fast is traveling twice as fast, then you reverse the following linked list and compare it to the head list
    #goal is to get fast to the end (null) using 2x speed so just need to check fast and fast.next before advancin two nodes forward
    fast = fast.next.next # wanna go twice as fast as slow
    slow = slow.next

nd = None
while slow:#reverse the linked list at midway pt
    nxt = slow.next
    slow.next = nd
    nd = slow
    slow = nxt

while nd: # compare the reverse linked list and head
    if nd.val != head.val:
        return (False)
    nd = nd.next
    head = head.next
return (True)

#Linked list cycle
#48ms ,16.1 mb, 94%
if not head:
    return (False)

slow = head
fast = head.next # start off at head.next cus we already know head is not null and dont want to trigger comparison equals

while slow != fast: #this part check if there is a cycle
    if fast == None or fast.next == None: #this part check if there is no cycle
        return (False)
    fast = fast.next.next #if we cant prove either then, advance both forward until we prove one
    slow = slow.next
return (True)

#maximum depth of binary tree
#40ms ,13.8mb, 96%
#BFS , breadth first search
level = [root] if root else []
depth = 0

while len(level)>0:
    depth += 1
    queue = []
    for i in level:
        if i.left:
            queue.append(i.left)
        if i.right:
            queue.append(i.right)
    level = queue
return (depth)

#Depth first search, dfs
#40ms, 13.9mb,96%
depth = 0
stack = [(root, 1)] if root else []

while len(stack)>0:
    root, leng = stack.pop()

    if leng > depth:
        depth = leng

    if root.right:
        stack.append((root.right, leng + 1))
    if root.left:
        stack.append((root.left, leng + 1))

return (depth)

#Validate Binary Search Tree #just checking that each left element is less than its root val and right is greater
#alt way is to store inorder traversal of bt in a temp array and check if array is sorted in increasin order, cons r memory space and have to traverse an entire new array on top of traversin bt which we do for either methods, adding O(n) time

#44ms, 15 mb ,94%
def isValidBST(self, root: TreeNode, floor=float('-inf'), ceiling=float('inf')) -> bool:
    if not root:
        return True
    if root.val >= ceiling or root.val <= floor : # right clause is for right subtree meaning values should never be lower than the root, left clause is for left subtree meaning values shud never be higher than the root. floor and ceiling in this case is the root node above the one bein compared
        return False
    # in the left branch, root is the new ceiling; contrarily root is the new floor in right branch
    return self.isValidBST(root.left, floor, root.val) and self.isValidBST(root.right, root.val, ceiling)

#Symmetric Tree
#40 ms, 12.8mb,71%
if not root:
    return (True)

stack = [(root.left, root.right)]

while len(stack) > 0:
    left, right = stack.pop()
    if left is None and right is None:
        continue
    if left is None or right is None:
        return (False)

    if left.val == right.val:
        stack.append((left.left, right.right))
        stack.append((left.right, right.left))
    else:
        return (False)
return (True)

#Binary Tree Level Order Traversal
#28 ms, 13 mb,99%
if not root:
    return ([])

ans = []
level = [root]

while len(level) > 0:
    ans.append([node.val for node in level])
    temp = []
    for node in level:
        temp.extend([node.left, node.right])
    level = [leaf for leaf in temp if leaf]

return (ans)

# or

if not root:
    return([])
level=[root]
ans=[]
while len(level)>0:
    res=[x.val for x in level]
    ans.append(res)
    queue=[]
    for i in level:
        if i.left:
            queue.append(i.left)
        if i.right:
            queue.append(i.right)
    level=queue
return(ans)
#sorted array to BST
#64 ms , 14.9mb, 98%
if not nums:
    return (None)

mid = (len(nums) - 1) // 2

root = TreeNode(nums[mid])
root.left = self.sortedArrayToBST(nums[:mid])
root.right = self.sortedArrayToBST(nums[mid + 1:])

return (root)

#merge sorted array
#36 ms ,12.6mb, 93%
#3 pointers
while n > 0:
    if m == 0 or nums1[m - 1] < nums2[n - 1]:
        nums1[m + n - 1] = nums2[n - 1]
        n -= 1
    else:
        nums1[m + n - 1] = nums1[m - 1]


#first bad version
#binary search method
#28ms, 12.6mb, 90
        i = 1
        j = n

        while i < j:
            mid = i + (j - i) // 2

            if isBadVersion(mid) == True:
                j = mid
            else:
                i = mid + 1

        return (i)