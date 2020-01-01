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
#set takes O(n) and also takes O(n) space since its creating new array
return(len(set(nums))!=len(nums))

#Also O(n)  and O(n) space but doesnt use built in cheap functions
        d={}
        for i in nums:
            d[i]=d.get(i,0)+1
        
        for i in d.values():
            if i>1:
                return(True)
        return(False)

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

#alt way of 2 pters
        nums1.sort()
        nums2.sort()

        index_i, index_j = 0, 0
        result = []
        while index_i < len(nums1) and index_j < len(nums2):
        	if nums1[index_i] == nums2[index_j]:
        		result.append(nums1[index_i])
        		index_i += 1
        		index_j += 1
        	elif nums1[index_i] > nums2[index_j]:
        		index_j += 1
        	else:
        		index_i += 1
        return result
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
#99%
#linear time since using d.get is constantt tim
        d={}
        for i,n in enumerate(nums):
            sol=target-n
            if d.get(sol,-1)!=-1: 
                return([i,d[sol]])
            
            d[n]=i
#alternative way, bit slower
        dic = {}
        for i, num in enumerate(nums):
            if num in dic:
                return [dic[num], i]
            else:
                dic[target - num] = i
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


#similar ans but doesnt use built in fct
#O(n**2) which if fastest time 0(1) since in place
        n = len(matrix[0])        
        # transpose matrix
        for i in range(n):
            for j in range(i, n):
                matrix[j][i], matrix[i][j] = matrix[i][j], matrix[j][i] 
        
        # reverse each row
        for i in range(n):
            matrix[i].reverse()

#Reverse String
#51% returns an iterator ready to traverse the list in reversed order
s[:]=reversed(s)
#93% does it inplace, modifies the list itself and does not return a value , only workson a list
s.reversed()
#15% does not change list but returns reversed slice
s[:]=s[::-1]
#reverse real way 2 pts 85%
#constant space, linear time
        left=0
        right=len(s)-1
        
        while left<right:
            s[left],s[right]=s[right],s[left]
            left+=1
            right-=1
        

#Reverse Integer
#96%
#linear time because of the reverse 
#constant space
sign = (x > 0) - (x < 0) #alt way of findiing sign  sign = [1,-1][x < 0] cus [1,-1] is a list and use 0 or 1 to index
nums = sign * int((str(abs(x))[::-1]))
if -(2) ** 31 <= nums < 2 ** 31:
    return (nums)
else:
    return (0)

#bit cleaner
        sign=(x>0)-(x<0)
        ans=int(str(abs(x))[::-1]) #have to use [::-1] because its a string not a list,and we dont want an iterator
        ans=sign*(ans)
        return(ans if -2**31<ans<2**31-1 else 0)

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

#this way makes much faster by making seen a hash map for constant time search
        d={}
        seen={}
        for ind,num in enumerate(s):
            if num not in seen:
                d[num]=ind
                seen[num]=ind
            elif num in d:
                del d[num]
        
        return(-1 if len(d)==0 else min(d.values()))

#or this way faster time O(N) space O(N) since gotta make two N passses and save memory for hash map
90%
        count = collections.Counter(s)
        for i,n in enumerate(s):
            if count[n]==1:
                return(i)
        
        return(-1)

#Valid Anagram
# 1 dic 70%, this one better than two dict becaues two dict takes 3n time and 2n space, while 1 dict takes 3nt time and 1n space
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

if d == d2: #comparting two dicts is a recursive lookup, takes O(n) fro comparison
    return (True)
else:
    return (False)
#Valid Palindrome
#81% takes O(n) time  O(N) space because create temp array to hold the reverse slice array
s = s.lower()
s = [x for x in s if x.isalnum()]
return (s == s[::-1])

#takes 0(N) time and 0(1) space since just doing lookups/comparisons and moving pointers
        l=0
        r=len(s)-1
        
        while l<r: # has to go thru this check everytime after an event is triggered
            if not s[l].isalnum():
                l+=1
            elif not s[r].isalnum():
                r-=1
            else:
                if s[l].lower()==s[r].lower():
                    l+=1    
                    r-=1
                else:
                    return(False)
        return(True)

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
#73% linear time and constant space
if not strs:
    return ("")
shortest = min(strs, key=len)
for i, l in enumerate(shortest):
    for others in strs: #O(N here)
        if others[i] != l:
            return (shortest[:i]) # O(1) here since input doesnt affect lenght of shortest word
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
#32 ms, 13.8mb , 97% O(L) time O(1) space, linear time cus gotta make a pass thru all # of nodes
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
#O(n+m) so linear time and O(1) since only uisng a few pointers
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
#this takes O(N) time and O(1) since just using pointers
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

#alternative way 50%
#copies the linkedlist into an array then uses two pointers to find reverse and compare
#O(n) time for first pass to make the array and then later on for making reverse O(n) space cus making new array to hold linked list vals and also for reverse later on
        ans=[]
        cur=head
        while cur!=None:
            ans.append(cur.val)
            cur=cur.next
        
        l=0
        r=len(ans)-1
        og=ans.copy()
        while l<r:
            ans[l],ans[r]=ans[r],ans[l]
            l+=1
            r-=1
        return(og==ans)
#Linked list cycle
#48ms ,16.1 mb, 94%
#O(N) time cus it does one pass and O(1) cus we only use two pointers
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

#50% using hash tables 
# O(n) time  does one pass ,and O(n) space for creating the hash
        if not head:
            return(False)
        
        d={}
        while head!=None:
            if d.get(id(head),0)!=0:
                return(True)
            else:
                d[id(head)]=1
            head=head.next
        return(False)

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
#find midpt, then split into two subtrees, repeat the same for both subtrees
if not nums:
    return (None)

mid = len(nums) // 2 #takes the floor so 5//2= 2

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
        m-=1

#first bad version
#binary search method
#28ms, 12.6mb, 90
#https://www.youtube.com/watch?v=SNDE-C86n88
        left = 1
        right = n

        while i < j:
            mid = left + (right - left) // 2

            if isBadVersion(mid) == True:
                right = mid
            else:
                left = mid + 1

        return (left)

#brute force way 
#overflow error    
        res=None
        for i in range(1,n+1):
            if isBadVersion(i)==True:
                res=i
                break
        
        return(res)

#Climbing stairs
#78% bottom up , instead of waiting to make the computation calls when you need em, just compute the entire dp array right away, linear space
        if n==1: #have to include this initial check or else res[1]=2 will error since res only has one element ex:[0]
            return(1)
        res=[0 for x in range(n)]   
        
        res[0]=1
        res[1]=2
        for i in range(2,n):
            res[i]=res[i-1]+res[i-2]
        return(res[n-1])

#91%  Top down + memorization (list)  top down means ur making the computation calls as you need them / recursion/ might error in jupyter because of limit on amt of  recurcision calls
        if n not in self.dic:
            self.dic[n] = self.climbStairs(n-1) + self.climbStairs(n-2)
        return self.dic[n]

    def __init__(self): # creates attributes within the instance of the class to call back later, can call it from other functions by referring it to by self within the funct
        self.dic = {1:1, 2:2}

#Best time to buy n sell
#97%, linear time, #basically keep global maximum going, only update if i-(i-1) turns a profit

        max_profit, min_price = 0, float('inf')
        
        for price in prices: #perform check to see if we shud use price as min price or to calc max profit or none
            min_price = min(min_price, price)#iterate thru and find the lowest price
            profit = price - min_price #keep trackin of profit but we only gon keep when the best one
            max_profit = max(max_profit, profit) #keep track of which day will give us the highest profit with respective to the lowest price we been tracking
        
        return max_profit

#Maximum Subarray
#50% linear time and constant space 
#Kadane's algorithm
        # dp=[0]*len(nums)
        # dp[0]=nums[0]
        
        for i in range(1,len(nums)):
            nums[i]=max(nums[i],nums[i-1]+nums[i])
        return(max(nums))

#naive way would be bruteforce take the sum of all possible sub array 
#that would take n^2 time, n^3 if you compute each subarray from the start 

#House Robber
#87% bottom up
#linear time, linear space

        if not nums:
            return(0)
        if len(nums)==1:
            return(nums[0])
        dp=[0]*len(nums)
        dp[0]=nums[0]
        dp[1]=max(nums[0],nums[1])
        for i in range(2,len(dp)):
            dp[i]=max(nums[i]+dp[i-2],dp[i-1])
        return(dp[len(dp)-1])

#96 bottom up 
#linear time, constant space

        if not nums:
            return(0)
        if len(nums)==1:
            return(nums[0])
    
        nums[1]=max(nums[0],nums[1])
        for i in range(2,len(nums)):
            nums[i]=max(nums[i]+nums[i-2],nums[i-1])
        return(nums[len(nums)-1])

#shuffle an array
#88%
    def __init__(self, nums: List[int]):
        self.arr=nums

    def reset(self) -> List[int]:
        """
        Resets the array to its original configuration and return it.
        """
        return(self.arr)

    def shuffle(self) -> List[int]:
        """
        Returns a random shuffling of the array.
        """
        new=random.sample(self.arr,len(self.arr))
        return(new)

#min stack
#17% linear time bad
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.arr=[]

    def push(self, x: int) -> None:
        self.arr.append(x)

    def pop(self) -> None:
        del(self.arr[-1])

    def top(self) -> int:
        return(self.arr[-1])

    def getMin(self) -> int:
        return(min(self.arr))

#75% constant time
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.arr=[]

    def push(self, x: int) -> None:
        currentmin=self.getMin()
        if currentmin==None or x<currentmin:
            currentmin=x
        self.arr.append((x,currentmin))

    def pop(self) -> None:
        self.arr.pop()

    def top(self) -> int:
        if not self.arr:
            return(None)
        
        return(self.arr[-1][0])

    def getMin(self) -> int:
        if not self.arr:
            return(None)
        
        return(self.arr[-1][1])

    #FizzBuzz
    #96% verbose
            ans=[]
        for i in range(1,n+1):
            if (i)%3 ==0 and (i)%5==0:
                ans.append('FizzBuzz')
                
            elif (i)%3==0:
                ans.append('Fizz')
            
            elif (i)%5==0:
                ans.append('Buzz')
            
            else:
                ans.append(str(i))
        return(ans)

    # 96% cleaner
    return ['Fizz' * (not i % 3) + 'Buzz' * (not i % 5) or str(i) for i in range(1, n+1)]# i % 5 == 0 and not 0 is 1
#choose whichever is present, if both are present then choose value on left
    #count primes
    #88%
        if n<3:
        return(0)
    
    ans=[True]*n # keep it at n even tho  Q wants less than n
    ans[0]=False # becus we throwaway this 0 element
    ans[1]=False #and the index will have meaning equal the element which is good cus the index will go up to right before n 
    for i in range(2,int(n/2)+1): # just need to iterate up to the last # whose squared value is equal to or greater than n(ex:lowest bound is 2*2=4)
        if ans[i]:
            ans[i*i:n:i]=[False]*len(ans[i*i:n:i]) 
            
    return(sum(ans))

    #Power of 3
    #78%
    if n<=0:
    return(False)
    
    while n%3==0:
        n=n/3
        
    return(n==1)
    #60% not faster but no loop or recursion
    return n > 0 and 1162261467 % n == 0

    #Roman to Int
    #55%
    dic={'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
    amt=0
    for i in range(len(s)-1):
        if dic[s[i]]<dic[s[i+1]]:
            amt-=dic[s[i]]
        else:
            amt+=dic[s[i]]
    
    return(amt+dic[s[-1]])

#Number of "1" Bits
#98% uses built in fct
return(bin(n).count('1'))

#71%
        c=0
        while n:
            n= n & n-1
            c+=1
            
        return(c)

#37% 
#recurcsion but not efficient

return 0 if n == 0 else 1 + self.hammingWeight(n&(n-1))

#Hamming Distance
#90% uses biult it bin funct

return(bin(x^y).count('1'))

#74%
# #1010
# XOR
# 1001
# =
# 0011
# x & x-1 is to remove the last bit
# x ^ y is XOR so it basically gives to you as a result the different bits.
        x = x ^ y
        y = 0
        while x:
            y += 1
            x = x & (x - 1)
        return y

#reverse bits

# x << y
# Returns x with the bits shifted to the left by y places 
#8%
        res = 0
        for _ in range(32):
            res = (res<<1) + (n&1)
            n>>=1
        return res

#97%
        res = 0
        for _ in range(32):
            res = res << 1 | (n&1)# changed this part
            n>>=1
        return res

#pascals triangle
# one way 42%
        res = [[1]]
        for i in range(1, numRows):
            res += [list(map(lambda x, y: x+y, res[-1] + [0], [0] + res[-1]))]
        return res[:numRows]
#more intuitive 82%
#adds the row to res array before modifying it
        lists = []
        for i in range(numRows):
            lists.append([1]*(i+1)) #i+1 is the # of elements for that row #
            if i>1 :
                for j in range(1,i):# so i is the index of last element and we want to exclude it since its gonna be always 1
                    lists[i][j]=lists[i-1][j-1]+lists[i-1][j]
        return(lists)
#method of modifying row first before adding to res array
#same time space as above
        res=[]
        for i in range(numRows):
            row=[1]*(i+1)
            if i>1:
                for j in range(1,i):
                    row[j]=res[i-1][j-1]+res[i-1][j]
            res.append(row)
        return(res)

#Valid parenthesis
#83% using stacks
        stack=[]
        dic={')':'(',']':'[','}':'{'}
        
        for i in s:
            if i in dic.values():# have to set closing  par as keys cus we comparing ( to ) , its not commutative where () is == )( since its iterating in order of s
                stack.append(i)
            elif i in dic.keys():
                if stack==[] or dic[i]!=stack.pop():
                    return(False)
            else:
                return(False)
        return(stack==[])
    
#missing number
#96% using constant space and constant time
#proofs here that the n-th Triangular number, 1+2+3+...+n is n(n+1)/2
        n = len(nums)
        return int(n * (n+1) / 2 - sum(nums))

#83% using linear time and linear space
        tot=0
        for i in range(len(nums)+1):
             # full array shud be 0-9 which is 10 values  but they only give us 9 out of those 10 values, so compute the full by taking range of len(n)+1 
            tot+=i
        return(tot-sum(nums))