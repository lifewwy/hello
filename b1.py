import sys
class Solution(object):
    def threeSumClosest(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        # print(len(nums))
        if len(nums) in [0,1,2]: # 成员运算符符 in 和 not in # 内置函数 len()
            return 0
        else:
            min_diff =  sys.maxsize
            result =  0
            sorted_nums =  sorted(nums)
            for i in range(len(nums)):
                print(i)
                start =  i +  1
                end =  len(nums) - 1
                while start <  end:
                    curr_sum =  sorted_nums[i] +  sorted_nums[start] +  sorted_nums[end]
                    diff =  abs(curr_sum - target)
                    if diff ==  0:
                        return curr_sum
                    if diff <  min_diff:
                        min_diff =  diff
                        result =  curr_sum
                    if curr_sum <=  target:
                        start +=  1
                    else:
                        end -= 1
            return result

if  __name__    ==  "__main__":
    soln =  Solution()
    print(soln.threeSumClosest([-1, 2, 1, -4], 1))
    print(soln.threeSumClosest([-1, 2, 1, -4], 3))









