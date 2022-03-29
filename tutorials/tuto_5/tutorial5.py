'''
Dazhou Hou(Howard)-1130052
Jiayi Zhang(Libre)-1129829
'''

import re

# Question A
string1 = 'abb29ABCLK%lCnrDBCabbbb'
substring1 = re.search(r'abbbb', string1)    # match exact substring 'abbb'
substring2 = re.search(r'ab{4}', string1)    # match the exact assigned number of preceding RE
substring3 = re.search(r'abbb+', string1)    # Causes the resulting RE to match 1 or more repetitions of the preceding RE
substring4 = re.search(r'abbbb*', string1)    # Causes the resulting RE to match 0 or more repetitions of the preceding RE

print('---------------Qestion A--------------')
print('substring1', substring1.group())
print('substring2', substring2.group())
print('substring3', substring3.group())
print('substring4', substring4.group())


# Question B
# [A-Z]: specifies any capital letter
# [a-z]: specifies any lowercase letter
# +: match 1 or more repetitions of the preceding RE
# Thus, it will find the first substring that begins with an uppercase letter and ends with multiple lowercase letters.
substring5 = re.search(r'[A-Z][a-z]+', string1)
print('---------------Qestion B--------------')
print('substring5', substring5.group())
