'''
Dazhou Hou(Howard)
Jiayi Zhang(Libre)
'''
from re import A


def main():
    # get initial input
    num = input("enter a positive integer:")

    # use while loop to get new input value until the new input is integer 
    while not num.isdigit():
        # prompt user again to input a new number
        num = input("Wrong type! Please enter a positive integer:")

    # use while loop to get new input value until the new input is positive
    while int(num) <= 0 :
        # prompt user again to input a new number
        num = input("Wrong type! Please enter a positive integer:")

    # string cast to int
    num = int(num)

    # check whether the number is a multiple of 2,3,4 collectively
    if num % 2 == 0 and num % 3 == 0 and num % 4 == 0:
        print("The number %d is multiple of 2,3,4" %num)
    else:
        print("The number %d is not a multiple of 2,3,4" %num)

    # program ends
    print("End of Program")

if __name__ == '__main__':
    main()
