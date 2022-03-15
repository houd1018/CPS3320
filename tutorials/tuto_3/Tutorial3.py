'''
Dazhou Hou(Howard)-1130052
Jiayi Zhang(Libre)-1129829
'''
class Person(object):
    # Create a constructor function to initiate the variables of any person.
    def __init__(self):
        self.name = None
        self.age = None
        self.weight = None
        self.height = None

    # allows the user to input user attributes (variables)
    def input_person_data(self):
        self.name = input('enter your name:')
        self.age = input('enter your age:')
        self.weight = input('enter your weight:')
        self.height = input('enter your height:')

    # print a person name, age, weight.
    def get_person_data(self):
        print('Name:%s Age:%s Weight:%s Height:%s' % (self.name, self.age, self.weight, self.height))

# Create a main() function in which you create two instances(objects), then use the above methods to input and print the objects attributes
def main():
    # first person
    Person_1 = Person()
    Person_1.input_person_data()
    Person_1.get_person_data()
    # second person
    Person_2 = Person()
    Person_2.input_person_data()
    Person_2.get_person_data()

if __name__ == "__main__":
    main()