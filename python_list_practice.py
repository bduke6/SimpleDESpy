import random

class minievent():
    def __init__(self, time, created):
        self.time = time
        self.color = "blue"
        self.size = "big"
        self.created = created

'''Some test function to figure out how sorts work on python lists.'''
def queue_test():
    l = []
    for i in range(20):
        t = random.randint(0,1000)
        l.append(minievent(t, i))


    print("Original List:")
    for a in l:

        print(f"Agent: {a.created} {a.time} {a.color} {a.size}")

    print(" ")

    l.sort(key=lambda minievent: minievent.time, reverse=True)
    print("Sorted List:")
    for a in l:

        print(f"Agent: {a.created} {a.time} {a.color} {a.size}")

    print(" ")

    pop_a = l.pop()
    print(f"Popped agent: {pop_a.created} {pop_a.time}\n")
    print("After popped:")
    for a in l:

        print(f"Agent:{a.created} {a.time} {a.color} {a.size}")

    print(" ")


def main():

    # A little test function to see how Python lists get sorted
    queue_test()


if __name__=='__main__':
    main()