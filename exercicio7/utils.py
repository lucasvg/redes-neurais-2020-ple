def ParseBoolInput(a):
    if(a == "Y"):
        return True
    if(a == "N"):
        return False
    return None

def ParseLayerNeuronFunction(a):
    a = a.split(" ")
    if(len(a)!= 3):
        return None
    return tuple([int(i) for i in a])

def GetInput(msg, parseResult):
    result = None
    while(result is None):
        print(msg)
        result = parseResult(input())