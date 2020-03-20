input = input().split(" ")
grayScale = []
for i in range(len(input)):
    onePix = input[i].split(",")
    a = int((int(onePix[0]) + int(onePix[1]) + int(onePix[2]))/3)
    grayScale.append(a)

