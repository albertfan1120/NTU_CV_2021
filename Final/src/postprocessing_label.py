
import argparse
import os


def read_txt(filename):
    f = open(filename)
    text = []
    for line in f:
        # remove '\n'
        text.append(int(line[-2]))
    
    return text

def write_txt(filename, label):
    textfile = open(filename, "w+")
    for i in range(len(label)):
        textfile.write("%s\n" %(label[i]))
    textfile.close()



parser = argparse.ArgumentParser()
parser.add_argument('--seq', default = "S5")
args = parser.parse_args() 
seq = args.seq
filepath = "./output/"+seq

allFileList = os.walk(filepath)

pattern1 = [1,1,1,1,0,1,1,1,1]
pattern2 = [0,0,0,0,0,1,1,0,1,0,0,0,0,0]
pattern3 = [0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0]
pattern4 = [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
pattern5 = [1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,1,1,1,1,1,1]
pattern6 = [1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1]

patterns = [pattern1,pattern2,pattern3,pattern4,pattern5,pattern6]
ans = [1,0,0,0,1,1]

# generate smooth pattern

for root, dirs, files in allFileList:
    for file in files:
        if ".txt" in file:
            txt_path = os.path.join(root,file)
            txt = read_txt(txt_path)
            for i,pattern in enumerate(patterns):
                for idx in range(len(txt) - len(pattern) + 1):
                    if txt[idx : idx + len(pattern)] == pattern:
                        # replace with smooth pattern
                        #print("{} has pattern{}".format(txt_path,i+1))
                        txt[idx : idx + len(pattern)] = [ans[i]]*len(pattern)

            write_txt(txt_path,txt)




        
            

