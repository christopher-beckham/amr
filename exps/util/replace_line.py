import sys
filename = sys.argv[1]
query = sys.argv[2]
replace = sys.argv[3]

with open(filename) as f:
  for line in f:
    line = line.rstrip()
    if line.startswith(query):
      line = replace
    print(line)
