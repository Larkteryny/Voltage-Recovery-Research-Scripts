import sys
import re
r"""
with open(r'C:\Users\user\Downloads\pX Sensor Test(33).nox', 'rb') as f:
    data = f.read()
    hex = data.hex()
    print(hex)
    print(hex.find('000000c0ff99c93e'))

    r''' Image extractor
    start = int("0x80b", 16)
    end = int("0x266c", 16)
    f.seek(start)
    with open(r'C:\Users\user\Downloads\what.png', 'wb') as o:
        o.write(f.read(end - start))
    '''
sys.exit()"""

data = []
with open(r"C:\Users\user\Downloads\2025_summer_research_data\raw data\2025_08_22\Whole Cell +1V7200s 1sdischarge 0V2hrshort 8.21 overnight.nox", 'rb') as f:
    data = f.read()

print(data.count(b"NrOfDataPoints"))
hex = data.hex()
#print(hex)
#D601003A02000006
print(len(list(re.finditer(b"[^\x00][\x00\x01]\x00[\x00\x3A][^\x00]\x00\x00\x06", data))))
print(hex.count('e500000008'))

for match in re.finditer(b"Total Number of points to measure".hex(), hex):
    print(hex[match.end():match.end() + 64])
sys.exit()

data = '\n'.join(''.join(chr(x) for x in l if x < 128) for l in data)
print(len(data))

ocpi = data.find("OCP determination", 0)
while ocpi != -1:
    nocpi = data.find("OCP determination", ocpi + 5)
    print(ocpi, data.count('WE(1).Cell', ocpi, nocpi))
    ocpi = nocpi