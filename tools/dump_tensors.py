import struct, pathlib, sys

path = pathlib.Path(sys.argv[1])
data = path.read_bytes()
version, n_tensors, n_kv = struct.unpack_from('<IQQ', data, 4)
pos = 24

def read_string(p):
    l = struct.unpack_from('<Q', data, p)[0]
    p += 8
    return data[p:p+l].decode('utf-8', 'replace'), p + l

for _ in range(n_kv):
    key, pos = read_string(pos)
    vtype = struct.unpack_from('<I', data, pos)[0]
    pos += 4
    if vtype == 8:
        l = struct.unpack_from('<Q', data, pos)[0]
        pos += 8 + l
    elif vtype == 9:
        elem, pos = struct.unpack_from('<I', data, pos)[0], pos + 4
        count, pos = struct.unpack_from('<Q', data, pos)[0], pos + 8
        for _ in range(count):
            if elem == 8:
                l = struct.unpack_from('<Q', data, pos)[0]
                pos += 8 + l
            else:
                sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}[elem]
                pos += sz
    else:
        sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,8:0,9:0,10:8,11:8,12:8}[vtype]
        pos += sz

pos = (pos + 31) // 32 * 32

with open(sys.argv[2], 'w') as f:
    for i in range(n_tensors):
        name, pos = read_string(pos)
        n_dims = struct.unpack_from('<I', data, pos)[0]
        pos += 4
        dims = list(struct.unpack_from('<' + 'Q' * n_dims, data, pos))
        pos += 8 * n_dims
        type_id, offset = struct.unpack_from('<IQ', data, pos)
        pos += 12
        if type_id in (100, 101, 0, 1):
            f.write(f"  {name:55s} type={type_id} dims={dims}\n")

print(f"wrote {n_tensors} tensors to {sys.argv[2]}")
