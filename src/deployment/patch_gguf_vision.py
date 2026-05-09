import sys, struct

SRC = sys.argv[1]
DST = sys.argv[2]
KEY = b'clip.has_llava_projector'
BOOL_TYPE = 7

GGUF_FIXED_SIZE = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,9:8,10:8,11:8}

def read_u32(d, pos): return struct.unpack_from('<I', d, pos)[0], pos+4
def read_u64(d, pos): return struct.unpack_from('<Q', d, pos)[0], pos+8

def skip_value(d, pos, vtype):
    if vtype in GGUF_FIXED_SIZE: return pos + GGUF_FIXED_SIZE[vtype]
    if vtype == 8:
        slen, pos = read_u64(d, pos)
        return pos + slen
    if vtype == 12:
        etype, pos = read_u32(d, pos)
        count, pos = read_u64(d, pos)
        for _ in range(count): pos = skip_value(d, pos, etype)
        return pos
    raise ValueError("Unknown type: " + str(vtype))

with open(SRC, 'rb') as f: data = bytearray(f.read())

magic, = struct.unpack_from('<4s', data, 0)
assert magic == b'GGUF'

version, pos = read_u32(data, 4)
tensor_count, pos = read_u64(data, pos)   # pos 8->16
kv_count, pos = read_u32(data, pos)       # pos 16->20
kv_count_offset = 16
pos += 4                                   # skip padding, pos 20->24

print("version=" + str(version) + " tensors=" + str(tensor_count) + " kv=" + str(kv_count) + " first_kv_pos=" + str(pos))

for i in range(kv_count):
    key_len, pos = read_u64(data, pos)
    key = data[pos:pos+key_len].decode('utf-8', 'replace')
    pos += key_len
    vtype, pos = read_u32(data, pos)
    if vtype == 9:
        etype, pos = read_u32(data, pos)
        count, pos = read_u64(data, pos)
        for _ in range(count): pos = skip_value(data, pos, etype)
    else:
        pos = skip_value(data, pos, vtype)
kv_end = pos

ti_pos = kv_end
for i in range(tensor_count):
    name_len, ti_pos = read_u64(data, ti_pos)
    ti_pos += name_len
    n_dims, ti_pos = read_u32(data, ti_pos)
    ti_pos += n_dims * 8
    ti_pos += 4
    ti_pos += 8
ti_size = ti_pos - kv_end

ALIGN = 32
old_data_offset = (kv_end + ti_size + ALIGN - 1) & ~(ALIGN - 1)
new_kv = struct.pack('<Q', len(KEY)) + KEY + struct.pack('<I', BOOL_TYPE) + struct.pack('B', 1)
assert len(new_kv) == 37
new_kv_end = kv_end + 37
new_data_offset = (new_kv_end + ti_size + ALIGN - 1) & ~(ALIGN - 1)

print("kv_end=" + str(kv_end) + " ti_size=" + str(ti_size))
print("old_data_offset=" + str(old_data_offset) + " new_data_offset=" + str(new_data_offset))

# tensor offsets are RELATIVE to data_offset --- do NOT shift them
# just copy tensor info block verbatim
ti_block = data[kv_end:kv_end + ti_size]

struct.pack_into('<I', data, kv_count_offset, kv_count + 1)

new_padding = new_data_offset - new_kv_end - ti_size
new_data = data[:kv_end] + new_kv + ti_block + b'\x00' * new_padding + data[old_data_offset:]

with open(DST, 'wb') as f: f.write(new_data)
print("Done " + str(len(new_data)) + " bytes was " + str(len(data)))
print("kv_count " + str(kv_count) + " -> " + str(kv_count + 1))